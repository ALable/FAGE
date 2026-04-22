"""
FAGE — SubjectAdapter: Per-User Subject Appearance Preservation

Architecture separates two concerns:
  GazeControlNet (shared, ~3M, frozen after Phase 1)
    └─ handles gaze redirection via AdaLN (gaze_emb → scale/shift/gate per UNetBlock)

  SubjectAdapter (per-user, ~350K, fine-tuned per new subject)
    ├─ SubjectEncoder:    source_eye → z_s [B, subject_dim]
    └─ BlockModulators:  z_s → FiLM (scale_i, shift_i) applied in each UNetBlock

FiLM injection (applied AFTER gaze gate-residual in each UNetBlock):
    x_out = x_gaze * (1 + scale_i) + shift_i

Design properties:
  - SubjectAdapter input: source eye ONLY (no gaze) → pure appearance
  - Zero-init BlockModulators → identity at init (stable Phase 1 → 2 transfer)
  - Encode reference frames once per subject, cache modulations → fast inference

Fast inference for new subjects:
    z_s    = adapter.encode(ref_frames).mean(0, keepdim=True)  # run once
    cached = adapter.get_modulations(z_s)                      # run once
    out    = unet(eye, gaze_cond, subject_mods=cached)         # per-frame
"""
import torch
import torch.nn as nn


class AttentionPool(nn.Module):
    """Attention pooling: learns where to look instead of averaging spatially.

    Replaces AdaptiveAvgPool2d(1) to preserve fine-grained texture (iris, vessels).
    Only adds `channels` parameters (1×1 conv weight map).
    """
    def __init__(self, channels: int):
        super().__init__()
        self.attn = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # [B, C, H, W]
        w = self.attn(x).flatten(2).softmax(-1)            # [B, 1, H*W]
        return (x.flatten(2) * w).sum(-1)                  # [B, C]


class SubjectEncoder(nn.Module):
    """Lightweight appearance encoder: source_eye → subject embedding.

    4-stage conv (stride-2 × 3) + AttentionPool + Linear + LayerNorm.
    Input:  [B, C_in, H, W]   (default C_in=3, H≈80, W≈160 for width-concat)
    Output: [B, subject_dim]

    ~120K params (C_in=3, subject_dim=128)
    """
    def __init__(self, in_channels: int = 3, subject_dim: int = 128):
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),    # [B, 16, H, W]
            nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),   # [B, 32, H/2, W/2]
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),   # [B, 64, H/4, W/4]
            nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # [B, 128, H/8, W/8]
            nn.GELU(),
        )
        self.pool = AttentionPool(128)
        self.proj = nn.Sequential(
            nn.Linear(128, subject_dim),
            nn.LayerNorm(subject_dim),
        )

    def forward(self, source_eye: torch.Tensor) -> torch.Tensor:
        x = self.conv_net(source_eye)   # [B, 128, H/8, W/8]
        x = self.pool(x)                # [B, 128]
        return self.proj(x)             # [B, subject_dim]


class SubjectAdapter(nn.Module):
    """Per-user subject appearance adapter (~350K params).

    Generates per-block FiLM modulations from source eye appearance.
    These modulate GazeControlNet's UNet blocks to preserve subject-specific
    appearance while gaze direction is controlled independently by gaze_emb.

    Training:
        Phase 1: Trained jointly with GazeControlNet (all params trainable)
        Phase 2: Only SubjectAdapter fine-tuned for new users (300-1000 steps)

    Inference:
        Fast path — encode reference frames once, cache modulations:
            z_s    = adapter.encode(ref_eye)        # [1, subject_dim]
            cached = adapter.get_modulations(z_s)   # list of 12 (scale, shift)
            out    = unet(eye, gaze, subject_mods=cached)  # no adapter call
    """
    def __init__(
        self,
        in_channels: int = 3,
        subject_dim: int = 128,
        block_channels=None,
    ):
        """
        Args:
            in_channels:    eye crop input channels (3 = width-concat RGB)
            subject_dim:    subject embedding dimension (default 128)
            block_channels: list[int], out_channels per UNetBlock in traversal order
                            (enc0 → enc1 → latent → dec0 → dec1).
                            Default matches EyeOnlyGazeDiC depth=[2,2,4,2,2], hidden=32.
        """
        super().__init__()
        self.subject_dim = subject_dim

        if block_channels is None:
            # EyeOnlyGazeDiC: depth=[2,2,4,2,2], mult=[1,2,4,2,1], hidden_size=32
            block_channels = [32] * 2 + [64] * 2 + [128] * 4 + [64] * 2 + [32] * 2

        # ── Appearance encoder ──────────────────────────────────────────────
        self.encoder = SubjectEncoder(in_channels, subject_dim)

        # ── Shared projection + per-block FiLM heads ───────────────────────
        # Shared: z_s → z_shared (subject_dim → subject_dim//2), cross-block regularization
        # Heads:  z_shared → (scale, shift) per block, no extra activation needed
        # Zero-init heads → identity transformation at start
        shared_dim = subject_dim // 2
        self.shared_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(subject_dim, shared_dim, bias=True),
        )
        self.block_modulators = nn.ModuleList()
        for ch in block_channels:
            head = nn.Linear(shared_dim, ch * 2, bias=True)
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
            self.block_modulators.append(head)

    # ── API ─────────────────────────────────────────────────────────────────

    def encode(self, source_eye: torch.Tensor) -> torch.Tensor:
        """Encode source eye → subject embedding.

        Call once per subject (or average N reference frames) and cache:
            z_s = adapter.encode(ref_frames).mean(0, keepdim=True)

        Args:
            source_eye: [B, C_in, H, W]
        Returns:
            subject_emb: [B, subject_dim]
        """
        return self.encoder(source_eye)

    def get_modulations(self, subject_emb: torch.Tensor):
        """Generate per-block FiLM (scale, shift) from subject embedding.

        Result can be cached when subject_emb is fixed (e.g., per-user inference).

        Args:
            subject_emb: [B, subject_dim]
        Returns:
            mods: list of (scale, shift) tuples, each shaped [B, C_block, 1, 1]
        """
        z = self.shared_proj(subject_emb)   # [B, subject_dim//2] — shared across blocks
        mods = []
        for head in self.block_modulators:
            params = head(z)                             # [B, 2×C]
            scale, shift = params.chunk(2, dim=1)
            mods.append((
                scale.unsqueeze(-1).unsqueeze(-1),       # [B, C, 1, 1]
                shift.unsqueeze(-1).unsqueeze(-1),
            ))
        return mods

    def forward(self, source_eye: torch.Tensor):
        """Encode source eye and return per-block FiLM modulations.

        Args:
            source_eye: [B, C_in, H, W]
        Returns:
            mods: list of (scale, shift) tuples, each [B, C_block, 1, 1]
        """
        return self.get_modulations(self.encode(source_eye))

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
