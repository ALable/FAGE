"""
FAGE — SubjectAdapter: Per-User Subject Appearance Preservation (LoRA version)

Architecture separates two concerns:
  GazeControlNet (shared, ~3M, frozen after Phase 1)
    └─ handles gaze redirection via AdaLN (gaze_emb → scale/shift/gate per UNetBlock)

  SubjectAdapter (per-user, ~400K, fine-tuned per new subject)
    ├─ SubjectEncoder:    source_eye → z_s [B, subject_dim]
    └─ BlockLoRAHeads × N: z_s → (A, B) low-rank matrices per UNetBlock
         ΔW = A @ B  (rank-r decomposition of conv1 weight perturbation)

LoRA injection (applied inside each UNetBlock.conv1):
    W_eff = W_frozen + A @ B          (ΔW zero-init → identity at start)
    out   = F.conv2d(x, W_eff, ...)

Design properties:
  - SubjectAdapter input: source eye ONLY (no gaze) → pure appearance
  - Zero-init B matrices → ΔW=0 at init (stable Phase 1 → 2 transfer)
  - Encode reference frames once per subject, cache LoRA deltas → fast inference
  - More expressive than FiLM: modifies the feature transformation itself,
    not just per-channel scale/shift

Fast inference for new subjects:
    z_s    = adapter.encode(ref_frames).mean(0, keepdim=True)  # run once
    cached = adapter.get_lora_deltas(z_s)                      # run once
    out    = unet(eye, gaze_cond, subject_mods=cached)         # per-frame
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SubjectEncoder(nn.Module):
    """Lightweight appearance encoder: source_eye → subject embedding.

    4-stage conv (stride-2 × 3) + GAP + Linear + LayerNorm.
    Input:  [B, C_in, H, W]   (default C_in=6, H≈64-80, W≈128-160)
    Output: [B, subject_dim]

    ~120K params (C_in=6, subject_dim=128)
    """
    def __init__(self, in_channels: int = 6, subject_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),    # [B, 16, H, W]
            nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),   # [B, 32, H/2, W/2]
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),   # [B, 64, H/4, W/4]
            nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # [B, 128, H/8, W/8]
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),                      # [B, 128, 1, 1]
            nn.Flatten(),                                  # [B, 128]
            nn.Linear(128, subject_dim),
            nn.LayerNorm(subject_dim),
        )

    def forward(self, source_eye: torch.Tensor) -> torch.Tensor:
        return self.net(source_eye)


class BlockLoRAHead(nn.Module):
    """Per-block LoRA head: shared_hidden → (A, B) for one UNetBlock's conv1.

    Takes pre-computed shared hidden features (from SubjectAdapter.shared_trunk),
    outputs ΔW = A @ B for conv1 weight perturbation.

    conv1 weight shape: [C_out, C_in, k, k]
    LoRA:  A [C_out, rank] @ B [rank, C_in*k*k]  → ΔW [C_out, C_in, k, k]
    Zero-init B → ΔW=0 at initialization.
    """
    def __init__(self, hidden_dim: int, c_out: int, c_in: int, k: int = 3, rank: int = 4):
        super().__init__()
        self.c_out = c_out
        self.c_in = c_in
        self.k = k
        self.rank = rank

        self.head_A = nn.Linear(hidden_dim, c_out * rank, bias=True)
        self.head_B = nn.Linear(hidden_dim, rank * c_in * k * k, bias=True)

        nn.init.zeros_(self.head_B.weight)
        nn.init.zeros_(self.head_B.bias)
        nn.init.normal_(self.head_A.weight, std=0.02)
        nn.init.zeros_(self.head_A.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: [B, hidden_dim]  shared trunk features
        Returns:
            delta_W: [B, C_out, C_in, k, k]
        """
        B = h.shape[0]
        A = self.head_A(h).view(B, self.c_out, self.rank)
        Bm = self.head_B(h).view(B, self.rank, self.c_in * self.k * self.k)
        return torch.bmm(A, Bm).view(B, self.c_out, self.c_in, self.k, self.k)


class SubjectAdapter(nn.Module):
    """Per-user subject appearance adapter using dynamic LoRA (~400K params).

    Generates per-block LoRA weight deltas from source eye appearance.
    These are applied to GazeControlNet's UNetBlock.conv1 weights to preserve
    subject-specific appearance while gaze direction is controlled independently.

    vs FiLM: LoRA modifies the feature transformation itself (ΔW on conv weights),
    enabling finer-grained appearance control (iris texture, eyelash detail, etc.)
    rather than just per-channel scale/shift.

    Training:
        Phase 1: Trained jointly with GazeControlNet (all params trainable)
        Phase 2: Only SubjectAdapter fine-tuned for new users (300-1000 steps)

    Inference:
        Fast path — encode reference frames once, cache LoRA deltas:
            z_s    = adapter.encode(ref_eye)          # [1, subject_dim]
            cached = adapter.get_lora_deltas(z_s)     # list of N delta_W
            out    = unet(eye, gaze, subject_mods=cached)
    """
    def __init__(
        self,
        in_channels: int = 6,
        subject_dim: int = 128,
        block_channels=None,
        lora_rank: int = 4,
    ):
        """
        Args:
            in_channels:    eye crop input channels (6 = left+right concat)
            subject_dim:    subject embedding dimension (default 128)
            block_channels: list[int], out_channels per UNetBlock in traversal order
                            (enc0 → enc1 → latent → dec0 → dec1).
                            Default matches EyeOnlyGazeDiC depth=[2,2,4,2,2], hidden=32.
            lora_rank:      rank r for LoRA decomposition (default 4)
        """
        super().__init__()
        self.subject_dim = subject_dim
        self.lora_rank = lora_rank

        if block_channels is None:
            # EyeOnlyGazeDiC: depth=[2,2,4,2,2], mult=[1,2,4,2,1], hidden_size=32
            block_channels = [32] * 2 + [64] * 2 + [128] * 4 + [64] * 2 + [32] * 2

        # ── Appearance encoder ──────────────────────────────────────────────
        self.encoder = SubjectEncoder(in_channels, subject_dim)

        # ── Shared trunk: z_s → shared hidden features ──────────────────────
        # All block heads share this trunk to reduce param count
        self.trunk_hidden = max(subject_dim // 2, 32)
        self.shared_trunk = nn.Sequential(
            nn.SiLU(),
            nn.Linear(subject_dim, self.trunk_hidden, bias=True),
            nn.SiLU(),
        )

        # ── Per-block LoRA heads ────────────────────────────────────────────
        # conv1 in each UNetBlock: [C_out, C_out, 3, 3]  (same in/out after conv0)
        # We target conv1 (post-norm, post-gaze-modulation conv)
        # Rank scales with channel count to keep param budget reasonable:
        #   ch=32  → rank=lora_rank      (e.g. 4)
        #   ch=64  → rank=lora_rank//2   (e.g. 2)
        #   ch=128 → rank=max(1, lora_rank//4)  (e.g. 1)
        self.block_lora_heads = nn.ModuleList()
        base_ch = min(block_channels)
        for ch in block_channels:
            scale = ch // base_ch          # 1, 2, or 4
            rank = max(1, lora_rank // scale)
            self.block_lora_heads.append(
                BlockLoRAHead(self.trunk_hidden, c_out=ch, c_in=ch, k=3, rank=rank)
            )

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

    def get_lora_deltas(self, subject_emb: torch.Tensor):
        """Generate per-block LoRA weight deltas from subject embedding.

        Result can be cached when subject_emb is fixed (e.g., per-user inference).

        Args:
            subject_emb: [B, subject_dim]
        Returns:
            deltas: list of delta_W tensors, each [B, C_out, C_in, k, k]
        """
        h = self.shared_trunk(subject_emb)   # [B, trunk_hidden]
        return [head(h) for head in self.block_lora_heads]

    def forward(self, source_eye: torch.Tensor):
        """Encode source eye and return per-block LoRA deltas.

        Args:
            source_eye: [B, C_in, H, W]
        Returns:
            deltas: list of delta_W tensors, each [B, C_out, C_in, k, k]
        """
        return self.get_lora_deltas(self.encode(source_eye))

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
