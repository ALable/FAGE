"""
FAGE — SubjectAdapter: Per-User Subject Appearance Preservation

Architecture separates two concerns:
  GazeControlNet (shared, ~3M, frozen after Phase 1)
    └─ handles gaze redirection via AdaLN (gaze_emb → scale/shift/gate per UNetBlock)

  SubjectAdapter (per-user, ~430K, fine-tuned per new subject)
    ├─ SubjectEncoder:    source_eye → z_s [B, subject_dim]
    │    Uses UNetBlock-style EncoderBlocks (GroupNorm + Conv + complementary-gate skip)
    │    with Downsample layers for multi-scale feature extraction.
    └─ BlockModulators:  z_s → FiLM (scale_i, shift_i) applied in each UNetBlock

FiLM injection (applied AFTER gaze gate-residual in each UNetBlock):
    x_out = x_gaze * (1 + scale_i) + shift_i

Design properties:
  - SubjectAdapter input: source eye ONLY (no gaze) → pure appearance
  - EncoderBlocks share the same GroupNorm+Conv design language as UNetBlock
  - Residual connections via complementary gate: gate*x + (1-gate)*skip
  - Zero-init BlockModulators → identity at init (stable Phase 1 → 2 transfer)
  - Encode reference frames once per subject, cache modulations → fast inference

Fast inference for new subjects:
    z_s    = adapter.encode(ref_frames).mean(0, keepdim=True)  # run once
    cached = adapter.get_modulations(z_s)                      # run once
    out    = unet(eye, gaze_cond, subject_mods=cached)         # per-frame
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.gaze_dic import GroupNorm, Downsample


class AttentionPool(nn.Module):
    """优化后的注意力池化：空间-通道联合动态路由"""
    def __init__(self, channels: int):
        super().__init__()
        # 使用更深层的 MLP 提取注意力图
        self.attn = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=1),
            nn.GELU(),
            # 输出与 channels 数量一致的权重图，实现 per-channel 空间权重
            nn.Conv2d(channels // 2, channels, kernel_size=1) 
        )
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, C, H, W]
        B, C, H, W = x.shape
        # w: [B, C, H*W]
        w = self.attn(x).flatten(2)
        # 在空间维度上执行 Softmax
        w = F.softmax(w, dim=-1)
        
        # x_flat: [B, C, H*W]
        x_flat = x.flatten(2)
        
        # 逐通道进行空间加权求和: [B, C]
        pooled = (x_flat * w).sum(dim=-1) 
        
        return self.norm(pooled)

class EncoderBlock(nn.Module):
    """Unconditional UNetBlock-style block for SubjectEncoder.

    Mirrors UNetBlock's structure (GroupNorm + Conv + complementary-gate residual)
    but removes the gaze affine branch — pure appearance feature extraction.

    Residual: gate * conv_out + (1 - gate) * skip
    where gate is learned from conv_out, ensuring bounded output.
    """
    def __init__(self, in_channels: int, out_channels: int, num_groups: int = 32, min_channels: int = 4):
        super().__init__()
        # min_channels=1 on norm0 to handle small in_channels (e.g. 3 for RGB input)
        self.norm0 = GroupNorm(in_channels, num_groups, min_channels_per_group=1, affine=False)
        self.conv0 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = GroupNorm(out_channels, num_groups, min_channels)
        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        # Gate: scalar per channel, sigmoid → complementary weighting
        self.gate_proj = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        # Skip projection for channel mismatch
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x) if self.skip is not None else x

        # Conv first when in_channels is too small for GroupNorm (e.g. 3 < min_channels*groups)
        h = self.conv0(self.act(self.norm0(x)))
        h = self.conv1(self.act(self.norm1(h)))

        # Complementary gate residual (same design as UNetBlock)
        gate = torch.sigmoid(self.gate_proj(h))
        return gate * h + (1 - gate) * residual


class SubjectEncoder(nn.Module):
    """Multi-scale appearance encoder using UNetBlock-style EncoderBlocks.

    3-level encoder matching EyeOnlyGazeDiC channel schedule (hidden=32):
        Level 0: [B, C_in, H,   W  ] → [B, 32, H,   W  ]  (1 block)
        Down  0: [B, 32,   H,   W  ] → [B, 64, H/2, W/2]
        Level 1: [B, 64,   H/2, W/2] → [B, 64, H/2, W/2]  (1 block)
        Down  1: [B, 64,   H/2, W/2] → [B, 64, H/4, W/4]  (64 not 128, saves params)
        Level 2: [B, 64,   H/4, W/4] → [B, 64, H/4, W/4]  (1 block)
        AttentionPool + Linear + LayerNorm → [B, subject_dim]

    Input:  [B, C_in, H, W]  (default C_in=3, H≈80, W≈160 for width-concat)
    Output: [B, subject_dim]
    """
    def __init__(self, in_channels: int = 3, subject_dim: int = 128):
        super().__init__()
        self.level0 = EncoderBlock(in_channels, 32)
        self.down0 = Downsample(32, 64)

        self.level1 = EncoderBlock(64, 64)
        self.down1 = Downsample(64, 64)   # 64→64 (not 128) to keep params ~150K

        self.level2 = EncoderBlock(64, 64)

        self.pool = AttentionPool(64)
        self.proj = nn.Sequential(
            nn.Linear(64, subject_dim),
            nn.LayerNorm(subject_dim),
        )

    def forward(self, source_eye: torch.Tensor) -> torch.Tensor:
        x = self.level0(source_eye)
        x = self.down0(x)
        x = self.level1(x)
        x = self.down1(x)
        x = self.level2(x)
        x = self.pool(x)       # [B, 64]
        return self.proj(x)    # [B, subject_dim]


class SubjectAdapter(nn.Module):
    """Per-user subject appearance adapter (~430K params).

    Generates per-block FiLM modulations from source eye appearance.
    These modulate GazeControlNet's UNet blocks to preserve subject-specific
    appearance while gaze direction is controlled independently by gaze_emb.

    SubjectEncoder uses UNetBlock-style EncoderBlocks with residual connections,
    sharing the same design language (GroupNorm + Conv + complementary gate) as
    the main UNet backbone.

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


class GazeAwareSubjectAdapter(nn.Module):
    """Subject adapter that conditions personalization on the target gaze.

    The original SubjectAdapter is appearance-only: it encodes a source eye crop
    and emits per-block FiLM parameters. This variant keeps the same zero-init
    FiLM contract but builds the modulation vector from subject appearance,
    target head embedding, target gaze embedding, and their interactions.

    This makes the adapter a compact research path for personalized gaze
    redirection: per-user eye geometry can influence how strongly each UNet
    block should rotate iris/eyelid features for a specific target gaze.
    """

    requires_gaze_condition = True

    def __init__(
        self,
        in_channels: int = 3,
        subject_dim: int = 128,
        gaze_dim: int = 64,
        hidden_dim: int = 128,
        block_channels=None,
        dropout: float = 0.0,
    ):
        """
        Args:
            in_channels: eye crop input channels.
            subject_dim: output dimension of the subject encoder.
            gaze_dim: dimension of head/gaze embeddings from GazeMLP.
            hidden_dim: shared modulation dimension.
            block_channels: output channels for each UNetBlock.
            dropout: dropout in the subject-gaze fusion MLP.
        """
        super().__init__()
        self.subject_dim = subject_dim
        self.gaze_dim = gaze_dim
        self.hidden_dim = hidden_dim

        if block_channels is None:
            block_channels = [32] * 2 + [64] * 2 + [128] * 4 + [64] * 2 + [32] * 2

        self.encoder = SubjectEncoder(in_channels, subject_dim)

        self.subject_proj = nn.Sequential(
            nn.Linear(subject_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.head_proj = nn.Sequential(
            nn.Linear(gaze_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.gaze_proj = nn.Sequential(
            nn.Linear(gaze_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.delta_proj = nn.Sequential(
            nn.Linear(gaze_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        fusion_dim = hidden_dim * 6
        self.context_mlp = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.context_gate = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.context_norm = nn.LayerNorm(hidden_dim)

        self.block_modulators = nn.ModuleList()
        for ch in block_channels:
            head = nn.Linear(hidden_dim, ch * 2, bias=True)
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
            self.block_modulators.append(head)

    @staticmethod
    def _split_gaze_condition(gaze_condition: torch.Tensor):
        """Return head and gaze embeddings from [B, 2, D] or [B, D]."""
        if gaze_condition.dim() == 3 and gaze_condition.shape[1] == 2:
            return gaze_condition[:, 0, :], gaze_condition[:, 1, :]
        if gaze_condition.dim() == 2:
            return torch.zeros_like(gaze_condition), gaze_condition
        raise ValueError(
            "gaze_condition must have shape [B, 2, gaze_dim] or [B, gaze_dim]"
        )

    def encode(self, source_eye: torch.Tensor) -> torch.Tensor:
        """Encode one or more reference eye crops into a subject embedding.

        Args:
            source_eye: [B, C, H, W] or [B, R, C, H, W].
        Returns:
            subject_emb: [B, subject_dim].
        """
        if source_eye.dim() == 5:
            bsz, refs, channels, height, width = source_eye.shape
            flat = source_eye.reshape(bsz * refs, channels, height, width)
            emb = self.encoder(flat).reshape(bsz, refs, -1)
            return emb.mean(dim=1)
        return self.encoder(source_eye)

    def fuse_context(
        self,
        subject_emb: torch.Tensor,
        gaze_condition: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse subject and target gaze context into a modulation vector."""
        head_emb, gaze_emb = self._split_gaze_condition(gaze_condition)
        subject_feat = self.subject_proj(subject_emb)
        head_feat = self.head_proj(head_emb)
        gaze_feat = self.gaze_proj(gaze_emb)
        delta_feat = self.delta_proj(gaze_emb - head_emb)

        fusion = torch.cat(
            [
                subject_feat,
                head_feat,
                gaze_feat,
                delta_feat,
                subject_feat * gaze_feat,
                subject_feat * delta_feat,
            ],
            dim=-1,
        )
        candidate = self.context_mlp(fusion)
        gate = self.context_gate(fusion)
        return self.context_norm(subject_feat + gate * candidate)

    def get_modulations(
        self,
        subject_emb: torch.Tensor,
        gaze_condition: torch.Tensor,
    ):
        """Generate per-block FiLM parameters conditioned on subject and gaze."""
        z = self.fuse_context(subject_emb, gaze_condition)
        mods = []
        for head in self.block_modulators:
            params = head(z)
            scale, shift = params.chunk(2, dim=1)
            mods.append((
                scale.unsqueeze(-1).unsqueeze(-1),
                shift.unsqueeze(-1).unsqueeze(-1),
            ))
        return mods

    def forward(self, source_eye: torch.Tensor, gaze_condition: torch.Tensor):
        """Return gaze-aware per-block FiLM modulations."""
        return self.get_modulations(self.encode(source_eye), gaze_condition)

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
