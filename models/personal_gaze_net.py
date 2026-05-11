"""
PersonalGazeNet: Personalized Gaze Redirection via Spatial Identity Tokens
and Gaze-Guided Warping

Overview
--------
Personalized gaze redirection requires two orthogonal capabilities:
  (a) Appearance preservation  — retaining the subject's iris colour, eyelid
      shape, and skin texture regardless of gaze direction.
  (b) Geometric redirection    — accurately moving the iris/pupil to the
      target viewing direction.

Existing EyeOnlyWrapper (FiLM-based SubjectAdapter) conflates both concerns
through a shared global modulation vector, limiting per-pixel specificity.

PersonalGazeNet decouples them via four key innovations:

  1. IdentityEncoder + Multi-Reference Aggregation
     A gaze-unconditional CNN encoder processes N_ref reference frames from
     the *same* subject. Features are mean-pooled across frames, producing
     spatially-structured identity maps at every encoder scale.
     This gives robust, gaze-invariant appearance tokens even when individual
     frames are partially occluded or vary in illumination.

  2. GazeDeltaWarpModule
     A lightweight MLP-CNN flow predictor takes the *relative* gaze delta
     (Δpitch, Δyaw) and outputs a dense 2-D displacement field at the
     bottleneck resolution. Applying this field via grid_sample provides an
     explicit geometric prior for iris movement *before* appearance synthesis
     begins, separating geometric and photometric learning objectives.

  3. Cross-Attention Bottleneck
     After warping, the source latent features attend to identity tokens from
     the reference frames via scaled dot-product cross-attention. Unlike
     global FiLM, this is *spatially aware*: iris-region queries primarily
     attend to iris-region identity keys, naturally preserving fine-grained
     appearance at the correct locations.

  4. Identity-FiLM in Decoder
     A global identity embedding (avg-pooled from the identity latent) drives
     FiLM (scale + shift) modulations in each decoder UNetBlock via the
     existing subject_scale/subject_shift API — zero-initialised for
     training stability and compatible with SubjectAdapter-style fine-tuning.

Two-Phase Training
------------------
  Phase 1 (N_ref=1, reference=source):
      All parameters trained jointly. Cross-attention is self-attention
      (trivially satisfied), providing stable initialisation.

  Phase 2 (N_ref≥2, held-out subject frames):
      IdentityEncoder + IdentityAggregator fine-tuned; backbone frozen.
      Requires only 300-1000 steps per new subject.

Shape conventions (same as EyeOnlyWrapper):
  eye crops  : [B, 3, H, W*2]  (left | right horizontally concatenated)
  gaze_cond  : [B, 2, gaze_dim] ([:,0,:]=head_emb  [:,1,:]=gaze_emb)
  ref_eyes   : [B, N_ref, 3, H, W*2]   (N reference frames)
  raw gaze   : [B, 2]  pitch / yaw in radians (for warp module)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.gaze_dic import (
    GroupNorm, OverlapPatchEmbed, UNetBlock, Downsample, Upsample,
    DiCFinalLayer, ConditionEmbedder,
)
from models.subject_adapter import EncoderBlock


# ---------------------------------------------------------------------------
# 1. Multi-Head Cross-Attention
# ---------------------------------------------------------------------------

class CrossAttention(nn.Module):
    """Spatial cross-attention: source latent queries identity key-values.

    Pre-norm on both branches; output projection is zero-initialised so that
    at initialisation the module is an identity (residual = 0).

    Args:
        q_dim   : channel dimension of query (source latent)
        kv_dim  : channel dimension of key/value (identity features)
        num_heads: attention heads (q_dim must be divisible by num_heads)
        dropout : attention dropout probability during training
    """

    def __init__(self, q_dim: int, kv_dim: int, num_heads: int = 4,
                 dropout: float = 0.0):
        super().__init__()
        assert q_dim % num_heads == 0, (
            f"q_dim ({q_dim}) must be divisible by num_heads ({num_heads})"
        )
        self.num_heads = num_heads
        self.head_dim  = q_dim // num_heads
        self.dropout   = dropout

        self.norm_q  = nn.LayerNorm(q_dim)
        self.norm_kv = nn.LayerNorm(kv_dim)

        self.q_proj  = nn.Linear(q_dim,  q_dim, bias=False)
        self.k_proj  = nn.Linear(kv_dim, q_dim, bias=False)
        self.v_proj  = nn.Linear(kv_dim, q_dim, bias=False)
        self.out_proj = nn.Linear(q_dim, q_dim)

        # Zero-init output projection → identity residual at init
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    # ------------------------------------------------------------------
    def forward(self, q_feat: torch.Tensor,
                kv_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            q_feat  : [B, N_q,  q_dim]  — flattened source spatial features
            kv_feat : [B, N_kv, kv_dim] — flattened identity features
        Returns:
            [B, N_q, q_dim]  (residual increment; caller adds to q_feat)
        """
        B, N_q, _ = q_feat.shape

        Q = self.q_proj(self.norm_q(q_feat))   # [B, N_q,  q_dim]
        K = self.k_proj(self.norm_kv(kv_feat)) # [B, N_kv, q_dim]
        V = self.v_proj(kv_feat)               # [B, N_kv, q_dim]

        # Reshape for multi-head attention
        def split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        Q, K, V = split_heads(Q), split_heads(K), split_heads(V)

        attn_out = F.scaled_dot_product_attention(
            Q, K, V,
            dropout_p=self.dropout if self.training else 0.0,
        )  # [B, num_heads, N_q, head_dim]

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, N_q, -1)
        return self.out_proj(attn_out)  # [B, N_q, q_dim]


# ---------------------------------------------------------------------------
# 2. Identity Encoder (gaze-unconditional)
# ---------------------------------------------------------------------------

class IdentityEncoder(nn.Module):
    """Gaze-unconditional CNN encoder for extracting appearance features.

    Uses EncoderBlock (from SubjectAdapter) which mirrors UNetBlock design
    language (GroupNorm + GELU + Conv + complementary-gate residual) but
    *without* any gaze-conditioning affine transform.

    This forces the identity representation to be gaze-invariant — the model
    cannot use gaze direction as a shortcut for appearance encoding.

    Output feature maps are at three scales, mirroring EyeOnlyGazeDiC:
        feat_0  : [B, C0, H,   W  ]
        feat_1  : [B, C1, H/2, W/2]
        feat_lat: [B, C2, H/4, W/4]
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_size: int = 32,
        mult_channels: list = None,
        depth: list = None,
    ):
        super().__init__()
        if mult_channels is None:
            mult_channels = [1, 2, 4]
        if depth is None:
            depth = [2, 2, 4]

        ch = [hidden_size * m for m in mult_channels]  # [C0, C1, C2]
        self.channels = ch

        # Initial patch embedding (same as EyeOnlyGazeDiC)
        self.x_embedder = OverlapPatchEmbed(3, 1, in_channels, ch[0])

        # Encoder stages — unconditional residual blocks
        self.enc0 = nn.ModuleList(
            [EncoderBlock(ch[0], ch[0]) for _ in range(depth[0])]
        )
        self.down0 = Downsample(ch[0], ch[1])

        self.enc1 = nn.ModuleList(
            [EncoderBlock(ch[1], ch[1]) for _ in range(depth[1])]
        )
        self.down1 = Downsample(ch[1], ch[2])

        self.lat = nn.ModuleList(
            [EncoderBlock(ch[2], ch[2]) for _ in range(depth[2])]
        )

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, 3, H, W]
        Returns:
            feat_0  : [B, C0, H,   W  ]
            feat_1  : [B, C1, H/2, W/2]
            feat_lat: [B, C2, H/4, W/4]
        """
        x = self.x_embedder(x)
        for blk in self.enc0:
            x = blk(x)
        feat_0 = x

        x = self.down0(x)
        for blk in self.enc1:
            x = blk(x)
        feat_1 = x

        x = self.down1(x)
        for blk in self.lat:
            x = blk(x)
        feat_lat = x

        return feat_0, feat_1, feat_lat


# ---------------------------------------------------------------------------
# 3. Identity Aggregator
# ---------------------------------------------------------------------------

class IdentityAggregator(nn.Module):
    """Aggregate multi-reference identity features into tokens for conditioning.

    Given N_ref sets of identity feature maps (at three scales), this module
    produces:
      - ``id_tokens_lat`` : spatially-structured tokens at latent scale
                           used as Key/Value in cross-attention bottleneck
      - ``id_emb``        : global identity embedding (for FiLM in decoder)

    Aggregation strategy: mean-pool across N_ref (simple & robust; can be
    replaced with learned attention-pool for future work).

    The global embedding is computed by average-pooling ``id_tokens_lat``
    and projecting through a small MLP.
    """

    def __init__(self, lat_channels: int, id_emb_dim: int = 128):
        super().__init__()
        self.id_emb_dim = id_emb_dim

        # Project id_emb_dim from latent channels
        self.global_pool_proj = nn.Sequential(
            nn.Linear(lat_channels, id_emb_dim),
            nn.LayerNorm(id_emb_dim),
            nn.SiLU(),
            nn.Linear(id_emb_dim, id_emb_dim),
            nn.LayerNorm(id_emb_dim),
        )

    # ------------------------------------------------------------------
    def forward(
        self,
        feats_0_list:   list,  # list of N_ref tensors [B, C0, H,   W  ]
        feats_1_list:   list,  # list of N_ref tensors [B, C1, H/2, W/2]
        feats_lat_list: list,  # list of N_ref tensors [B, C2, H/4, W/4]
    ):
        """
        Returns:
            id_feat_0  : [B, C0, H,   W  ]  — mean-pooled identity, scale 0
            id_feat_1  : [B, C1, H/2, W/2]  — mean-pooled identity, scale 1
            id_feat_lat: [B, C2, H/4, W/4]  — mean-pooled identity, latent
            id_emb     : [B, id_emb_dim]    — global identity embedding
        """
        def _mean_pool(feat_list):
            return torch.stack(feat_list, dim=1).mean(dim=1)

        id_feat_0   = _mean_pool(feats_0_list)
        id_feat_1   = _mean_pool(feats_1_list)
        id_feat_lat = _mean_pool(feats_lat_list)

        # Global identity embedding: avg-pool spatial dims → MLP
        pooled = id_feat_lat.mean(dim=(-2, -1))   # [B, C2]
        id_emb = self.global_pool_proj(pooled)     # [B, id_emb_dim]

        return id_feat_0, id_feat_1, id_feat_lat, id_emb


# ---------------------------------------------------------------------------
# 4. Gaze-Delta Warp Module
# ---------------------------------------------------------------------------

class GazeDeltaWarpModule(nn.Module):
    """Predict a dense 2-D flow field from relative gaze change.

    The *relative* gaze delta (Δpitch, Δyaw) encodes how much the eyeball
    rotates.  This module learns a subject-agnostic mapping from gaze delta
    to a dense pixel-displacement field at the latent (coarsest) resolution.

    Applying this field via bilinear grid_sample provides an explicit
    geometric prior for iris displacement *before* appearance synthesis in
    the decoder — decoupling geometric and photometric learning objectives.

    Architecture:
        delta [B, 2]
        → MLP  [B, warp_hidden]
        → reshape [B, warp_hidden // (seed_h*seed_w), seed_h, seed_w]
        → PixelShuffle-based upsampling to [B, 2, lat_h, lat_w]
        → tanh * max_disp  (bounded displacement in normalised coords)
    """

    def __init__(
        self,
        lat_h: int,
        lat_w: int,
        warp_hidden: int = 128,
        seed_h: int = 4,
        seed_w: int = 8,
        max_disp: float = 0.3,
    ):
        """
        Args:
            lat_h, lat_w : latent feature map spatial size
            warp_hidden  : hidden dimension of the MLP branch
            seed_h/w     : initial spatial seed before ConvTranspose upsampling
            max_disp     : maximum displacement in normalised [-1,1] coords
        """
        super().__init__()
        self.lat_h    = lat_h
        self.lat_w    = lat_w
        self.max_disp = max_disp
        self.seed_h   = seed_h
        self.seed_w   = seed_w

        seed_channels = warp_hidden  # channels at seed resolution

        # MLP: gaze delta → seed feature map (flattened)
        self.mlp = nn.Sequential(
            nn.Linear(2, warp_hidden),
            nn.SiLU(),
            nn.Linear(warp_hidden, warp_hidden),
            nn.SiLU(),
            nn.Linear(warp_hidden, seed_channels * seed_h * seed_w),
        )

        # Flow decoder: seed (seed_h×seed_w) → intermediate feature map.
        # Final upsampling to actual latent size is done in forward() via
        # F.interpolate so the module handles any latent resolution.
        mid_ch = seed_channels // 2
        self.flow_decoder = nn.Sequential(
            nn.Conv2d(seed_channels, mid_ch, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(mid_ch, 16, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 2, 1),   # output: (dy, dx) displacement at seed res
        )

        # Zero-init final conv → no warp at initialisation
        nn.init.zeros_(self.flow_decoder[-1].weight)
        nn.init.zeros_(self.flow_decoder[-1].bias)

    # ------------------------------------------------------------------
    def forward(self, latent: torch.Tensor,
                source_gaze: torch.Tensor,
                target_gaze: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent      : [B, C, lat_h, lat_w]  source latent features
            source_gaze : [B, 2]  raw (pitch, yaw) in radians
            target_gaze : [B, 2]  raw (pitch, yaw) in radians
        Returns:
            warped_latent: [B, C, lat_h, lat_w]
            flow         : [B, 2, lat_h, lat_w] (for optional smoothness loss)
        """
        B, C, lat_h, lat_w = latent.shape
        delta = target_gaze - source_gaze  # [B, 2]

        # Predict flow at seed resolution, then upsample to latent resolution
        seed = self.mlp(delta)                                       # [B, seed_ch*seed_h*seed_w]
        seed = seed.view(B, -1, self.seed_h, self.seed_w)           # [B, seed_ch, seed_h, seed_w]
        flow_seed = self.flow_decoder(seed)                          # [B, 2, seed_h, seed_w]

        # Upsample to actual latent spatial size (handles any resolution)
        flow = F.interpolate(
            flow_seed, size=(lat_h, lat_w),
            mode='bilinear', align_corners=False,
        )                                                            # [B, 2, lat_h, lat_w]
        flow = torch.tanh(flow) * self.max_disp                     # bounded displacement

        # Build sampling grid: base grid + predicted displacement
        # grid coordinates are in [-1, 1] (as required by grid_sample)
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, lat_h, device=latent.device),
            torch.linspace(-1, 1, lat_w, device=latent.device),
            indexing='ij',
        )
        base_grid = torch.stack([grid_x, grid_y], dim=-1)           # [lat_h, lat_w, 2]
        base_grid = base_grid.unsqueeze(0).expand(B, -1, -1, -1)    # [B, lat_h, lat_w, 2]

        # flow: [B, 2, lat_h, lat_w] → [B, lat_h, lat_w, 2] (dx, dy)
        disp = flow.permute(0, 2, 3, 1)                              # [B, lat_h, lat_w, 2]
        sampling_grid = base_grid + disp

        warped = F.grid_sample(
            latent, sampling_grid,
            mode='bilinear', padding_mode='border', align_corners=True,
        )
        return warped, flow


# ---------------------------------------------------------------------------
# 5. Identity FiLM Heads (for decoder)
# ---------------------------------------------------------------------------

class IdentityFiLMHead(nn.Module):
    """Project global identity embedding to FiLM (scale, shift) for one decoder stage.

    Zero-initialised → identity transform at init; stable Phase 1 → 2 transfer.
    """

    def __init__(self, id_emb_dim: int, num_channels: int):
        super().__init__()
        self.proj = nn.Linear(id_emb_dim, num_channels * 2)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, id_emb: torch.Tensor):
        """
        Args:
            id_emb: [B, id_emb_dim]
        Returns:
            scale, shift: each [B, C, 1, 1]
        """
        params = self.proj(id_emb)                      # [B, 2C]
        scale, shift = params.chunk(2, dim=-1)          # [B, C], [B, C]
        return (
            scale.unsqueeze(-1).unsqueeze(-1),          # [B, C, 1, 1]
            shift.unsqueeze(-1).unsqueeze(-1),
        )


# ---------------------------------------------------------------------------
# 6. PersonalGazeNet — Main Network
# ---------------------------------------------------------------------------

class PersonalGazeNet(nn.Module):
    """Personalized Gaze Redirection via Spatial Identity Tokens and Warping.

    Inputs
    ------
    source_eye_crops : [B, 3, H, W]        — width-concatenated (left|right) eye crop
    gaze_cond        : [B, 2, gaze_dim]    — [:,0,:]=head_emb  [:,1,:]=gaze_emb
    ref_eyes         : [B, N_ref, 3, H, W] — N reference frames from the same subject
    source_gaze_raw  : [B, 2]              — raw (pitch, yaw) in radians for warp module
    target_gaze_raw  : [B, 2]             — raw target (pitch, yaw) in radians

    Output
    ------
    generated_eyes   : [B, 3, H, W]        — redirected eye crop (same shape as input)

    Architecture summary
    --------------------
    [IdentityEncoder]  ×  N_ref  →  IdentityAggregator
                                        ↓ id_feat_lat (spatial tokens)
                                        ↓ id_emb (global)

    [SourceEncoder]  →  (feat_0, feat_1, src_lat)
                                        ↓
                         GazeDeltaWarpModule(src_lat, delta_gaze)
                                        ↓ warped_lat
                         CrossAttention(warped_lat, id_feat_lat)
                                        ↓ fused_lat

    [GazeCondDecoder]
        fused_lat  →  Upsample + skip(feat_1) + UNetBlock(gaze) + IdFiLM
                  →  Upsample + skip(feat_0) + UNetBlock(gaze) + IdFiLM
                  →  DiCFinalLayer  →  Δimage
                  →  source + Δimage  (residual prediction, same as EyeOnlyGazeDiC)
    """

    def __init__(
        self,
        in_channels:   int  = 3,
        hidden_size:   int  = 32,
        mult_channels: list = None,
        depth:         list = None,
        gaze_dim:      int  = 64,
        id_emb_dim:    int  = 128,
        num_attn_heads: int = 4,
        warp_hidden:   int  = 128,
        warp_max_disp: float = 0.3,
        affinef:       int  = 3,
        actfunc:       str  = 'gelu',
        dropout:       float = 0.1,
        num_groups:    int  = 32,
        min_channels:  int  = 4,
        **kwargs,
    ):
        super().__init__()

        if mult_channels is None:
            mult_channels = [1, 2, 4, 2, 1]
        if depth is None:
            depth = [2, 2, 4, 2, 2]

        # channels[0..4]: enc0, enc1, latent, dec0, dec1
        channels = [hidden_size * m for m in mult_channels]
        C0, C1, C2 = channels[0], channels[1], channels[2]
        D0, D1     = channels[3], channels[4]  # decoder channels

        self.channels   = channels
        self.gaze_dim   = gaze_dim
        self.id_emb_dim = id_emb_dim

        # ------------------------------------------------------------------
        # A. Identity Encoder (gaze-unconditional, shared across N_ref frames)
        # ------------------------------------------------------------------
        self.identity_encoder = IdentityEncoder(
            in_channels=in_channels,
            hidden_size=hidden_size,
            mult_channels=mult_channels[:3],
            depth=depth[:3],
        )

        # ------------------------------------------------------------------
        # B. Identity Aggregator
        # ------------------------------------------------------------------
        self.identity_aggregator = IdentityAggregator(
            lat_channels=C2,
            id_emb_dim=id_emb_dim,
        )

        # ------------------------------------------------------------------
        # C. Source Encoder (gaze-conditioned UNetBlocks for richer features)
        # ------------------------------------------------------------------
        # Stage-specific gaze embedders
        self.gaze_embedder_enc0 = ConditionEmbedder(gaze_dim, C0)
        self.gaze_embedder_enc1 = ConditionEmbedder(gaze_dim, C1)
        self.gaze_embedder_lat  = ConditionEmbedder(gaze_dim, C2)

        self.src_x_embedder = OverlapPatchEmbed(3, 1, in_channels, C0)

        def _make_stage(in_ch, out_ch, emb_ch, n):
            return nn.ModuleList([
                UNetBlock(
                    in_channels=(in_ch if j == 0 else out_ch),
                    out_channels=out_ch,
                    emb_channels=emb_ch,
                    dropout=dropout,
                    affinef=affinef,
                    actfunc=actfunc,
                    num_groups=num_groups,
                    min_channels=min_channels,
                )
                for j in range(n)
            ])

        self.src_enc0 = _make_stage(C0, C0, C0, depth[0])
        self.src_down0 = Downsample(C0, C1)
        self.src_enc1 = _make_stage(C1, C1, C1, depth[1])
        self.src_down1 = Downsample(C1, C2)
        self.src_lat   = _make_stage(C2, C2, C2, depth[2])

        # ------------------------------------------------------------------
        # D. Gaze-Delta Warp Module
        # ------------------------------------------------------------------
        # seed_h/w are the MLP output spatial dims before upsampling.
        # The flow_decoder upsamples to actual latent size in forward(),
        # so these just need to be small: 4×8 covers the directional structure.
        self.warp_module = GazeDeltaWarpModule(
            lat_h=20, lat_w=40,    # nominal size for __init__; forward adapts
            warp_hidden=warp_hidden,
            seed_h=4,
            seed_w=8,
            max_disp=warp_max_disp,
        )

        # ------------------------------------------------------------------
        # E. Cross-Attention Bottleneck (warped src_lat ← id_feat_lat)
        # ------------------------------------------------------------------
        self.cross_attn_lat = CrossAttention(
            q_dim=C2,
            kv_dim=C2,
            num_heads=num_attn_heads,
            dropout=dropout,
        )
        self.cross_attn_norm = nn.LayerNorm(C2)  # post-attn layer norm

        # ------------------------------------------------------------------
        # F. Gaze-Conditioned Decoder with Identity FiLM
        # ------------------------------------------------------------------
        # Gaze embedders for decoder stages (symmetric to encoder)
        self.gaze_embedder_dec0 = ConditionEmbedder(gaze_dim, D0)
        self.gaze_embedder_dec1 = ConditionEmbedder(gaze_dim, D1)

        self.up0 = Upsample(C2, D0)
        self.up1 = Upsample(D0, D1)

        # First decoder block takes skip connection (channel concat)
        self.dec_blocks0 = _make_stage(D0 + C1, D0, D0, depth[3])
        self.dec_blocks1 = _make_stage(D1 + C0, D1, D1, depth[4])

        # Identity FiLM heads for each decoder stage
        self.id_film_dec0 = IdentityFiLMHead(id_emb_dim, D0)
        self.id_film_dec1 = IdentityFiLMHead(id_emb_dim, D1)

        # Final output layer (residual prediction as in EyeOnlyGazeDiC)
        self.final_layer = DiCFinalLayer(D1, in_channels, D1)

    # ======================================================================
    # Forward
    # ======================================================================

    def _encode_source(self, source_eye: torch.Tensor, gaze_cond: torch.Tensor):
        """Run gaze-conditioned source encoder.

        Returns:
            feat_0, feat_1, feat_lat
        """
        emb0  = self.gaze_embedder_enc0(gaze_cond)
        emb1  = self.gaze_embedder_enc1(gaze_cond)
        emb_l = self.gaze_embedder_lat(gaze_cond)

        x = self.src_x_embedder(source_eye)

        for blk in self.src_enc0:
            x = blk(x, emb0)
        feat_0 = x

        x = self.src_down0(x)
        for blk in self.src_enc1:
            x = blk(x, emb1)
        feat_1 = x

        x = self.src_down1(x)
        for blk in self.src_lat:
            x = blk(x, emb_l)
        feat_lat = x

        return feat_0, feat_1, feat_lat

    # ------------------------------------------------------------------
    def forward(
        self,
        source_eye_crops: torch.Tensor,
        gaze_cond:        torch.Tensor,
        ref_eyes:         torch.Tensor,
        source_gaze_raw:  torch.Tensor = None,
        target_gaze_raw:  torch.Tensor = None,
    ):
        """
        Args:
            source_eye_crops : [B, 3, H, W]
            gaze_cond        : [B, 2, gaze_dim]
            ref_eyes         : [B, N_ref, 3, H, W]
            source_gaze_raw  : [B, 2]  raw pitch/yaw (optional; disables warp if None)
            target_gaze_raw  : [B, 2]  raw pitch/yaw (optional)
        Returns:
            generated_eyes   : [B, 3, H, W]
            flow             : [B, 2, lat_h, lat_w] or None  (for optional loss)
        """
        B, N_ref, C, H, W = ref_eyes.shape

        # ── A. Identity encoding: process each reference frame ──────────
        feats_0_list, feats_1_list, feats_lat_list = [], [], []
        for n in range(N_ref):
            f0, f1, fl = self.identity_encoder(ref_eyes[:, n])  # [B, Ci, Hi, Wi]
            feats_0_list.append(f0)
            feats_1_list.append(f1)
            feats_lat_list.append(fl)

        id_feat_0, id_feat_1, id_feat_lat, id_emb = self.identity_aggregator(
            feats_0_list, feats_1_list, feats_lat_list
        )

        # ── B. Source encoding (gaze-conditioned) ───────────────────────
        src_feat_0, src_feat_1, src_lat = self._encode_source(
            source_eye_crops, gaze_cond
        )

        # ── C. Gaze-delta warping at latent resolution ──────────────────
        flow = None
        if source_gaze_raw is not None and target_gaze_raw is not None:
            warped_lat, flow = self.warp_module(
                src_lat, source_gaze_raw, target_gaze_raw
            )
        else:
            warped_lat = src_lat   # no warp (still valid; cross-attn handles identity)

        # ── D. Cross-attention: fuse warped source with identity ─────────
        lat_h, lat_w = warped_lat.shape[-2], warped_lat.shape[-1]
        N_lat = lat_h * lat_w

        # Flatten spatial → token dimension
        q_tokens  = warped_lat.permute(0, 2, 3, 1).reshape(B, N_lat, -1)   # [B, N_lat, C2]
        kv_tokens = id_feat_lat.permute(0, 2, 3, 1).reshape(B, N_lat, -1)  # [B, N_lat, C2]

        attn_delta = self.cross_attn_lat(q_tokens, kv_tokens)              # [B, N_lat, C2]
        fused_tokens = self.cross_attn_norm(q_tokens + attn_delta)          # pre-residual
        fused_lat = fused_tokens.reshape(B, lat_h, lat_w, -1).permute(0, 3, 1, 2)  # [B, C2, lat_h, lat_w]

        # ── E. Gaze-conditioned decoder with Identity FiLM ──────────────
        emb_dec0 = self.gaze_embedder_dec0(gaze_cond)
        emb_dec1 = self.gaze_embedder_dec1(gaze_cond)

        # Stage 0: up(fused_lat) + skip(src_feat_1)
        x = self.up0(fused_lat)
        x = torch.cat([x, src_feat_1], dim=1)         # skip connection
        id_s0, id_sh0 = self.id_film_dec0(id_emb)
        for i, blk in enumerate(self.dec_blocks0):
            if i == 0:
                # first block processes skip concat
                x = blk(x, emb_dec0, subject_scale=id_s0, subject_shift=id_sh0)
            else:
                x = blk(x, emb_dec0, subject_scale=id_s0, subject_shift=id_sh0)

        # Stage 1: up(stage0) + skip(src_feat_0)
        x = self.up1(x)
        x = torch.cat([x, src_feat_0], dim=1)         # skip connection
        id_s1, id_sh1 = self.id_film_dec1(id_emb)
        for blk in self.dec_blocks1:
            x = blk(x, emb_dec1, subject_scale=id_s1, subject_shift=id_sh1)

        # Final residual output
        delta = self.final_layer(x, emb_dec1)
        generated_eyes = torch.clamp(source_eye_crops + delta, -1.0, 1.0)

        return generated_eyes, flow

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode_identity(self, ref_eyes: torch.Tensor):
        """Pre-compute and return cached identity representation.

        Call once per subject at inference time.

        Args:
            ref_eyes: [B, N_ref, 3, H, W]  (or [1, N_ref, 3, H, W])
        Returns:
            id_feat_lat: [B, C2, lat_h, lat_w]
            id_emb     : [B, id_emb_dim]
        """
        B, N_ref = ref_eyes.shape[:2]
        feats_0_list, feats_1_list, feats_lat_list = [], [], []
        for n in range(N_ref):
            f0, f1, fl = self.identity_encoder(ref_eyes[:, n])
            feats_0_list.append(f0)
            feats_1_list.append(f1)
            feats_lat_list.append(fl)
        _, _, id_feat_lat, id_emb = self.identity_aggregator(
            feats_0_list, feats_1_list, feats_lat_list
        )
        return id_feat_lat, id_emb

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# 7. PersonalGazeWrapper — drop-in replacement for EyeOnlyWrapper
# ---------------------------------------------------------------------------

class PersonalGazeWrapper(nn.Module):
    """Drop-in wrapper that adapts PersonalGazeNet to the existing training API.

    The existing train.py passes:
        source_eye_crops, encoder_hidden_states, source_face (optional)

    This wrapper adds:
        ref_eyes         — defaults to source_eye_crops (self-supervised Phase 1)
        source_gaze_raw  — raw gaze angles (provided via kwargs or batch)
        target_gaze_raw  — raw gaze angles (provided via kwargs or batch)

    It also exposes ``paste_eyes`` for inference (same as EyeOnlyWrapper).
    """

    def __init__(self, config: dict):
        super().__init__()
        self.net = PersonalGazeNet(
            in_channels   = config.get('in_channels',    3),
            hidden_size   = config.get('hidden_size',    32),
            mult_channels = config.get('mult_channels',  [1, 2, 4, 2, 1]),
            depth         = config.get('depth',          [2, 2, 4, 2, 2]),
            gaze_dim      = config.get('gaze_dim',       64),
            id_emb_dim    = config.get('id_emb_dim',     128),
            num_attn_heads= config.get('num_attn_heads', 4),
            warp_hidden   = config.get('warp_hidden',    128),
            warp_max_disp = config.get('warp_max_disp',  0.3),
            affinef       = config.get('affinef',        3),
            actfunc       = config.get('actfunc',        'gelu'),
            dropout       = config.get('dropout',        0.1),
            num_groups    = config.get('num_groups',     32),
            min_channels  = config.get('min_channels',   4),
        )

    def forward(
        self,
        source_eye_crops:  torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        ref_eyes:          torch.Tensor  = None,
        source_gaze_raw:   torch.Tensor  = None,
        target_gaze_raw:   torch.Tensor  = None,
        source_face:       torch.Tensor  = None,   # unused; kept for API compat
    ):
        """
        Args:
            source_eye_crops       : [B, 3, H, W]
            encoder_hidden_states  : [B, 2, gaze_dim]  (same format as EyeOnlyWrapper)
            ref_eyes               : [B, N_ref, 3, H, W] or None (→ self-supervised)
            source_gaze_raw        : [B, 2] raw gaze angles (optional)
            target_gaze_raw        : [B, 2] raw gaze angles (optional)
            source_face            : ignored (kept for API compatibility)
        Returns:
            generated_eyes : [B, 3, H, W]
            flow           : [B, 2, lat_h, lat_w] or None
        """
        if ref_eyes is None:
            # Phase 1 self-supervised: use source as own reference
            ref_eyes = source_eye_crops.unsqueeze(1)   # [B, 1, 3, H, W]

        return self.net(
            source_eye_crops=source_eye_crops,
            gaze_cond=encoder_hidden_states,
            ref_eyes=ref_eyes,
            source_gaze_raw=source_gaze_raw,
            target_gaze_raw=target_gaze_raw,
        )

    # ------------------------------------------------------------------
    def paste_eyes(self, generated_eyes, source_image, eye_bbox, blend_margin=4):
        """Paste generated eye crops back into source face image.

        Identical API to EyeOnlyWrapper.paste_eyes.
        """
        B, _, H, W = source_image.shape
        result = source_image.clone()
        w_single = generated_eyes.shape[-1] // 2
        eye_crops = [
            generated_eyes[:, :, :, :w_single],
            generated_eyes[:, :, :, w_single:],
        ]

        for b in range(B):
            for eye_idx, bbox_slice in enumerate([slice(0, 4), slice(4, 8)]):
                x1, y1, x2, y2 = (eye_bbox[b, bbox_slice] * H).int().tolist()
                x1, x2 = max(0, x1), min(W, x2)
                y1, y2 = max(0, y1), min(H, y2)
                eh, ew = y2 - y1, x2 - x1
                if eh > 0 and ew > 0:
                    resized = F.interpolate(
                        eye_crops[eye_idx][b:b+1], size=(eh, ew),
                        mode='bilinear', align_corners=False,
                    )[0]
                    mask = self._create_blend_mask(eh, ew, blend_margin,
                                                   source_image.device)
                    result[b, :, y1:y2, x1:x2] = (
                        mask * resized + (1 - mask) * result[b, :, y1:y2, x1:x2]
                    )
        return result

    @staticmethod
    def _create_blend_mask(h, w, margin, device):
        """Edge-feathered blending mask [1, H, W]."""
        if margin <= 0:
            return torch.ones(1, h, w, device=device)

        def edge_ramp(size, m):
            t = torch.ones(size, device=device)
            if m > 0:
                ramp = torch.linspace(
                    1 / (m + 1), m / (m + 1),
                    steps=min(m, size // 2), device=device,
                )
                t[:len(ramp)]         = ramp
                t[size - len(ramp):]  = ramp.flip(0)
            return t

        row = edge_ramp(h, margin)
        col = edge_ramp(w, margin)
        return torch.outer(row, col).unsqueeze(0)
