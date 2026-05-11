"""
GazeFormer — Hybrid CNN-Transformer for Personalized Gaze Redirection

Replaces the pure-CNN UNet backbone (EyeOnlyGazeDiC) with a hybrid architecture
that uses Transformer blocks at reduced resolutions for:
  1) Gaze Cross-Attention:     spatially-adaptive gaze control
  2) Self-Attention:           global eye structure / bilateral symmetry
  3) Identity Cross-Attention: content-adaptive personalization

Key differences from EyeOnlyGazeDiC (AdaLN UNet):
  - AdaLN applies *uniform* channel-wise scale/shift across all spatial locations.
    Cross-attention lets each pixel attend independently to gaze tokens,
    so iris regions can react strongly while eyelids stay stable.
  - Self-attention captures long-range spatial correlations (left–right symmetry
    in the width-concat eye input) that conv receptive fields miss.
  - Identity tokens in cross-attention provide spatially-varying personalization
    (vs. FiLM which modulates every pixel the same way per channel).

Architecture (3-level encoder–decoder, skip connections):
    Stage 0  (64×128, C0):  ConvBlock × 2    — local features
    ↓ Downsample
    Stage 1  (32×64,  C1):  ConvBlock + GazeFormerBlock × 1  — windowed attn
    ↓ Downsample
    Bottleneck (16×32, C2): GazeFormerBlock × N  — full attn + identity attn
    ↑ Upsample + skip
    Stage 2  (32×64,  C3):  ConvBlock + GazeFormerBlock × 1  — windowed + id
    ↑ Upsample + skip
    Stage 3  (64×128, C4):  ConvBlock × 2    — local refinement

    Final: DiCFinalLayer (gaze-conditioned AdaLN-Zero) → residual delta
    Output: clamp(source + delta, -1, 1)

Drop-in compatible with EyeOnlyWrapper interface via GazeFormerWrapper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.gaze_dic import (
    GroupNorm, OverlapPatchEmbed, Downsample, Upsample, DiCFinalLayer,
)


# ============================================================
#  1. Attention Primitives
# ============================================================

class WindowSelfAttention(nn.Module):
    """Multi-head self-attention with optional non-overlapping window partition.

    When the spatial size exceeds ``4 × window_size²``, tokens are grouped into
    ``window_size × window_size`` windows and attention is computed per window.
    Otherwise global (full-sequence) attention is used.
    """

    def __init__(self, dim, num_heads=4, window_size=8,
                 qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, f"dim={dim} not divisible by num_heads={num_heads}"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.window_size = window_size

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    # ------------------------------------------------------------------
    def _window_partition(self, x, H, W):
        B, _, C = x.shape
        ws = self.window_size
        x = x.view(B, H, W, C)
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h or pad_w:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        Hp, Wp = H + pad_h, W + pad_w
        nH, nW = Hp // ws, Wp // ws
        x = (x.view(B, nH, ws, nW, ws, C)
               .permute(0, 1, 3, 2, 4, 5)
               .reshape(B * nH * nW, ws * ws, C))
        return x, (Hp, Wp, nH, nW)

    def _window_unpartition(self, windows, B, H, W, Hp, Wp, nH, nW):
        ws = self.window_size
        C = windows.shape[-1]
        x = (windows.view(B, nH, nW, ws, ws, C)
                     .permute(0, 1, 3, 2, 4, 5)
                     .reshape(B, Hp, Wp, C))
        if Hp != H or Wp != W:
            x = x[:, :H, :W, :]
        return x.reshape(B, H * W, C)

    # ------------------------------------------------------------------
    def _scaled_dot_product(self, x):
        B, N, C = x.shape
        qkv = (self.qkv(x)
               .reshape(B, N, 3, self.num_heads, self.head_dim)
               .permute(2, 0, 3, 1, 4))
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(out))

    # ------------------------------------------------------------------
    def forward(self, x, H, W):
        """
        Args:
            x:  [B, H*W, C]
            H, W: spatial dims
        """
        B = x.shape[0]
        use_win = (self.window_size is not None
                   and H * W > self.window_size ** 2 * 4)
        if use_win:
            win, (Hp, Wp, nH, nW) = self._window_partition(x, H, W)
            win = self._scaled_dot_product(win)
            return self._window_unpartition(win, B, H, W, Hp, Wp, nH, nW)
        return self._scaled_dot_product(x)


class GazeCrossAttention(nn.Module):
    """Cross-attention: spatial feature tokens (Q) attend to context tokens (KV).

    Used for both gaze-direction conditioning and identity-token conditioning.
    Unlike AdaLN (uniform per-channel modulation), cross-attention produces a
    *different* update at every spatial location, enabling region-adaptive control.
    """

    def __init__(self, dim, context_dim, num_heads=4,
                 qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_proj = nn.Linear(context_dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.out_proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, context):
        """
        Args:
            x:       [B, N, C]            — spatial queries
            context: [B, M, context_dim]  — gaze / identity tokens
        """
        B, N, C = x.shape
        M = context.shape[1]
        h, d = self.num_heads, self.head_dim

        q = self.q_proj(x).reshape(B, N, h, d).permute(0, 2, 1, 3)
        kv = (self.kv_proj(context)
              .reshape(B, M, 2, h, d)
              .permute(2, 0, 3, 1, 4))
        k, v = kv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.out_proj(out))


# ============================================================
#  2. Feed-Forward with Depth-wise Conv
# ============================================================

class ConvFFN(nn.Module):
    """FFN with a 3×3 depth-wise conv for local spatial mixing.

    Compared to a plain MLP, the intermediate DW-Conv injects inductive bias
    for nearby-pixel coherence — useful for dense pixel-level generation.
    """

    def __init__(self, dim, hidden_dim=None, dropout=0.):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, H, W):
        """x: [B, H*W, C] → [B, H*W, C]"""
        B, N, C = x.shape
        x = self.fc1(x)
        x = self.act(self.dwconv(x.transpose(1, 2).view(B, -1, H, W))
                     .flatten(2).transpose(1, 2))
        return self.drop(self.fc2(self.drop(x)))


# ============================================================
#  3. Core Blocks
# ============================================================

class GazeFormerBlock(nn.Module):
    """Pre-norm Transformer block with three attention sub-layers:

    1. WindowSelfAttention — spatial structure / bilateral symmetry
    2. GazeCrossAttention  — spatially-adaptive gaze conditioning
    3. GazeCrossAttention  — identity tokens (optional, for personalization)
    4. ConvFFN             — local spatial mixing + channel transformation

    Identity cross-attention is zero-initialised so the block starts as if it
    did not exist, ensuring stable warm-start from pre-trained weights.
    """

    def __init__(self, dim, num_heads, gaze_token_dim, window_size=8,
                 mlp_ratio=3.0, dropout=0.1,
                 use_identity_attn=False, identity_token_dim=None):
        super().__init__()
        # -- self-attention --
        self.norm_sa = nn.LayerNorm(dim)
        self.self_attn = WindowSelfAttention(
            dim, num_heads, window_size,
            attn_drop=dropout, proj_drop=dropout)

        # -- gaze cross-attention --
        self.norm_gaze = nn.LayerNorm(dim)
        self.gaze_attn = GazeCrossAttention(
            dim, gaze_token_dim, num_heads,
            attn_drop=dropout, proj_drop=dropout)

        # -- identity cross-attention (optional) --
        self.use_identity_attn = use_identity_attn
        if use_identity_attn:
            self.norm_id = nn.LayerNorm(dim)
            self.identity_attn = GazeCrossAttention(
                dim, identity_token_dim or dim, num_heads,
                attn_drop=dropout, proj_drop=dropout)
            nn.init.zeros_(self.identity_attn.out_proj.weight)
            nn.init.zeros_(self.identity_attn.out_proj.bias)
            self.id_gate = nn.Parameter(torch.zeros(1))

        # -- ConvFFN --
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = ConvFFN(dim, int(dim * mlp_ratio), dropout)

    def forward(self, x, H, W, gaze_tokens, identity_tokens=None):
        """
        Args:
            x:               [B, H*W, C]
            gaze_tokens:     [B, T_g, gaze_token_dim]
            identity_tokens: [B, T_id, identity_token_dim] or None
        """
        x = x + self.self_attn(self.norm_sa(x), H, W)
        x = x + self.gaze_attn(self.norm_gaze(x), gaze_tokens)

        if self.use_identity_attn and identity_tokens is not None:
            x = x + torch.sigmoid(self.id_gate) * \
                self.identity_attn(self.norm_id(x), identity_tokens)

        x = x + self.ffn(self.norm_ffn(x), H, W)
        return x


class ConvBlock(nn.Module):
    """Lightweight conv residual block for full-resolution stages.

    GroupNorm → GELU → Conv3×3 → GroupNorm → GELU → Conv3×3 + skip.
    No attention — used where token count is too large for efficient attention.
    """

    def __init__(self, in_channels, out_channels, dropout=0.1):
        super().__init__()
        self.norm0 = GroupNorm(in_channels)
        self.conv0 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.norm1 = GroupNorm(out_channels)
        self.conv1 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.act = nn.GELU()
        self.skip = (nn.Conv2d(in_channels, out_channels, 1)
                     if in_channels != out_channels else None)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        residual = self.skip(x) if self.skip is not None else x
        h = self.conv0(self.act(self.norm0(x)))
        h = self.conv1(self.act(self.norm1(h)))
        return self.dropout(h) + residual


# ============================================================
#  4. Gaze & Identity Tokenisers
# ============================================================

class GazeTokenProjector(nn.Module):
    """Project gaze condition → per-stage gaze tokens for cross-attention.

    Input:  [B, 2, gaze_dim]  (head + gaze embeddings from MLPNetwork)
    Output: [B, 2 + num_extra, token_dim]

    The two base tokens (head, gaze) are projected via separate MLPs.
    Optional extra learnable tokens are mixed with the base tokens through
    a small self-attention layer, enriching the gaze context seen by the
    downstream cross-attention.
    """

    def __init__(self, gaze_dim, token_dim, num_extra_tokens=4):
        super().__init__()
        self.num_extra = num_extra_tokens

        self.head_proj = nn.Sequential(
            nn.Linear(gaze_dim, token_dim), nn.SiLU(),
            nn.Linear(token_dim, token_dim))
        self.gaze_proj = nn.Sequential(
            nn.Linear(gaze_dim, token_dim), nn.SiLU(),
            nn.Linear(token_dim, token_dim))

        if num_extra_tokens > 0:
            self.extra_tokens = nn.Parameter(
                torch.randn(1, num_extra_tokens, token_dim) * 0.02)
            n_heads = max(1, min(4, token_dim // 16))
            self.mixer = nn.MultiheadAttention(
                token_dim, n_heads, batch_first=True, dropout=0.)

        self.norm = nn.LayerNorm(token_dim)

    def forward(self, gaze_cond):
        """gaze_cond: [B, 2, gaze_dim] or [B, gaze_dim]"""
        if gaze_cond.dim() == 2:
            gaze_cond = gaze_cond.unsqueeze(1).expand(-1, 2, -1)

        B = gaze_cond.shape[0]
        head_tok = self.head_proj(gaze_cond[:, 0]).unsqueeze(1)   # [B,1,D]
        gaze_tok = self.gaze_proj(gaze_cond[:, 1]).unsqueeze(1)   # [B,1,D]
        tokens = torch.cat([head_tok, gaze_tok], dim=1)           # [B,2,D]

        if self.num_extra > 0:
            extra = self.extra_tokens.expand(B, -1, -1)
            tokens = torch.cat([tokens, extra], dim=1)
            tokens, _ = self.mixer(tokens, tokens, tokens)

        return self.norm(tokens)


class IdentityTokenEncoder(nn.Module):
    """Encode source eye appearance into a set of identity tokens.

    A lightweight CNN extracts multi-scale features, then a cross-attention
    pooling with learnable query tokens distils them into a fixed-length
    token set.  These tokens participate in cross-attention inside the
    decoder to provide *spatially-adaptive* personalization (richer than
    channel-wise FiLM used in SubjectAdapter).

    For per-user fine-tuning:
        1. Freeze GazeFormer backbone.
        2. Only train IdentityTokenEncoder (~300K params).
        3. At inference, encode reference eye(s) once → cache tokens.
    """

    def __init__(self, in_channels=3, token_dim=128, num_tokens=8):
        super().__init__()
        self.num_tokens = num_tokens

        self.stem = nn.Conv2d(in_channels, 32, 3, 1, 1)
        self.enc = nn.Sequential(
            GroupNorm(32), nn.GELU(),
            Downsample(32, 64),
            GroupNorm(64), nn.GELU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            GroupNorm(64), nn.GELU(),
            Downsample(64, 128),
            GroupNorm(128), nn.GELU(),
            nn.Conv2d(128, 128, 3, 1, 1),
        )

        self.query_tokens = nn.Parameter(
            torch.randn(1, num_tokens, 128) * 0.02)
        self.cross_pool = nn.MultiheadAttention(
            128, num_heads=4, batch_first=True)
        self.proj = nn.Sequential(
            nn.Linear(128, token_dim),
            nn.LayerNorm(token_dim),
        )

    def forward(self, source_eye):
        """source_eye: [B, 3, H, W] → [B, num_tokens, token_dim]"""
        B = source_eye.shape[0]
        feat = self.enc(self.stem(source_eye))           # [B,128,H/4,W/4]
        feat_flat = feat.flatten(2).transpose(1, 2)      # [B,N,128]
        queries = self.query_tokens.expand(B, -1, -1)
        tokens, _ = self.cross_pool(queries, feat_flat, feat_flat)
        return self.proj(tokens)


# ============================================================
#  5. GazeFormer — Full Encoder-Decoder
# ============================================================

class GazeFormer(nn.Module):
    """Hybrid CNN-Transformer for gaze-conditioned eye image generation.

    Resolution schedule (default):
        64×128 → 32×64 → 16×32 (bottleneck) → 32×64 → 64×128

    Transformer blocks operate only at ≤ 32×64 resolution for efficiency;
    full-resolution stages use ConvBlocks.  Gaze control is injected via
    cross-attention at reduced resolutions and via AdaLN-Zero in the
    final output layer.
    """

    def __init__(
        self,
        in_channels=3,
        hidden_size=32,
        mult_channels=None,
        depth=None,
        gaze_dim=64,
        num_heads=None,
        window_size=8,
        mlp_ratio=3.0,
        num_gaze_tokens=4,
        identity_token_dim=128,
        num_identity_tokens=8,
        dropout=0.1,
        **kwargs,
    ):
        super().__init__()
        if mult_channels is None:
            mult_channels = [1, 2, 4, 2, 1]
        if depth is None:
            depth = [2, 1, 4, 1, 2]
        if num_heads is None:
            num_heads = [2, 4, 8, 4, 2]

        channels = [hidden_size * m for m in mult_channels]
        self.channels = channels
        self.depth = depth
        self.gaze_dim = gaze_dim

        # ---- patch embedding ----
        self.x_embedder = OverlapPatchEmbed(3, 1, in_channels, channels[0])

        # ---- per-stage gaze token projectors (stages 1 & bottleneck) ----
        self.gaze_projectors = nn.ModuleList([
            GazeTokenProjector(gaze_dim, channels[1], num_gaze_tokens),
            GazeTokenProjector(gaze_dim, channels[2], num_gaze_tokens),
        ])

        # ---- encoder stage 0: conv only (full res) ----
        self.enc0_blocks = nn.ModuleList(
            [ConvBlock(channels[0], channels[0], dropout) for _ in range(depth[0])])

        # ---- encoder stage 1: conv + transformer (½ res) ----
        self.enc1_conv = ConvBlock(channels[1], channels[1], dropout)
        self.enc1_transformers = nn.ModuleList([
            GazeFormerBlock(channels[1], num_heads[1], channels[1],
                            window_size, mlp_ratio, dropout)
            for _ in range(max(1, depth[1]))])

        # ---- bottleneck: transformer (¼ res, global attention) ----
        self.lat_transformers = nn.ModuleList([
            GazeFormerBlock(channels[2], num_heads[2], channels[2],
                            window_size=None,
                            mlp_ratio=mlp_ratio, dropout=dropout,
                            use_identity_attn=True,
                            identity_token_dim=identity_token_dim)
            for _ in range(depth[2])])

        # ---- down / up ----
        self.downs = nn.ModuleList([
            Downsample(channels[0], channels[1]),
            Downsample(channels[1], channels[2]),
        ])
        self.ups = nn.ModuleList([
            Upsample(channels[2], channels[3]),
            Upsample(channels[3], channels[4]),
        ])

        # ---- decoder stage 0: conv + transformer (½ res, skip from enc1) ----
        self.dec0_conv = ConvBlock(channels[3] + channels[1],
                                   channels[3], dropout)
        self.dec0_transformers = nn.ModuleList([
            GazeFormerBlock(channels[3], num_heads[3], channels[1],
                            window_size, mlp_ratio, dropout,
                            use_identity_attn=True,
                            identity_token_dim=identity_token_dim)
            for _ in range(max(1, depth[3]))])

        # ---- decoder stage 1: conv only (full res, skip from enc0) ----
        self.dec1_blocks = nn.ModuleList()
        for j in range(depth[4]):
            c_in = channels[4] + channels[0] if j == 0 else channels[4]
            self.dec1_blocks.append(ConvBlock(c_in, channels[4], dropout))

        # ---- final: gaze-conditioned residual output ----
        self.gaze_final_embedder = nn.Sequential(
            nn.Linear(gaze_dim * 2, channels[0]),
            nn.SiLU(),
            nn.Linear(channels[0], channels[0]),
        )
        self.final_layer = DiCFinalLayer(channels[4], in_channels, channels[0])

    # ------------------------------------------------------------------
    def forward(self, eye_input, gaze_cond, identity_tokens=None):
        """
        Args:
            eye_input:       [B, 3, H, W]   — source eye crops
            gaze_cond:       [B, 2, gaze_dim] — head + gaze embeddings
            identity_tokens: [B, T, D]       — from IdentityTokenEncoder
        Returns:
            [B, 3, H, W]  — generated eye image (residual prediction)
        """
        B = eye_input.shape[0]

        # Gaze tokens for transformer stages
        gaze_tok_1 = self.gaze_projectors[0](gaze_cond)     # for ½-res
        gaze_tok_2 = self.gaze_projectors[1](gaze_cond)     # for ¼-res

        # Patch embed
        x = self.x_embedder(eye_input)                       # [B, C0, H, W]

        # ---- encoder stage 0 (full res, conv only) ----
        for blk in self.enc0_blocks:
            x = blk(x)
        skip0 = x

        x = self.downs[0](x)                                 # → ½ res

        # ---- encoder stage 1 (½ res, conv + transformer) ----
        x = self.enc1_conv(x)
        H1, W1 = x.shape[2], x.shape[3]
        x_seq = x.flatten(2).transpose(1, 2)                 # [B, N, C1]
        for tf in self.enc1_transformers:
            x_seq = tf(x_seq, H1, W1, gaze_tok_1)
        x = x_seq.transpose(1, 2).view(B, self.channels[1], H1, W1)
        skip1 = x

        x = self.downs[1](x)                                 # → ¼ res

        # ---- bottleneck (¼ res, global attention) ----
        Hl, Wl = x.shape[2], x.shape[3]
        x_seq = x.flatten(2).transpose(1, 2)
        for tf in self.lat_transformers:
            x_seq = tf(x_seq, Hl, Wl, gaze_tok_2, identity_tokens)
        x = x_seq.transpose(1, 2).view(B, self.channels[2], Hl, Wl)

        # ---- decoder stage 0 (½ res) ----
        x = self.ups[0](x)
        x = torch.cat([x, skip1], dim=1)
        x = self.dec0_conv(x)
        H1, W1 = x.shape[2], x.shape[3]
        x_seq = x.flatten(2).transpose(1, 2)
        for tf in self.dec0_transformers:
            x_seq = tf(x_seq, H1, W1, gaze_tok_1, identity_tokens)
        x = x_seq.transpose(1, 2).view(B, self.channels[3], H1, W1)

        # ---- decoder stage 1 (full res) ----
        x = self.ups[1](x)
        x = torch.cat([x, skip0], dim=1)
        for blk in self.dec1_blocks:
            x = blk(x)

        # ---- gaze-conditioned residual output ----
        gaze_flat = gaze_cond.flatten(1)                      # [B, 2*gaze_dim]
        final_cond = self.gaze_final_embedder(gaze_flat)
        delta = self.final_layer(x, final_cond)

        return torch.clamp(eye_input + delta, min=-1.0, max=1.0)


# ============================================================
#  6. GazeFormerWrapper — Drop-In for EyeOnlyWrapper
# ============================================================

class GazeFormerWrapper(nn.Module):
    """Complete generation wrapper, API-compatible with EyeOnlyWrapper.

    Training:  source_eye_crops + target_gaze → generated_eyes
    Inference: source_image + target_gaze → crop → generate → paste_eyes

    Personalization:
        Built-in IdentityTokenEncoder extracts identity tokens from the
        source eye crops and injects them via cross-attention.
        For per-user fine-tuning, freeze the backbone and only train the
        IdentityTokenEncoder (~300 K params).
    """

    def __init__(self, config, identity_config=None):
        super().__init__()

        self.gazeformer = GazeFormer(
            in_channels=config.get('in_channels', 3),
            hidden_size=config.get('hidden_size', 32),
            mult_channels=config.get('mult_channels', [1, 2, 4, 2, 1]),
            depth=config.get('depth', [2, 1, 4, 1, 2]),
            gaze_dim=config.get('gaze_dim', 64),
            num_heads=config.get('num_heads', [2, 4, 8, 4, 2]),
            window_size=config.get('window_size', 8),
            mlp_ratio=config.get('mlp_ratio', 3.0),
            num_gaze_tokens=config.get('num_gaze_tokens', 4),
            identity_token_dim=config.get('identity_token_dim', 128),
            num_identity_tokens=config.get('num_identity_tokens', 8),
            dropout=config.get('dropout', 0.1),
        )

        self.identity_encoder = None
        if identity_config is not None:
            self.identity_encoder = IdentityTokenEncoder(
                in_channels=identity_config.get('in_channels', 3),
                token_dim=identity_config.get('token_dim', 128),
                num_tokens=identity_config.get('num_tokens', 8),
            )

    # ------------------------------------------------------------------
    def forward(self, source_eye_crops, encoder_hidden_states,
                source_face=None):
        """
        Args:
            source_eye_crops:      [B, C, H, W]
            encoder_hidden_states: [B, 2, gaze_dim]
            source_face:           [B, 3, 256, 256]  (unused by default)
        """
        identity_tokens = None
        if self.identity_encoder is not None:
            identity_tokens = self.identity_encoder(source_eye_crops)

        return self.gazeformer(
            source_eye_crops, encoder_hidden_states, identity_tokens)

    # ------------------------------------------------------------------
    def paste_eyes(self, generated_eyes, source_image, eye_bbox,
                   blend_margin=4):
        """Paste generated eyes back onto source face image.

        Identical to EyeOnlyWrapper.paste_eyes — kept here for interface
        compatibility so training/inference scripts work unchanged.
        """
        B, _, H, W = source_image.shape
        result = source_image.clone()
        w_single = generated_eyes.shape[-1] // 2
        eye_crops = [generated_eyes[:, :, :, :w_single],
                     generated_eyes[:, :, :, w_single:]]

        for b in range(B):
            for eye_idx, bbox_slice in enumerate([slice(0, 4), slice(4, 8)]):
                x1, y1, x2, y2 = (eye_bbox[b, bbox_slice] * H).int().tolist()
                x1, x2 = max(0, x1), min(W, x2)
                y1, y2 = max(0, y1), min(H, y2)
                eh, ew = y2 - y1, x2 - x1
                if eh > 0 and ew > 0:
                    resized = F.interpolate(
                        eye_crops[eye_idx][b:b + 1], size=(eh, ew),
                        mode='bilinear', align_corners=False)[0]
                    mask = self._create_blend_mask(
                        eh, ew, blend_margin, source_image.device)
                    result[b, :, y1:y2, x1:x2] = (
                        mask * resized
                        + (1 - mask) * result[b, :, y1:y2, x1:x2])
        return result

    @staticmethod
    def _create_blend_mask(h, w, margin, device):
        if margin <= 0:
            return torch.ones(1, h, w, device=device)

        def edge_ramp(size, m):
            t = torch.ones(size, device=device)
            if m > 0:
                steps = min(m, size // 2)
                ramp = torch.linspace(1 / (m + 1), m / (m + 1),
                                      steps=steps, device=device)
                t[:len(ramp)] = ramp
                t[size - len(ramp):] = ramp.flip(0)
            return t

        return torch.outer(edge_ramp(h, margin),
                           edge_ramp(w, margin)).unsqueeze(0)

    # ------------------------------------------------------------------
    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
