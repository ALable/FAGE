"""
DualStreamGazeNet — Dual-Stream Architecture for Personalized Gaze Redirection

Core idea: separate identity-preserving appearance features from gaze-controlling
transform features into two explicit streams, then fuse them through cross-attention.

Stream 1 — AppearanceEncoder:
    Source eye crops → multi-scale identity features (no gaze conditioning).
    Extracts hierarchical spatial features that describe *who* the person is
    (iris texture, eyelid shape, skin tone, sclera patterns).

Stream 2 — GazeTransformDecoder:
    Takes latent representation + delta-gaze condition → generates target eyes.
    At each decoder level, cross-attends to Stream 1's features, selectively
    copying appearance details while adapting them for the new gaze direction.

Key innovations over EyeOnlyGazeDiC:
  1. Delta-gaze conditioning: models Δgaze = target - source (identity-map at Δ=0)
  2. Cross-attention fusion: queries from gaze stream, keys/values from appearance
     stream → richer interaction than FiLM scale/shift
  3. Multi-scale appearance features: spatial identity information preserved at all
     decoder resolutions, not compressed to a single vector
  4. Gaze-appearance disentanglement: appearance encoder receives no gaze signal,
     encouraging clean separation of *what to preserve* vs *what to change*
  5. Lightweight Personalization Head: replaces SubjectAdapter with attention-bias
     injection that can be fine-tuned per user with ~200K trainable params
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from models.gaze_dic import (
    GroupNorm, OverlapPatchEmbed, UNetBlock, Downsample, Upsample,
    DiCFinalLayer, ConditionEmbedder, modulate,
)


# ═══════════════════════════════════════════════════════════════════════
#  1. Delta-Gaze Embedder
# ═══════════════════════════════════════════════════════════════════════

class DeltaGazeEmbedder(nn.Module):
    """Maps (source_gaze, target_gaze, head_pose) → per-stage gaze embeddings.

    Uses sinusoidal positional encoding of angular deltas for smooth interpolation,
    then stage-specific MLPs to produce embeddings for each UNet level.

    Delta representation: target_gaze - source_gaze captures the *redirection*
    rather than absolute angles. This makes Δ=0 a natural identity mapping,
    simplifies learning, and improves generalization to unseen gaze combinations.
    """
    def __init__(self, angle_dim: int = 2, head_dim: int = 2,
                 freq_bands: int = 8, hidden_dim: int = 256,
                 out_dim: int = 64, num_stages: int = 3):
        super().__init__()
        self.freq_bands = freq_bands
        self.num_stages = num_stages

        # Sinusoidal encoding: each scalar → 2*freq_bands features
        enc_dim_per_scalar = 2 * freq_bands
        # Encoded dimensions: delta_gaze (angle_dim) + source_gaze (angle_dim) + head (head_dim)
        total_input_dim = (angle_dim * 2 + head_dim) * enc_dim_per_scalar

        self.shared_mlp = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        # Per-stage projection heads
        self.stage_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, out_dim),
                nn.LayerNorm(out_dim),
            )
            for _ in range(num_stages)
        ])

    def _sinusoidal_encode(self, x: torch.Tensor) -> torch.Tensor:
        """Positional encoding: x [B, D] → [B, D * 2 * freq_bands]"""
        freqs = torch.arange(self.freq_bands, device=x.device, dtype=x.dtype)
        freqs = (2.0 * math.pi * 2.0 ** freqs)  # [freq_bands]
        # x [B, D] → x_expanded [B, D, 1] × freqs [1, 1, F] → [B, D, F]
        angles = x.unsqueeze(-1) * freqs.unsqueeze(0).unsqueeze(0)
        return torch.cat([angles.sin(), angles.cos()], dim=-1).flatten(1)

    def forward(self, source_gaze: torch.Tensor, target_gaze: torch.Tensor,
                head_pose: torch.Tensor) -> list:
        """
        Args:
            source_gaze: [B, 2] source gaze angles (pitch, yaw)
            target_gaze: [B, 2] target gaze angles
            head_pose:   [B, 2] head pose angles
        Returns:
            list of [B, out_dim] embeddings, one per stage
        """
        delta_gaze = target_gaze - source_gaze  # [B, 2]

        # Encode all angular inputs with sinusoidal PE
        enc_delta = self._sinusoidal_encode(delta_gaze)
        enc_source = self._sinusoidal_encode(source_gaze)
        enc_head = self._sinusoidal_encode(head_pose)

        combined = torch.cat([enc_delta, enc_source, enc_head], dim=-1)
        shared_feat = self.shared_mlp(combined)  # [B, hidden_dim]

        return [head(shared_feat) for head in self.stage_heads]


# ═══════════════════════════════════════════════════════════════════════
#  2. Appearance Encoder (Stream 1)
# ═══════════════════════════════════════════════════════════════════════

class AppearanceEncoder(nn.Module):
    """Multi-scale appearance encoder that extracts hierarchical identity features.

    Architecture mirrors the main UNet encoder but receives NO gaze conditioning.
    This enforces disentanglement: the appearance stream only captures identity
    information (iris texture, eyelid shape, skin tone).

    Outputs feature maps at 3 scales for cross-attention in the decoder.

    Resolution flow (for 64×128 input):
        Level 0: [B, C0, 64, 128]  — fine detail (eyelash, iris pattern)
        Level 1: [B, C1, 32, 64]   — mid-level structure (eye shape)
        Level 2: [B, C2, 16, 32]   — coarse/global (face region context)
    """
    def __init__(self, in_channels: int = 3, base_ch: int = 32,
                 mult: list = None, depth: list = None):
        super().__init__()
        if mult is None:
            mult = [1, 2, 4]
        if depth is None:
            depth = [2, 2, 2]

        channels = [base_ch * m for m in mult]
        self.stem = OverlapPatchEmbed(3, 1, in_channels, channels[0])

        # Level 0 blocks
        self.level0 = nn.ModuleList()
        for j in range(depth[0]):
            self.level0.append(self._make_block(channels[0], channels[0]))

        # Down 0 → Level 1
        self.down0 = Downsample(channels[0], channels[1])
        self.level1 = nn.ModuleList()
        for j in range(depth[1]):
            self.level1.append(self._make_block(channels[1], channels[1]))

        # Down 1 → Level 2
        self.down1 = Downsample(channels[1], channels[2])
        self.level2 = nn.ModuleList()
        for j in range(depth[2]):
            self.level2.append(self._make_block(channels[2], channels[2]))

        self.out_channels = channels

    @staticmethod
    def _make_block(in_ch, out_ch):
        """Lightweight residual block without gaze conditioning."""
        return nn.Sequential(
            GroupNorm(in_ch, affine=True),
            nn.GELU(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            GroupNorm(out_ch, affine=True),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> list:
        """
        Args:
            x: [B, C_in, H, W] source eye crops
        Returns:
            list of feature maps at 3 scales: [level0, level1, level2]
        """
        features = []

        x = self.stem(x)
        for block in self.level0:
            x = x + block(x)  # residual
        features.append(x)

        x = self.down0(x)
        for block in self.level1:
            x = x + block(x)
        features.append(x)

        x = self.down1(x)
        for block in self.level2:
            x = x + block(x)
        features.append(x)

        return features


# ═══════════════════════════════════════════════════════════════════════
#  3. Cross-Attention Fusion Block
# ═══════════════════════════════════════════════════════════════════════

class GazeCrossAttentionBlock(nn.Module):
    """Cross-attention block where gaze-stream queries attend to appearance features.

    This is the key fusion mechanism: the gaze transform stream (queries) selectively
    retrieves identity-relevant information from the appearance stream (keys/values),
    guided by the current spatial location and gaze condition.

    Architecture:
        1. LayerNorm on query features
        2. Q = Linear(gaze_features), K/V = Linear(appearance_features)
        3. Gaze-modulated attention: attention weights biased by gaze condition
        4. Output projection + residual connection
        5. FFN with gaze AdaLN modulation

    The gaze condition modulates both the attention computation (via Q bias)
    and the post-attention features (via AdaLN), providing dual-path control.
    """
    def __init__(self, dim: int, app_dim: int, gaze_emb_dim: int,
                 num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Query from gaze stream
        self.norm_q = GroupNorm(dim)
        self.to_q = nn.Conv2d(dim, dim, 1)

        # Key/Value from appearance stream
        self.norm_kv = GroupNorm(app_dim)
        self.to_k = nn.Conv2d(app_dim, dim, 1)
        self.to_v = nn.Conv2d(app_dim, dim, 1)

        # Gaze-conditioned query bias: shifts Q based on desired gaze direction
        self.gaze_q_bias = nn.Sequential(
            nn.SiLU(),
            nn.Linear(gaze_emb_dim, dim),
        )
        nn.init.zeros_(self.gaze_q_bias[-1].weight)
        nn.init.zeros_(self.gaze_q_bias[-1].bias)

        self.proj = nn.Conv2d(dim, dim, 1)
        self.dropout = nn.Dropout(dropout)

        # Post-attention FFN with gaze AdaLN modulation
        self.norm_ffn = GroupNorm(dim, affine=False)
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, dim * 2, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(dim * 2, dim, 1),
        )
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(gaze_emb_dim, dim * 2),
        )
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

        # Learnable gate for residual connection (zero-init → skip-through at start)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, app_feat: torch.Tensor,
                gaze_emb: torch.Tensor,
                person_attn_bias: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x:         [B, C, H, W] gaze stream features (queries)
            app_feat:  [B, C_app, H', W'] appearance features (keys/values)
            gaze_emb:  [B, gaze_emb_dim] stage-specific gaze embedding
            person_attn_bias: [B, num_heads, 1, 1] optional per-person attention bias
        Returns:
            [B, C, H, W] updated gaze stream features
        """
        B, C, H, W = x.shape
        _, C_app, H_app, W_app = app_feat.shape

        # Compute Q, K, V
        q = self.to_q(self.norm_q(x))  # [B, C, H, W]
        k = self.to_k(self.norm_kv(app_feat))  # [B, C, H_app, W_app]
        v = self.to_v(self.norm_kv(app_feat))

        # Add gaze-conditioned bias to queries
        q_bias = self.gaze_q_bias(gaze_emb)  # [B, C]
        q = q + q_bias.unsqueeze(-1).unsqueeze(-1)

        # Reshape for multi-head attention
        def reshape_for_attn(t, h, w):
            return t.reshape(B, self.num_heads, self.head_dim, h * w).permute(0, 1, 3, 2)

        q = reshape_for_attn(q, H, W)             # [B, heads, H*W, head_dim]
        k = reshape_for_attn(k, H_app, W_app)     # [B, heads, H'*W', head_dim]
        v = reshape_for_attn(v, H_app, W_app)

        # Attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, heads, HW, H'W']

        if person_attn_bias is not None:
            attn = attn + person_attn_bias

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # [B, heads, HW, head_dim]
        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)
        out = self.proj(out)

        # Gated residual (gate starts at 0 → pure skip-through initially)
        x = x + torch.sigmoid(self.gate) * out

        # FFN with gaze AdaLN modulation
        shift, scale = self.adaLN(gaze_emb).chunk(2, dim=1)
        h = modulate(self.norm_ffn(x), shift, scale)
        x = x + self.ffn(h)

        return x


# ═══════════════════════════════════════════════════════════════════════
#  4. Personalization Head
# ═══════════════════════════════════════════════════════════════════════

class PersonalizationHead(nn.Module):
    """Lightweight per-user personalization via attention bias injection.

    Instead of FiLM scale/shift (current SubjectAdapter), this module learns
    per-user attention biases that steer cross-attention to focus on user-specific
    appearance features. This is more expressive than channel-wise scaling because
    it modulates *which spatial locations* to attend to, not just channel gains.

    Architecture:
        - Shared encoder: source eye → compact identity vector
        - Per-block bias generators: identity → attention bias per cross-attn block

    Fine-tuning protocol:
        Phase 1: Trained jointly (all params)
        Phase 2: Freeze main network, only fine-tune PersonalizationHead (~200K params)
    """
    def __init__(self, in_channels: int = 3, identity_dim: int = 128,
                 num_cross_attn_blocks: int = 2, num_heads: int = 4):
        super().__init__()
        self.identity_dim = identity_dim
        self.num_heads = num_heads

        # Compact identity encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),
            nn.GELU(),
            GroupNorm(32),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GELU(),
            GroupNorm(64),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Sequential(
            nn.Linear(64, identity_dim),
            nn.LayerNorm(identity_dim),
        )

        # Per-block attention bias + FiLM generators
        self.attn_bias_heads = nn.ModuleList()
        self.film_heads = nn.ModuleList()
        for _ in range(num_cross_attn_blocks):
            # Attention bias: [B, num_heads, 1, 1] — global per-head bias
            head = nn.Linear(identity_dim, num_heads)
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
            self.attn_bias_heads.append(head)

        # Additional FiLM for UNet blocks (like SubjectAdapter)
        self.unet_film_heads = nn.ModuleList()

    def add_unet_film_heads(self, block_channels: list):
        """Add per-UNet-block FiLM heads for backward compatibility."""
        shared_dim = self.identity_dim // 2
        self.unet_shared = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.identity_dim, shared_dim),
        )
        for ch in block_channels:
            head = nn.Linear(shared_dim, ch * 2)
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
            self.unet_film_heads.append(head)

    def encode(self, source_eye: torch.Tensor) -> torch.Tensor:
        """Encode source eye → identity embedding. Cacheable for inference."""
        feat = self.encoder(source_eye).flatten(1)
        return self.proj(feat)

    def get_attn_biases(self, identity_emb: torch.Tensor) -> list:
        """Generate per-block attention biases from cached identity embedding."""
        biases = []
        for head in self.attn_bias_heads:
            bias = head(identity_emb)  # [B, num_heads]
            biases.append(bias.unsqueeze(-1).unsqueeze(-1))  # [B, heads, 1, 1]
        return biases

    def get_unet_films(self, identity_emb: torch.Tensor) -> list:
        """Generate per-UNet-block FiLM modulations."""
        if not self.unet_film_heads:
            return None
        z = self.unet_shared(identity_emb)
        mods = []
        for head in self.unet_film_heads:
            params = head(z)
            scale, shift = params.chunk(2, dim=1)
            mods.append((
                scale.unsqueeze(-1).unsqueeze(-1),
                shift.unsqueeze(-1).unsqueeze(-1),
            ))
        return mods

    def forward(self, source_eye: torch.Tensor):
        """Full forward: encode + generate all modulations."""
        identity_emb = self.encode(source_eye)
        return {
            'identity_emb': identity_emb,
            'attn_biases': self.get_attn_biases(identity_emb),
            'unet_films': self.get_unet_films(identity_emb),
        }


# ═══════════════════════════════════════════════════════════════════════
#  5. GazeTransformDecoder (Stream 2 — Main Generator)
# ═══════════════════════════════════════════════════════════════════════

class GazeTransformDecoder(nn.Module):
    """Gaze-conditioned decoder with cross-attention to appearance features.

    Encoder–decoder architecture where:
        - Encoder: processes source eye crops with gaze conditioning
        - Decoder: cross-attends to appearance encoder features at each scale
        - Final: residual prediction (Δ on source eye, like EyeOnlyGazeDiC)

    The cross-attention at each decoder level allows the model to selectively
    retrieve identity details from the appearance stream while the gaze conditioning
    controls *how* those details are spatially rearranged.

    Resolution flow (64×128 input):
        Enc0: [B, C0, 64, 128] → Enc1: [B, C1, 32, 64] → Lat: [B, C2, 16, 32]
        Dec0: [B, C1, 32, 64] + CrossAttn(app_level1) → Dec1: [B, C0, 64, 128] + CrossAttn(app_level0)
    """
    def __init__(
        self,
        in_channels: int = 3,
        hidden_size: int = 32,
        mult_channels: list = None,
        depth: list = None,
        gaze_emb_dim: int = 64,
        num_heads: int = 4,
        affinef: int = 3,
        actfunc: str = 'gelu',
        dropout: float = 0.1,
        blockconfig: int = 2,
        cross_attn_levels: list = None,
    ):
        super().__init__()
        if mult_channels is None:
            mult_channels = [1, 2, 4, 2, 1]
        if depth is None:
            depth = [2, 2, 4, 2, 2]
        if cross_attn_levels is None:
            cross_attn_levels = [0, 1]  # cross-attention at decoder levels 0 and 1

        channels = [hidden_size * m for m in mult_channels]
        self.channels = channels
        self.depth = depth
        self.cross_attn_levels = cross_attn_levels

        # Patch embedding
        self.x_embedder = OverlapPatchEmbed(3, 1, in_channels, channels[0])

        # Stage-specific gaze projections: map gaze_emb_dim → per-stage emb_channels
        # UNetBlock expects emb_channels matching its affine input dim.
        self.gaze_projections = nn.ModuleList([
            nn.Sequential(nn.Linear(gaze_emb_dim, channels[0]), nn.LayerNorm(channels[0])),
            nn.Sequential(nn.Linear(gaze_emb_dim, channels[1]), nn.LayerNorm(channels[1])),
            nn.Sequential(nn.Linear(gaze_emb_dim, channels[2]), nn.LayerNorm(channels[2])),
        ])

        block_kwargs = dict(
            dropout=dropout, skip_scale=1, blockconfig=blockconfig,
            actfunc=actfunc, affinef=affinef,
        )

        # Encoder blocks
        self.enc_blocks = nn.ModuleList()
        for i in range(2):
            stage = nn.ModuleList()
            for j in range(depth[i]):
                stage.append(UNetBlock(
                    in_channels=channels[i],
                    out_channels=channels[i],
                    emb_channels=channels[i],
                    **block_kwargs,
                ))
            self.enc_blocks.append(stage)

        # Downsample
        self.downs = nn.ModuleList([
            Downsample(channels[0], channels[1]),
            Downsample(channels[1], channels[2]),
        ])

        # Latent blocks
        self.lat_blocks = nn.ModuleList()
        for j in range(depth[2]):
            self.lat_blocks.append(UNetBlock(
                in_channels=channels[2],
                out_channels=channels[2],
                emb_channels=channels[2],
                **block_kwargs,
            ))

        # Upsample
        self.ups = nn.ModuleList([
            Upsample(channels[2], channels[3]),
            Upsample(channels[3], channels[4]),
        ])

        # Decoder blocks with cross-attention
        self.dec_blocks = nn.ModuleList()
        self.cross_attn_blocks = nn.ModuleList()

        # AppearanceEncoder output channels (needs to match)
        app_channels = [hidden_size * m for m in [1, 2, 4]]

        for i in range(2):
            dec_idx = i + 3
            enc_idx = 1 - i
            stage = nn.ModuleList()
            for j in range(depth[dec_idx]):
                if j == 0:
                    block_in = channels[dec_idx] + channels[enc_idx]
                else:
                    block_in = channels[dec_idx]
                stage.append(UNetBlock(
                    in_channels=block_in,
                    out_channels=channels[dec_idx],
                    emb_channels=channels[min(enc_idx, 2)],
                    **block_kwargs,
                ))
            self.dec_blocks.append(stage)

            # Cross-attention after all blocks in this decoder level
            if i in cross_attn_levels:
                app_ch = app_channels[enc_idx]
                self.cross_attn_blocks.append(
                    GazeCrossAttentionBlock(
                        dim=channels[dec_idx],
                        app_dim=app_ch,
                        gaze_emb_dim=channels[min(enc_idx, 2)],
                        num_heads=num_heads,
                        dropout=dropout,
                    )
                )
            else:
                self.cross_attn_blocks.append(None)

        # Final layer
        self.final_layer = DiCFinalLayer(channels[4], in_channels, channels[0])

    def forward(self, eye_input: torch.Tensor, gaze_embs: list,
                app_features: list = None,
                attn_biases: list = None,
                subject_mods: list = None) -> torch.Tensor:
        """
        Args:
            eye_input:    [B, C_in, H, W] source eye crops
            gaze_embs:    list of 3 [B, gaze_emb_dim] per-stage gaze embeddings
            app_features: list of 3 appearance feature maps from AppearanceEncoder
            attn_biases:  list of [B, heads, 1, 1] per-user attention biases
            subject_mods: list of (scale, shift) tuples from PersonalizationHead
        Returns:
            [B, C_in, H, W] generated eye image
        """
        block_idx = 0

        def _sub(idx):
            if subject_mods is not None and idx < len(subject_mods):
                return subject_mods[idx]
            return None, None

        x = self.x_embedder(eye_input)

        # Project gaze embeddings to per-stage channel dimensions
        emb_enc0 = self.gaze_projections[0](gaze_embs[0])
        emb_enc1 = self.gaze_projections[1](gaze_embs[1])
        emb_lat  = self.gaze_projections[2](gaze_embs[2])

        # Encoder
        skip_features = []

        for block in self.enc_blocks[0]:
            ss, sh = _sub(block_idx)
            x = block(x, emb_enc0, subject_scale=ss, subject_shift=sh)
            block_idx += 1
        skip_features.append(x)
        x = self.downs[0](x)

        for block in self.enc_blocks[1]:
            ss, sh = _sub(block_idx)
            x = block(x, emb_enc1, subject_scale=ss, subject_shift=sh)
            block_idx += 1
        skip_features.append(x)
        x = self.downs[1](x)

        # Latent
        for block in self.lat_blocks:
            ss, sh = _sub(block_idx)
            x = block(x, emb_lat, subject_scale=ss, subject_shift=sh)
            block_idx += 1

        # Decoder with cross-attention
        emb_dec = [emb_enc1, emb_enc0]
        ca_idx = 0

        for i in range(2):
            x = self.ups[i](x)
            skip = skip_features[1 - i]
            x = torch.cat([x, skip], dim=1)

            for block in self.dec_blocks[i]:
                ss, sh = _sub(block_idx)
                x = block(x, emb_dec[i], subject_scale=ss, subject_shift=sh)
                block_idx += 1

            # Cross-attention with appearance features
            if self.cross_attn_blocks[i] is not None and app_features is not None:
                app_idx = 1 - i  # dec0 attends to app_level1, dec1 to app_level0
                attn_bias = attn_biases[ca_idx] if attn_biases is not None else None
                x = self.cross_attn_blocks[i](
                    x, app_features[app_idx], emb_dec[i],
                    person_attn_bias=attn_bias,
                )
                ca_idx += 1

        # Residual prediction
        delta = self.final_layer(x, emb_enc0)
        return torch.clamp(eye_input + delta, min=-1.0, max=1.0)


# ═══════════════════════════════════════════════════════════════════════
#  6. DualStreamGazeNet (Full Model Wrapper)
# ═══════════════════════════════════════════════════════════════════════

class DualStreamGazeNet(nn.Module):
    """Dual-Stream Personalized Gaze Redirection Network.

    Combines:
        1. AppearanceEncoder — extracts multi-scale identity features (no gaze)
        2. GazeTransformDecoder — gaze-conditioned generation with cross-attention
        3. DeltaGazeEmbedder — delta-gaze angular embedding
        4. PersonalizationHead — lightweight per-user adaptation

    Training phases:
        Phase 1: Train all components jointly on multi-user data
        Phase 2: Freeze AppearanceEncoder + GazeTransformDecoder,
                 fine-tune only PersonalizationHead for new users (~200K params)

    Inference fast-path:
        1. Cache appearance features: app_feats = model.encode_appearance(ref_eye)
        2. Cache person mods: person = model.personalize(ref_eye)
        3. Per-frame: out = model.redirect(source_eye, source_gaze, target_gaze, head,
                                           app_feats, person)
    """
    def __init__(self, config: dict):
        super().__init__()

        in_channels = config.get('in_channels', 3)
        hidden_size = config.get('hidden_size', 32)
        gaze_emb_dim = config.get('gaze_dim', 64)
        mult_channels = config.get('mult_channels', [1, 2, 4, 2, 1])
        depth = config.get('depth', [2, 2, 4, 2, 2])
        num_heads = config.get('num_heads', 4)
        dropout = config.get('dropout', 0.1)
        affinef = config.get('affinef', 3)
        actfunc = config.get('actfunc', 'gelu')
        blockconfig = config.get('blockconfig', 2)
        app_mult = config.get('app_mult', [1, 2, 4])
        app_depth = config.get('app_depth', [2, 2, 2])
        cross_attn_levels = config.get('cross_attn_levels', [0, 1])

        # Stream 1: Appearance Encoder
        self.appearance_encoder = AppearanceEncoder(
            in_channels=in_channels,
            base_ch=hidden_size,
            mult=app_mult,
            depth=app_depth,
        )

        # Delta-Gaze Embedder
        self.delta_gaze_embedder = DeltaGazeEmbedder(
            angle_dim=2,
            head_dim=2,
            freq_bands=config.get('freq_bands', 8),
            hidden_dim=config.get('gaze_hidden_dim', 256),
            out_dim=gaze_emb_dim,
            num_stages=3,
        )

        # Stream 2: Gaze Transform Decoder
        self.gaze_decoder = GazeTransformDecoder(
            in_channels=in_channels,
            hidden_size=hidden_size,
            mult_channels=mult_channels,
            depth=depth,
            gaze_emb_dim=gaze_emb_dim,
            num_heads=num_heads,
            affinef=affinef,
            actfunc=actfunc,
            dropout=dropout,
            blockconfig=blockconfig,
            cross_attn_levels=cross_attn_levels,
        )

        # Personalization Head
        personal_config = config.get('personalization', {})
        self.personal_head = None
        if personal_config.get('enabled', True):
            self.personal_head = PersonalizationHead(
                in_channels=in_channels,
                identity_dim=personal_config.get('identity_dim', 128),
                num_cross_attn_blocks=len(cross_attn_levels),
                num_heads=num_heads,
            )
            if personal_config.get('unet_film', False):
                block_channels = self._get_block_channels()
                self.personal_head.add_unet_film_heads(block_channels)

        # Optional: head image encoder for face context (backward compat)
        self.head_encoder = None
        self.head_fusion = None
        if config.get('head_img_encoder', False):
            from models.gaze_dic import HeadImageEncoder, HeadCondFusion
            head_img_dim = config.get('head_img_dim', 128)
            head_fusion_dim = config.get('head_fusion_dim', 128)
            self.head_encoder = HeadImageEncoder(
                in_channels=in_channels, base_ch=32, out_dim=head_img_dim,
            )
            self.head_fusion = HeadCondFusion(
                img_dim=head_img_dim, label_dim=0, out_dim=head_fusion_dim,
            )

    def _get_block_channels(self) -> list:
        """Auto-detect block output channels from the decoder for FiLM heads."""
        channels = []
        for stage in self.gaze_decoder.enc_blocks:
            for block in stage:
                channels.append(block.out_channels)
        for block in self.gaze_decoder.lat_blocks:
            channels.append(block.out_channels)
        for stage in self.gaze_decoder.dec_blocks:
            for block in stage:
                channels.append(block.out_channels)
        return channels

    def encode_appearance(self, source_eye: torch.Tensor) -> list:
        """Extract multi-scale appearance features. Cacheable per subject."""
        return self.appearance_encoder(source_eye)

    def forward(self, source_eye_crops: torch.Tensor,
                source_gaze: torch.Tensor, target_gaze: torch.Tensor,
                head_pose: torch.Tensor,
                source_face: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            source_eye_crops: [B, 3, H, W] source eye crops (width-concat)
            source_gaze:      [B, 2] source gaze angles
            target_gaze:      [B, 2] target gaze angles
            head_pose:        [B, 2] head pose angles
            source_face:      [B, 3, 256, 256] optional face for head encoder
        Returns:
            generated_eyes: [B, 3, H, W] redirected eye crops
        """
        # Stream 1: Appearance features (no gaze conditioning)
        app_features = self.appearance_encoder(source_eye_crops)

        # Delta-gaze embeddings (per-stage)
        gaze_embs = self.delta_gaze_embedder(source_gaze, target_gaze, head_pose)

        # Personalization (if available)
        attn_biases = None
        subject_mods = None
        if self.personal_head is not None:
            person_out = self.personal_head(source_eye_crops)
            attn_biases = person_out['attn_biases']
            subject_mods = person_out['unet_films']

        # Stream 2: Gaze-conditioned generation with cross-attention
        generated = self.gaze_decoder(
            source_eye_crops, gaze_embs,
            app_features=app_features,
            attn_biases=attn_biases,
            subject_mods=subject_mods,
        )

        return generated

    def paste_eyes(self, generated_eyes, source_image, eye_bbox, blend_margin=4):
        """Paste generated eyes back onto the source face image.

        Reuses the same logic as EyeOnlyWrapper.paste_eyes for compatibility.
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
        if margin <= 0:
            return torch.ones(1, h, w, device=device)
        def edge_ramp(size, m):
            t = torch.ones(size, device=device)
            if m > 0:
                ramp = torch.linspace(
                    1 / (m + 1), m / (m + 1),
                    steps=min(m, size // 2), device=device,
                )
                t[:len(ramp)] = ramp
                t[size - len(ramp):] = ramp.flip(0)
            return t
        row = edge_ramp(h, margin)
        col = edge_ramp(w, margin)
        return torch.outer(row, col).unsqueeze(0)

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freeze_for_personalization(self):
        """Phase 2: freeze everything except PersonalizationHead."""
        for param in self.parameters():
            param.requires_grad = False
        if self.personal_head is not None:
            for param in self.personal_head.parameters():
                param.requires_grad = True

    def get_personalization_params(self) -> list:
        """Return only PersonalizationHead parameters for Phase 2 optimizer."""
        if self.personal_head is not None:
            return list(self.personal_head.parameters())
        return []
