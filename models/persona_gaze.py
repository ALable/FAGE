"""
PersonaGazeNet (PGN) — Personalized Gaze Redirection via Reference Tokens + HyperLoRA.

See ``PersonaGazeNet.md`` for the full design rationale. This file is the
minimal-but-functional prototype: a self-contained model that can run a forward
pass end-to-end on random tensors and is wired so the loss / training pipeline
in ``train_persona_gaze.py`` can drive it.

Module map:
    RefTokenizer       : per-frame CNN, produces spatial token bank for K refs.
    RefCrossAttn       : multi-head cross-attention from decoder feats → token bank.
    LoRAConv2d         : Conv2d with externally-provided low-rank weight delta.
    HyperLoRAHead      : MLP that maps pooled ref tokens to per-conv (A, B) matrices.
    GazeEmbedder       : head+gaze MLP fusion (same shape as ConditionEmbedder).
    PGNEncoder         : 3-stage UNet encoder, AdaLN-Zero conditioned on gaze only.
    PGNDecoder         : 3-stage UNet decoder, gaze AdaLN + RefCrossAttn + LoRA.
    PersonaGazeNet     : top-level orchestrator with three operating modes:
                           - mode='K0'     : K=1, gaze-only fallback (Phase 1 train)
                           - mode='Kshot'  : K∈{1,4,8}, RFA on (Phase 2/3 train + zero-shot infer)
                           - mode='hyper'  : Kshot + HyperLoRA predicted ΔW    (Phase 3 + infer)

Design notes:
    - Decoder cross-attention gates initialised to ~0 → identity behaviour at start.
    - HyperLoRA B-matrices zero-init → ΔW=0 at start.
    - Final output is residual: source_eye + tanh(scale)·δI, clamped to [-1, 1].
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.gaze_dic import (
    GroupNorm,
    OverlapPatchEmbed,
    Downsample,
    Upsample,
    UNetBlock,
    ConditionEmbedder,
    DiCFinalLayer,
)


# ──────────────────────────────────────────────────────────────────────────────
#   1. Reference Tokenizer
# ──────────────────────────────────────────────────────────────────────────────

class RefTokenizer(nn.Module):
    """Encode an eye crop into a spatial token sequence.

    Input  : [B, 3, H_e, W_e]   (default 64×128)
    Output : tokens [B, N, C_t] and a pooled vector [B, C_t]
             where N = (H_e/down) * (W_e/down).

    We keep a (H_e/4, W_e/4) = (16, 32) spatial grid so the cross-attention can
    do *positional* identity retrieval (e.g. "iris is in the middle-right").
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_ch: int = 32,
        token_dim: int = 128,
        max_tokens: int = 16 * 32,  # for positional embedding table size
    ):
        super().__init__()
        self.stem = OverlapPatchEmbed(3, 1, in_channels, base_ch)
        # 64×128 → 32×64 → 16×32
        self.down1 = Downsample(base_ch, base_ch * 2)
        self.down2 = Downsample(base_ch * 2, base_ch * 4)
        self.norm = GroupNorm(base_ch * 4)
        self.act = nn.GELU()
        self.project = nn.Conv2d(base_ch * 4, token_dim, kernel_size=1)
        # Learned 2D positional embedding (flat over HW for simplicity).
        self.pos_embed = nn.Parameter(torch.zeros(1, max_tokens, token_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.pool_proj = nn.Linear(token_dim, token_dim)
        self.token_dim = token_dim

    def forward(self, eye: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (tokens [B, N, C_t], pooled [B, C_t])."""
        x = self.stem(eye)          # [B, base,   H,   W  ]
        x = self.down1(x)           # [B, base*2, H/2, W/2]
        x = self.down2(x)           # [B, base*4, H/4, W/4]
        x = self.act(self.norm(x))
        x = self.project(x)         # [B, C_t,   H/4, W/4]
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # [B, H*W, C_t]
        N = tokens.shape[1]
        tokens = tokens + self.pos_embed[:, :N]
        pooled = self.pool_proj(tokens.mean(dim=1))  # [B, C_t]
        return tokens, pooled


# ──────────────────────────────────────────────────────────────────────────────
#   2. Reference Cross-Attention block
# ──────────────────────────────────────────────────────────────────────────────

class RefCrossAttn(nn.Module):
    """Multi-head cross-attention with a learnable scalar gate.

    q : decoder features  [B, C_dec, h, w] → [B, h*w, C_attn]
    kv: ref token bank    [B, K*N, C_t]    → [B, K*N, C_attn]
    out: [B, C_dec, h, w] added back with sigmoid(gate) (gate init: large negative
         so the block starts as identity).
    """

    def __init__(
        self,
        dec_channels: int,
        token_dim: int,
        attn_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert attn_dim % num_heads == 0
        self.num_heads = num_heads
        self.attn_dim = attn_dim
        self.head_dim = attn_dim // num_heads

        self.q_norm = GroupNorm(dec_channels, affine=True)
        self.q_proj = nn.Conv2d(dec_channels, attn_dim, kernel_size=1)

        self.kv_norm = nn.LayerNorm(token_dim)
        self.k_proj = nn.Linear(token_dim, attn_dim, bias=False)
        self.v_proj = nn.Linear(token_dim, attn_dim, bias=False)

        self.out_proj = nn.Conv2d(attn_dim, dec_channels, kernel_size=1)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        # Scalar gate, init at -5 so sigmoid(-5) ≈ 6.7e-3 → near-identity start.
        self.gate = nn.Parameter(torch.full((1,), -5.0))
        self.dropout = nn.Dropout(dropout)

    def forward(self, dec_feat: torch.Tensor, ref_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            dec_feat   : [B, C_dec, h, w]
            ref_tokens : [B, M, C_t]  (M = K*N)
        Returns:
            [B, C_dec, h, w]  (gated residual added to dec_feat)
        """
        B, C_dec, h, w = dec_feat.shape

        q = self.q_norm(dec_feat)
        q = self.q_proj(q).flatten(2).transpose(1, 2)              # [B, h*w, A]

        kv = self.kv_norm(ref_tokens)
        k = self.k_proj(kv)                                         # [B, M, A]
        v = self.v_proj(kv)                                         # [B, M, A]

        # Reshape for multi-head: [B, heads, T, head_dim]
        def split(t):
            return t.view(t.shape[0], t.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        qh, kh, vh = split(q), split(k), split(v)

        # PyTorch ≥2.0 scaled_dot_product_attention is memory-friendly.
        attn = F.scaled_dot_product_attention(qh, kh, vh, dropout_p=self.dropout.p if self.training else 0.0)
        # attn: [B, heads, h*w, head_dim] → [B, h*w, A]
        attn = attn.transpose(1, 2).contiguous().view(B, h * w, self.attn_dim)
        attn = attn.transpose(1, 2).view(B, self.attn_dim, h, w)

        delta = self.out_proj(attn)
        return dec_feat + torch.sigmoid(self.gate) * delta


# ──────────────────────────────────────────────────────────────────────────────
#   3. LoRA-augmented Conv2d
# ──────────────────────────────────────────────────────────────────────────────

class LoRAConv2d(nn.Module):
    """Conv2d that optionally accepts an external (A, B) low-rank weight delta.

    When ``A``/``B`` are None, behaves like a vanilla Conv2d.
    When given, computes:

        ΔW[c_out, c_in, k_h, k_w] = (α/r) · (B @ A)[c_out, c_in] · δ(k_h=k_w=center)

    i.e. the LoRA delta is applied as a 1×1 add-on to the convolution kernel
    centre — equivalent to a (B @ A)/(α/r) extra 1×1 conv that has been folded
    into the main 3×3.  This keeps inference cost at zero once merged.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding)

    def forward(self,
                x: torch.Tensor,
                A: Optional[torch.Tensor] = None,
                B: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x : [B, C_in, h, w]
            A : optional, [B, rank, C_in]   (per-sample LoRA matrix from HyperLoRA)
            B : optional, [B, C_out, rank]  (per-sample LoRA matrix from HyperLoRA)
        """
        out = self.conv(x)
        if A is None or B is None:
            return out

        # Per-sample 1×1 conv equivalent: y_extra[b] = (B[b] @ A[b]) @ x_pool[b]
        # We do it as a grouped 1×1 conv with per-batch weights.
        BSZ, C_in, h, w = x.shape
        # ΔW: [B, C_out, C_in]
        delta_w = torch.bmm(B, A) * self.scaling
        # Implement per-sample matmul as einsum.
        # x:        [B, C_in, H, W]
        # delta_w:  [B, C_out, C_in]
        # extra:    [B, C_out, H, W]
        extra = torch.einsum("bcij,boc->boij", x, delta_w)
        return out + extra


# ──────────────────────────────────────────────────────────────────────────────
#   4. HyperLoRA Head
# ──────────────────────────────────────────────────────────────────────────────

class HyperLoRAHead(nn.Module):
    """Predict (A, B) LoRA matrices for a list of decoder convs from pooled refs.

    Inputs : [B, C_t]    (mean over (K * N) token-bank dim, optionally attn-pool)
    Outputs: list of (A, B) tuples, A=[B, r, C_in_l], B=[B, C_out_l, r]
             B is zero-initialised so the LoRA starts as a no-op.
    """

    def __init__(
        self,
        token_dim: int,
        conv_shapes: List[Tuple[int, int]],
        rank: int = 8,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.rank = rank
        self.shared = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.A_heads = nn.ModuleList()
        self.B_heads = nn.ModuleList()
        for (c_in, c_out) in conv_shapes:
            a = nn.Linear(hidden_dim, rank * c_in)
            b = nn.Linear(hidden_dim, c_out * rank)
            # A initialised with small kaiming, B initialised to zero → ΔW = 0.
            nn.init.kaiming_uniform_(a.weight, a=math.sqrt(5))
            nn.init.zeros_(a.bias)
            nn.init.zeros_(b.weight)
            nn.init.zeros_(b.bias)
            self.A_heads.append(a)
            self.B_heads.append(b)
        self.conv_shapes = conv_shapes

    def forward(self, pooled_ref: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        z = self.shared(pooled_ref)            # [B, hidden]
        out = []
        for (c_in, c_out), a_lin, b_lin in zip(self.conv_shapes, self.A_heads, self.B_heads):
            A = a_lin(z).view(-1, self.rank, c_in)
            B = b_lin(z).view(-1, c_out, self.rank)
            out.append((A, B))
        return out


# ──────────────────────────────────────────────────────────────────────────────
#   5. PGN UNet (encoder + decoder)
# ──────────────────────────────────────────────────────────────────────────────

class PGNEncoder(nn.Module):
    """3-stage UNet encoder. AdaLN-Zero conditioned on gaze only.

    Resolution flow (64 × 128 input):  64×128 → 32×64 → 16×32 (latent).
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_size: int = 32,
        mult_channels=(1, 2, 4),
        depth=(2, 2, 4),
        gaze_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        channels = [hidden_size * m for m in mult_channels]
        self.channels = channels

        self.x_embedder = OverlapPatchEmbed(3, 1, in_channels, channels[0])
        self.gaze_embedders = nn.ModuleList([
            ConditionEmbedder(gaze_dim, channels[i]) for i in range(3)
        ])

        self.enc_blocks = nn.ModuleList()
        for i in range(2):
            stage = nn.ModuleList()
            for j in range(depth[i]):
                stage.append(UNetBlock(
                    in_channels=channels[i],
                    out_channels=channels[i],
                    emb_channels=channels[i],
                    dropout=dropout,
                ))
            self.enc_blocks.append(stage)
        self.downs = nn.ModuleList([
            Downsample(channels[0], channels[1]),
            Downsample(channels[1], channels[2]),
        ])

        self.lat_blocks = nn.ModuleList()
        for j in range(depth[2]):
            self.lat_blocks.append(UNetBlock(
                in_channels=channels[2],
                out_channels=channels[2],
                emb_channels=channels[2],
                dropout=dropout,
            ))

    def forward(self, x: torch.Tensor, gaze_cond: torch.Tensor):
        """Returns (latent, skip_features list ordered enc0, enc1)."""
        emb0 = self.gaze_embedders[0](gaze_cond)
        emb1 = self.gaze_embedders[1](gaze_cond)
        emb2 = self.gaze_embedders[2](gaze_cond)

        h = self.x_embedder(x)
        for blk in self.enc_blocks[0]:
            h = blk(h, emb0)
        skip0 = h
        h = self.downs[0](h)
        for blk in self.enc_blocks[1]:
            h = blk(h, emb1)
        skip1 = h
        h = self.downs[1](h)
        for blk in self.lat_blocks:
            h = blk(h, emb2)
        return h, [skip0, skip1], (emb0, emb1, emb2)


class PGNDecoderStage(nn.Module):
    """One decoder stage: upsample → concat skip → UNetBlock(s) → RFA → LoRAConv.

    The LoRAConv adds an extra ID-conditioned 1×1 residual after the main UNet
    blocks; it is gated by the externally-supplied (A, B). The RFA module is
    inserted *between* the UNet blocks and the LoRA conv to let identity tokens
    influence the residual computation.
    """

    def __init__(
        self,
        in_ch_skip: int,
        in_ch_up: int,
        out_ch: int,
        gaze_emb_ch: int,
        token_dim: int,
        depth: int = 2,
        dropout: float = 0.1,
        use_rfa: bool = True,
        use_lora: bool = True,
        lora_rank: int = 8,
    ):
        super().__init__()
        self.up = Upsample(in_ch_up, out_ch)
        block_in = out_ch + in_ch_skip
        blocks = []
        for j in range(depth):
            bi = block_in if j == 0 else out_ch
            blocks.append(UNetBlock(
                in_channels=bi,
                out_channels=out_ch,
                emb_channels=gaze_emb_ch,
                dropout=dropout,
            ))
        self.blocks = nn.ModuleList(blocks)

        self.rfa = RefCrossAttn(out_ch, token_dim) if use_rfa else None
        self.lora_conv = LoRAConv2d(out_ch, out_ch, kernel_size=3,
                                    rank=lora_rank) if use_lora else None
        self.act = nn.GELU()

    def forward(
        self,
        h: torch.Tensor,
        skip: torch.Tensor,
        gaze_emb: torch.Tensor,
        ref_tokens: Optional[torch.Tensor],
        lora_AB: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        h = self.up(h)
        h = torch.cat([h, skip], dim=1)
        for blk in self.blocks:
            h = blk(h, gaze_emb)
        if self.rfa is not None and ref_tokens is not None:
            h = self.rfa(h, ref_tokens)
        if self.lora_conv is not None:
            A, B = (None, None) if lora_AB is None else lora_AB
            h = self.lora_conv(self.act(h), A=A, B=B)
        return h


class PGNDecoder(nn.Module):
    """Stack of decoder stages going from latent → input resolution."""

    def __init__(
        self,
        enc_channels: List[int],
        out_channels: int = 3,
        token_dim: int = 128,
        gaze_dim: int = 64,
        depth=(2, 2),
        dropout: float = 0.1,
        lora_rank: int = 8,
    ):
        super().__init__()
        # Mirror: dec0 = enc1, dec1 = enc0
        self.gaze_embedder_dec0 = ConditionEmbedder(gaze_dim, enc_channels[1])
        self.gaze_embedder_dec1 = ConditionEmbedder(gaze_dim, enc_channels[0])

        self.stage0 = PGNDecoderStage(
            in_ch_skip=enc_channels[1],
            in_ch_up=enc_channels[2],
            out_ch=enc_channels[1],
            gaze_emb_ch=enc_channels[1],
            token_dim=token_dim,
            depth=depth[0],
            dropout=dropout,
            use_rfa=True,
            use_lora=True,
            lora_rank=lora_rank,
        )
        self.stage1 = PGNDecoderStage(
            in_ch_skip=enc_channels[0],
            in_ch_up=enc_channels[1],
            out_ch=enc_channels[0],
            gaze_emb_ch=enc_channels[0],
            token_dim=token_dim,
            depth=depth[1],
            dropout=dropout,
            use_rfa=True,
            use_lora=True,
            lora_rank=lora_rank,
        )
        self.final = DiCFinalLayer(enc_channels[0], out_channels, enc_channels[0])

    def lora_conv_shapes(self) -> List[Tuple[int, int]]:
        return [
            (self.stage0.lora_conv.in_channels, self.stage0.lora_conv.out_channels),
            (self.stage1.lora_conv.in_channels, self.stage1.lora_conv.out_channels),
        ]

    def forward(
        self,
        latent: torch.Tensor,
        skips: List[torch.Tensor],
        gaze_cond: torch.Tensor,
        ref_tokens: Optional[torch.Tensor] = None,
        lora_list: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> torch.Tensor:
        emb0 = self.gaze_embedder_dec0(gaze_cond)  # mirrors enc1
        emb1 = self.gaze_embedder_dec1(gaze_cond)  # mirrors enc0

        lora0 = lora_list[0] if lora_list else None
        lora1 = lora_list[1] if lora_list else None
        h = self.stage0(latent, skips[1], emb0, ref_tokens, lora0)
        h = self.stage1(h, skips[0], emb1, ref_tokens, lora1)
        out = self.final(h, emb1)
        return out


# ──────────────────────────────────────────────────────────────────────────────
#   6. PersonaGazeNet (orchestrator)
# ──────────────────────────────────────────────────────────────────────────────

class PersonaGazeNet(nn.Module):
    """End-to-end personalised gaze redirection network.

    Forward signature:
        out = model(source_eye, gaze_cond, ref_eyes=None)

    Where:
        source_eye : [B, 3, H_e, W_e]   driving identity backbone
        gaze_cond  : [B, 2, gaze_dim]   (head_emb, gaze_emb)
        ref_eyes   : optional [B, K, 3, H_e, W_e] reference frames. If None we
                     fall back to gaze-only (Phase 1) behaviour.

    Output is residual: source_eye + scale * δI, clamped to [-1, 1].
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_size: int = 32,
        mult_channels=(1, 2, 4),
        enc_depth=(2, 2, 4),
        dec_depth=(2, 2),
        gaze_dim: int = 64,
        token_dim: int = 128,
        ref_base_ch: int = 32,
        lora_rank: int = 8,
        use_hyper_lora: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.use_hyper_lora = use_hyper_lora
        self.token_dim = token_dim

        self.encoder = PGNEncoder(
            in_channels=in_channels,
            hidden_size=hidden_size,
            mult_channels=mult_channels,
            depth=enc_depth,
            gaze_dim=gaze_dim,
            dropout=dropout,
        )
        enc_channels = self.encoder.channels
        self.decoder = PGNDecoder(
            enc_channels=enc_channels,
            out_channels=in_channels,
            token_dim=token_dim,
            gaze_dim=gaze_dim,
            depth=dec_depth,
            dropout=dropout,
            lora_rank=lora_rank,
        )

        self.ref_tokenizer = RefTokenizer(
            in_channels=in_channels,
            base_ch=ref_base_ch,
            token_dim=token_dim,
        )

        self.hyper_lora = None
        if use_hyper_lora:
            self.hyper_lora = HyperLoRAHead(
                token_dim=token_dim,
                conv_shapes=self.decoder.lora_conv_shapes(),
                rank=lora_rank,
            )

        # Final residual gate: model outputs δI · tanh(scale).
        # Init at atanh(0.05) ≈ 0.05 — tiny residual at start (≈identity), but
        # nonzero so gradients can flow into encoder/decoder from step 1.
        self.delta_scale = nn.Parameter(torch.tensor(0.05))

    # ── API ───────────────────────────────────────────────────────────────────

    def encode_refs(self, ref_eyes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenise K reference eyes into token bank + pooled summary.

        Args:
            ref_eyes : [B, K, 3, H_e, W_e]
        Returns:
            tokens   : [B, K*N, C_t]
            pooled   : [B, C_t]
        """
        B, K, C, H, W = ref_eyes.shape
        flat = ref_eyes.reshape(B * K, C, H, W)
        tokens, pooled = self.ref_tokenizer(flat)
        # tokens: [B*K, N, C_t]
        N = tokens.shape[1]
        tokens = tokens.reshape(B, K * N, self.token_dim)
        pooled = pooled.reshape(B, K, self.token_dim).mean(dim=1)
        return tokens, pooled

    def forward(
        self,
        source_eye: torch.Tensor,
        gaze_cond: torch.Tensor,
        ref_eyes: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            source_eye : [B, 3, H_e, W_e]
            gaze_cond  : [B, 2, gaze_dim]  (head+gaze fused upstream by gaze MLP)
            ref_eyes   : [B, K, 3, H_e, W_e] or None
            return_aux : if True also return dict with intermediate tensors
        """
        ref_tokens = None
        lora_list = None
        if ref_eyes is not None:
            ref_tokens, ref_pooled = self.encode_refs(ref_eyes)
            if self.hyper_lora is not None:
                lora_list = self.hyper_lora(ref_pooled)

        latent, skips, _ = self.encoder(source_eye, gaze_cond)
        delta = self.decoder(latent, skips, gaze_cond,
                             ref_tokens=ref_tokens,
                             lora_list=lora_list)
        # Residual prediction with a learned global scale (init 0 → identity behaviour).
        out = torch.clamp(source_eye + torch.tanh(self.delta_scale) * delta,
                          min=-1.0, max=1.0)
        if return_aux:
            return out, {
                "delta": delta,
                "ref_tokens": ref_tokens,
                "lora_list": lora_list,
            }
        return out

    # ── helpers ───────────────────────────────────────────────────────────────

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def num_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def set_phase(self, phase: str) -> None:
        """Freeze / unfreeze modules to match the 3-phase curriculum.

        phase == 'phase1' : train encoder + decoder (gaze only), freeze ref + hyper.
        phase == 'phase2' : also unfreeze ref_tokenizer + decoder.rfa modules.
        phase == 'phase3' : everything trainable.
        """
        def _set(module, flag):
            for p in module.parameters():
                p.requires_grad = flag

        if phase == 'phase1':
            _set(self.encoder, True)
            _set(self.decoder, True)
            _set(self.ref_tokenizer, False)
            # Freeze RFA gates by zeroing requires_grad on RFA params.
            for m in self.decoder.modules():
                if isinstance(m, RefCrossAttn):
                    _set(m, False)
            if self.hyper_lora is not None:
                _set(self.hyper_lora, False)
        elif phase == 'phase2':
            _set(self.encoder, True)
            _set(self.decoder, True)
            _set(self.ref_tokenizer, True)
            for m in self.decoder.modules():
                if isinstance(m, RefCrossAttn):
                    _set(m, True)
            if self.hyper_lora is not None:
                _set(self.hyper_lora, False)
        elif phase == 'phase3':
            _set(self, True)
        else:
            raise ValueError(f"Unknown phase: {phase}")
