"""
Personalized Gaze Memory Adapter.

PGMA keeps the existing EyeOnlyGazeDiC backbone intact and adds a lightweight,
gaze-aware subject memory path. The adapter encodes source-eye appearance into
spatial memory tokens, queries them with the target head/gaze embedding, and
produces per-block FiLM parameters for the generator.
"""
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from models.gaze_dic import Downsample, GroupNorm


Modulation = Tuple[torch.Tensor, torch.Tensor]


class MemoryEncoderBlock(nn.Module):
    """Unconditional residual block used by the subject memory encoder."""

    def __init__(self, in_channels: int, out_channels: int, num_groups: int = 32):
        super().__init__()
        self.norm0 = GroupNorm(in_channels, num_groups, min_channels_per_group=1, affine=False)
        self.conv0 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = GroupNorm(out_channels)
        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.gate = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x) if self.skip is not None else x
        h = self.conv0(self.act(self.norm0(x)))
        h = self.conv1(self.act(self.norm1(h)))
        gate = torch.sigmoid(self.gate(h))
        return gate * h + (1.0 - gate) * residual


class SubjectMemoryEncoder(nn.Module):
    """Encode source-eye appearance into a compact set of identity tokens."""

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        token_dim: int = 128,
        token_grid: Sequence[int] = (2, 4),
    ):
        super().__init__()
        if len(token_grid) != 2:
            raise ValueError("token_grid must contain [height, width].")

        self.token_grid = tuple(int(v) for v in token_grid)
        self.level0 = MemoryEncoderBlock(in_channels, base_channels)
        self.down0 = Downsample(base_channels, base_channels * 2)
        self.level1 = MemoryEncoderBlock(base_channels * 2, base_channels * 2)
        self.down1 = Downsample(base_channels * 2, base_channels * 4)
        self.level2 = MemoryEncoderBlock(base_channels * 4, base_channels * 4)
        self.token_proj = nn.Conv2d(base_channels * 4, token_dim, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d(self.token_grid)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.token_grid[0] * self.token_grid[1], token_dim))
        self.norm = nn.LayerNorm(token_dim)

    def forward(self, source_eye: torch.Tensor) -> torch.Tensor:
        x = self.level0(source_eye)
        x = self.down0(x)
        x = self.level1(x)
        x = self.down1(x)
        x = self.level2(x)
        x = self.pool(self.token_proj(x))
        tokens = x.flatten(2).transpose(1, 2)
        return self.norm(tokens + self.pos_embed)


class PersonalizedGazeAdapter(nn.Module):
    """Generate gaze-aware per-block FiLM parameters from subject memory tokens.

    The output matches SubjectAdapter's modulation API:
        list[(scale, shift)], each [B, C_block, 1, 1].
    """

    requires_gaze = True

    def __init__(
        self,
        in_channels: int = 3,
        gaze_dim: int = 64,
        block_channels: Optional[Sequence[int]] = None,
        token_dim: int = 128,
        base_channels: int = 32,
        token_grid: Sequence[int] = (2, 4),
        num_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        if block_channels is None:
            block_channels = [32] * 2 + [64] * 2 + [128] * 4 + [64] * 2 + [32] * 2
        if token_dim % num_heads != 0:
            raise ValueError("token_dim must be divisible by num_heads.")

        self.gaze_dim = gaze_dim
        self.encoder = SubjectMemoryEncoder(
            in_channels=in_channels,
            base_channels=base_channels,
            token_dim=token_dim,
            token_grid=token_grid,
        )
        self.query_mlp = nn.Sequential(
            nn.Linear(gaze_dim * 2, token_dim),
            nn.SiLU(),
            nn.Linear(token_dim, token_dim),
        )
        self.memory_attn = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.context_norm = nn.LayerNorm(token_dim)
        self.shared_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(token_dim, token_dim),
        )

        self.block_modulators = nn.ModuleList()
        for channels in block_channels:
            head = nn.Linear(token_dim, int(channels) * 2)
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
            self.block_modulators.append(head)

    def encode(self, source_eye: torch.Tensor) -> torch.Tensor:
        """Encode reference eye crops once for a subject and cache the tokens."""
        return self.encoder(source_eye)

    def _build_query(self, gaze_condition: torch.Tensor) -> torch.Tensor:
        if gaze_condition.dim() == 2:
            gaze_pair = torch.cat([gaze_condition, gaze_condition], dim=-1)
        elif gaze_condition.dim() == 3 and gaze_condition.shape[1] >= 2:
            gaze_pair = torch.cat([gaze_condition[:, 0], gaze_condition[:, 1]], dim=-1)
        else:
            raise ValueError("gaze_condition must be [B, D] or [B, 2, D].")
        return self.query_mlp(gaze_pair).unsqueeze(1)

    def get_modulations(
        self,
        subject_tokens: torch.Tensor,
        gaze_condition: torch.Tensor,
    ) -> List[Modulation]:
        """Create per-block modulations from cached subject tokens."""
        query = self._build_query(gaze_condition)
        context, _ = self.memory_attn(query=query, key=subject_tokens, value=subject_tokens, need_weights=False)
        context = self.context_norm(context + query).squeeze(1)
        context = self.shared_proj(context)

        mods: List[Modulation] = []
        for head in self.block_modulators:
            params = head(context)
            scale, shift = params.chunk(2, dim=1)
            mods.append((scale.unsqueeze(-1).unsqueeze(-1), shift.unsqueeze(-1).unsqueeze(-1)))
        return mods

    def forward(self, source_eye: torch.Tensor, gaze_condition: torch.Tensor) -> List[Modulation]:
        return self.get_modulations(self.encode(source_eye), gaze_condition)

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
