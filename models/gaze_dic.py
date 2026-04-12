"""
FAGE — Eye-Only Gaze Generation UNet
从 Unet-Gaze/models/gaze_dic.py 精简而来，仅保留 eye-only 所需的通用 UNet 组件。

保留: GroupNorm, modulate, OverlapPatchEmbed, DiCFinalLayer,
      ConditionEmbedder, UNetBlock, Downsample, Upsample
新增: EyeOnlyGazeDiC, EyeOnlyWrapper
删除: GazeDiC(全脸版), ImagePromptFusion, EyeMultiScaleEncoder, UnifiedConditionEmbedder
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================
#  1. Helper Classes & DiT Components
# ==========================================

class GroupNorm(nn.Module):
    def __init__(self, num_channels, num_groups=32, min_channels_per_group=4, eps=1e-5, affine=True):
        super().__init__()
        self.num_groups = min(num_groups, num_channels // min_channels_per_group)
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_channels))
            self.bias = nn.Parameter(torch.zeros(num_channels))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        return F.group_norm(
            x,
            num_groups=self.num_groups,
            weight=self.weight.to(x.dtype) if self.affine else None,
            bias=self.bias.to(x.dtype) if self.affine else None,
            eps=self.eps
        )

def modulate(x, shift, scale):
    """AdaLN modulation: x * (1 + scale) + shift"""
    return x * (1 + scale.unsqueeze(2).unsqueeze(3)) + shift.unsqueeze(2).unsqueeze(3)


class OverlapPatchEmbed(nn.Module):
    """Overlap Patch Embedding — larger receptive field for initial feature extraction."""
    def __init__(self, patch_size=3, stride=1, in_chans=3, embed_dim=64):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size - stride) // 2)

    def forward(self, x):
        return self.proj(x)


class DiCFinalLayer(nn.Module):
    """AdaLN-Zero Final Layer — gaze condition modulates final pixel distribution."""
    def __init__(self, hidden_size, out_channels, condition_dim):
        super().__init__()
        self.norm_final = GroupNorm(hidden_size, affine=False)
        self.conv_final = nn.Conv2d(hidden_size, out_channels, kernel_size=3, padding=1)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(condition_dim, 2 * hidden_size, bias=True)
        )
        # Zero-Init for stable training
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.conv_final.weight, 0)
        nn.init.constant_(self.conv_final.bias, 0)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.conv_final(x)
        return x


# ==========================================
#  2. Condition Embedder
# ==========================================

class ConditionEmbedder(nn.Module):
    """Stage-specific gaze condition embedder (head + gaze → fused embedding)."""
    def __init__(self, gaze_dim, hidden_size):
        super().__init__()
        self.head_mlp = nn.Sequential(
            nn.Linear(gaze_dim, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.gaze_mlp = nn.Sequential(
            nn.Linear(gaze_dim, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, gaze_condition):
        # [B, dim] → gaze only
        if gaze_condition.dim() == 2:
            return self.gaze_mlp(gaze_condition)
        # [B, 2, dim] → head + gaze
        if gaze_condition.shape[1] == 2:
            head_emb = gaze_condition[:, 0, :]
            gaze_emb = gaze_condition[:, 1, :]
            head_feat = self.head_mlp(head_emb)
            gaze_feat = self.gaze_mlp(gaze_emb)
            combined = torch.cat([head_feat, gaze_feat], dim=-1)
            return self.fusion_mlp(combined)
        return self.gaze_mlp(gaze_condition)


# ==========================================
#  3. UNet Building Blocks
# ==========================================

class UNetBlock(nn.Module):
    """DiC UNet Block with complementary gating and optional channel mask.

    Condition injection: gate × conv(act(shift + norm(x) × (scale + 1))) + (1-gate) × residual
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        emb_channels=None,
        dropout=0,
        skip_scale=1,
        eps=1e-5,
        blockconfig=0,
        actfunc='gelu',
        affinef=3,
        norm_type='gnorm',
        num_groups=32,
        min_channels=4,
        **kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.blockconfig = blockconfig
        self.affinef = affinef
        self.skip_scale = skip_scale
        self.dropout = dropout

        self.noise_scale = nn.Parameter(torch.zeros(1, out_channels, 1, 1))

        # Norm + Conv 0
        self.norm0 = GroupNorm(in_channels, num_groups, min_channels, eps)
        self.conv0 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Affine (Gaze Control): gate, scale, shift
        affine_out_dim = out_channels * affinef
        self.affine = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, affine_out_dim, bias=True)
        )

        # Norm + Conv 1
        self.norm1 = GroupNorm(out_channels, eps=eps)
        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # Skip connection (1x1 conv for channel mismatch)
        self.skip = None
        if out_channels != in_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.act0 = nn.GELU() if actfunc == 'gelu' else nn.SiLU()
        self.act1 = nn.GELU() if actfunc == 'gelu' else nn.SiLU()

    def forward(self, x, emb, channel_mask=None, subject_scale=None, subject_shift=None):
        """
        Args:
            x: [B, C_in, H, W]
            emb: [B, emb_channels] stage-specific gaze embedding
            channel_mask: [B, C_out] 可选，向后兼容保留（不推荐使用）
            subject_scale: [B, C_out, 1, 1] 可选，来自 SubjectAdapter FiLM
            subject_shift: [B, C_out, 1, 1] 可选，来自 SubjectAdapter FiLM
        """
        orig = x

        # 1. Norm + Conv0
        x = self.norm0(x)
        x = self.conv0(self.act0(x))

        # 2. Training noise
        # if self.training:
        #     noise = torch.randn(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device)
        #     x = x + noise * self.noise_scale

        # 3. Affine params from gaze condition
        params = self.affine(emb)

        if self.affinef == 3:
            gate, scale, shift = params.chunk(3, dim=1)
            gate = gate.unsqueeze(2).unsqueeze(3)  # [B, C, 1, 1]
        else:
            scale, shift = params.chunk(2, dim=1)
            gate = 1

        # 4. Modulate + Conv1
        x = self.act1(modulate(self.norm1(x), shift, scale))
        x = self.conv1(F.dropout(x, p=self.dropout, training=self.training))

        # 5. 互补门控残差连接 (凸组合，输出有界)
        gate = torch.sigmoid(gate)
        skip_input = self.skip(orig) if self.skip is not None else orig
        x = gate * x + (1 - gate) * skip_input

        # # 6. 动态通道掩码 (向后兼容)
        # if channel_mask is not None:
        #     x = x * channel_mask.unsqueeze(2).unsqueeze(3)

        # 7. Subject FiLM modulation (主体外观保持，来自 SubjectAdapter)
        #    x = x * (1 + scale) + shift  — 零初始化时为恒等变换
        if subject_scale is not None:
            x = x * (1 + subject_scale) + subject_shift

        x = x * self.skip_scale
        return x


class Downsample(nn.Module):
    def __init__(self, n_feat, out_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, out_feat // 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2)
        )
    def forward(self, x): return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat, out_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, out_feat * 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2)
        )
    def forward(self, x): return self.body(x)


# ==========================================
#  4. Eye-Only GazeDiC Model
# ==========================================

class EyeOnlyGazeDiC(nn.Module):
    """
    输入: [B, 3, 64, 128] (左眼64x64 | 右眼64x64 横向拼接)
    输出: [B, 3, 64, 128]

    3-level UNet: enc0 → enc1 → latent → dec1 → dec0
    分辨率: 64x128 → 32x64 → 16x32 → 32x64 → 64x128
    """
    def __init__(
        self,
        in_channels=3,
        hidden_size=32,
        mult_channels=[1, 2, 4, 2, 1],
        depth=[2, 2, 4, 2, 2],
        gaze_dim=64,
        skip_stride=2,
        affinef=3,
        actfunc='gelu',
        norm_type='gnorm',
        num_groups=32,
        min_channels=4,
        dropout=0.1,
        skip_dropout=0,
        blockconfig=2,
        actinada=1,
        init_zero=0,
        **kwargs
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.mult_channels = mult_channels
        self.depth = depth
        self.gaze_dim = gaze_dim

        # Channel sizes per stage
        channels = [hidden_size * m for m in mult_channels]
        # enc0=channels[0], enc1=channels[1], latent=channels[2], dec0=channels[3], dec1=channels[4]

        # Patch embedding
        self.x_embedder = OverlapPatchEmbed(3, 1, in_channels, channels[0])

        # Stage-specific gaze embedders (3 levels: enc, latent, dec)
        self.gaze_embedder_ls = nn.ModuleList([
            ConditionEmbedder(gaze_dim, channels[0]),   # enc level
            ConditionEmbedder(gaze_dim, channels[1]),   # enc level 1
            ConditionEmbedder(gaze_dim, channels[2]),   # latent level
        ])

        # Encoder blocks
        self.enc_blocks = nn.ModuleList()
        for i in range(2):  # enc0, enc1
            stage = nn.ModuleList()
            for j in range(depth[i]):
                block_in = channels[i] if j > 0 else (channels[0] if i == 0 else channels[i])
                stage.append(UNetBlock(
                    in_channels=block_in,
                    out_channels=channels[i],
                    emb_channels=channels[i],
                    dropout=dropout,
                    skip_scale=1,
                    blockconfig=blockconfig,
                    actfunc=actfunc,
                    affinef=affinef,
                    norm_type=norm_type,
                    num_groups=num_groups,
                    min_channels=min_channels,
                ))
            self.enc_blocks.append(stage)

        # Downsample layers
        self.downs = nn.ModuleList([
            Downsample(channels[0], channels[1]),
            Downsample(channels[1], channels[2]),
        ])

        # Latent blocks
        self.lat_blocks = nn.ModuleList()
        lat_stage = nn.ModuleList()
        for j in range(depth[2]):
            lat_stage.append(UNetBlock(
                in_channels=channels[2],
                out_channels=channels[2],
                emb_channels=channels[2],
                dropout=dropout,
                skip_scale=1,
                blockconfig=blockconfig,
                actfunc=actfunc,
                affinef=affinef,
                norm_type=norm_type,
                num_groups=num_groups,
                min_channels=min_channels,
            ))
        self.lat_blocks.append(lat_stage)

        # Upsample layers
        self.ups = nn.ModuleList([
            Upsample(channels[2], channels[3]),
            Upsample(channels[3], channels[4]),
        ])

        # Decoder blocks (with skip connections from encoder)
        self.dec_blocks = nn.ModuleList()
        for i in range(2):  # dec0, dec1
            dec_idx = i + 3  # channels[3], channels[4]
            enc_idx = 1 - i  # skip from enc1, enc0
            stage = nn.ModuleList()
            for j in range(depth[dec_idx]):
                # 第一个 block 接收 skip connection (concat 后通道翻倍)
                if j == 0:
                    block_in = channels[dec_idx] + channels[enc_idx]
                else:
                    block_in = channels[dec_idx]
                stage.append(UNetBlock(
                    in_channels=block_in,
                    out_channels=channels[dec_idx],
                    emb_channels=channels[min(enc_idx, 2)],  # 使用对应层的 embedder
                    dropout=dropout,
                    skip_scale=1,
                    blockconfig=blockconfig,
                    actfunc=actfunc,
                    affinef=affinef,
                    norm_type=norm_type,
                    num_groups=num_groups,
                    min_channels=min_channels,
                ))
            self.dec_blocks.append(stage)

        # Final layer
        self.final_layer = DiCFinalLayer(channels[4], in_channels, channels[0])

    def forward(self, eye_input, gaze_cond, masks=None, subject_mods=None):
        """
        Args:
            eye_input:     [B, C_in, H, W] 源眼部 crop
            gaze_cond:     [B, 2, gaze_dim] target head+gaze embedding
            masks:         list of [B, C_l] 可选，向后兼容（不推荐）
            subject_mods:  list of (scale, shift) tuples，来自 SubjectAdapter
                           每个元素 shape [B, C_block, 1, 1]，共 12 个
        Returns:
            [B, C_in, H, W] 生成的眼部图像
        """
        block_idx = 0
        mask_idx = 0

        def _sub(idx):
            """取第 idx 个 block 的 subject FiLM 参数"""
            if subject_mods is not None:
                s, sh = subject_mods[idx]
                return s, sh
            return None, None

        # Patch embed
        x = self.x_embedder(eye_input)  # [B, C0, H, W]

        # Stage-specific gaze embeddings
        emb_enc0 = self.gaze_embedder_ls[0](gaze_cond)
        emb_enc1 = self.gaze_embedder_ls[1](gaze_cond)
        emb_lat  = self.gaze_embedder_ls[2](gaze_cond)

        # Encoder
        skip_features = []

        # Enc stage 0
        for block in self.enc_blocks[0]:
            m = masks[mask_idx] if masks else None
            ss, sh = _sub(block_idx)
            x = block(x, emb_enc0, channel_mask=m, subject_scale=ss, subject_shift=sh)
            mask_idx += 1
            block_idx += 1
        skip_features.append(x)
        x = self.downs[0](x)

        # Enc stage 1
        for block in self.enc_blocks[1]:
            m = masks[mask_idx] if masks else None
            ss, sh = _sub(block_idx)
            x = block(x, emb_enc1, channel_mask=m, subject_scale=ss, subject_shift=sh)
            mask_idx += 1
            block_idx += 1
        skip_features.append(x)
        x = self.downs[1](x)

        # Latent
        for block in self.lat_blocks[0]:
            m = masks[mask_idx] if masks else None
            ss, sh = _sub(block_idx)
            x = block(x, emb_lat, channel_mask=m, subject_scale=ss, subject_shift=sh)
            mask_idx += 1
            block_idx += 1

        # Decoder
        emb_dec = [emb_enc1, emb_enc0]  # 对称: dec0 用 enc1 的 embedder, dec1 用 enc0

        for i in range(2):
            x = self.ups[i](x)
            # Skip connection (concat)
            skip = skip_features[1 - i]
            x = torch.cat([x, skip], dim=1)
            for block in self.dec_blocks[i]:
                m = masks[mask_idx] if masks else None
                ss, sh = _sub(block_idx)
                x = block(x, emb_dec[i], channel_mask=m, subject_scale=ss, subject_shift=sh)
                mask_idx += 1
                block_idx += 1

        # Final layer
        output = self.final_layer(x, emb_enc0)
        return output


class EyeOnlyWrapper(nn.Module):
    """Eye-only 生成的完整封装

    训练时: source_eye_crops + target_gaze → generated_eyes (对比 target_eye_crops)
    推理时: source_image + target_gaze → crop眼 → 生成 → paste_eyes 回 source 脸

    Architecture:
        GazeControlNet (eye_unet):  处理注视控制，Phase 2 后冻结
        SubjectAdapter:             处理主体外观保持，新用户只更新这部分 (~350K)
    """
    def __init__(self, unet_config, subject_adapter_config=None):
        super().__init__()
        self.eye_unet = EyeOnlyGazeDiC(
            in_channels=unet_config.get('in_channels', 3),
            hidden_size=unet_config.get('hidden_size', 32),
            mult_channels=unet_config.get('mult_channels', [1, 2, 4, 2, 1]),
            depth=unet_config.get('depth', [2, 2, 4, 2, 2]),
            gaze_dim=unet_config.get('gaze_dim', 64),
            skip_stride=unet_config.get('skip_stride', 2),
            affinef=unet_config.get('affinef', 3),
            actfunc=unet_config.get('actfunc', 'gelu'),
            dropout=unet_config.get('dropout', 0.1),
            blockconfig=unet_config.get('blockconfig', 2),
        )

        # SubjectAdapter: 主体外观保持模块（Phase 2 个性化时只训练此模块）
        self.subject_adapter = None
        if subject_adapter_config is not None:
            from models.subject_adapter import SubjectAdapter
            self.subject_adapter = SubjectAdapter(
                in_channels=subject_adapter_config.get('in_channels', 6),
                subject_dim=subject_adapter_config.get('subject_dim', 128),
                block_channels=self._get_block_channels(),
            )

    def _get_block_channels(self):
        """从 UNet 结构自动推导每个 block 的输出通道数"""
        channels = []
        for stage in self.eye_unet.enc_blocks:
            for block in stage:
                channels.append(block.out_channels)
        for stage in self.eye_unet.lat_blocks:
            for block in stage:
                channels.append(block.out_channels)
        for stage in self.eye_unet.dec_blocks:
            for block in stage:
                channels.append(block.out_channels)
        return channels

    def forward(self, source_eye_crops, encoder_hidden_states):
        """
        Args:
            source_eye_crops:       [B, C_in, H, W] 源眼部 crop
            encoder_hidden_states:  [B, 2, gaze_dim] 给 UNet 的 gaze 条件
        Returns:
            generated_eyes: [B, C_in, H, W]
        """
        subject_mods = None
        if self.subject_adapter is not None:
            subject_mods = self.subject_adapter(source_eye_crops)

        return self.eye_unet(source_eye_crops, encoder_hidden_states,
                             subject_mods=subject_mods)

    def paste_eyes(self, generated_eyes, source_image, eye_bbox, blend_margin=4):
        """推理时将生成的眼睛贴回原图

        Args:
            generated_eyes: [B, 6, 64, 64] (左眼前3通道 | 右眼后3通道)
            source_image: [B, 3, 256, 256] 原始 source 脸
            eye_bbox: [B, 8] 归一化坐标 [lx1,ly1,lx2,ly2, rx1,ry1,rx2,ry2]
            blend_margin: 边缘渐变像素数
        Returns:
            result: [B, 3, 256, 256] 替换眼睛后的完整图像
        """
        B, _, H, W = source_image.shape
        result = source_image.clone()
        left_eye  = generated_eyes[:, :3, :, :]
        right_eye = generated_eyes[:, 3:, :, :]

        for b in range(B):
            for eye_crop, bbox_slice in [(left_eye, slice(0, 4)), (right_eye, slice(4, 8))]:
                x1, y1, x2, y2 = (eye_bbox[b, bbox_slice] * H).int().tolist()
                x1, x2 = max(0, x1), min(W, x2)
                y1, y2 = max(0, y1), min(H, y2)
                eh, ew = y2 - y1, x2 - x1
                if eh > 0 and ew > 0:
                    resized = F.interpolate(
                        eye_crop[b:b+1], size=(eh, ew),
                        mode='bilinear', align_corners=False
                    )[0]
                    mask = self._create_blend_mask(eh, ew, blend_margin, source_image.device)
                    result[b, :, y1:y2, x1:x2] = mask * resized + (1 - mask) * result[b, :, y1:y2, x1:x2]
        return result

    @staticmethod
    def _create_blend_mask(h, w, margin, device):
        """创建边缘渐变混合遮罩 [1, H, W]"""
        mask = torch.ones(1, h, w, device=device)
        if margin <= 0:
            return mask
        for i in range(min(margin, h // 2)):
            alpha = (i + 1) / (margin + 1)
            mask[:, i, :] *= alpha
            mask[:, h - 1 - i, :] *= alpha
        for j in range(min(margin, w // 2)):
            alpha = (j + 1) / (margin + 1)
            mask[:, :, j] *= alpha
            mask[:, :, w - 1 - j] *= alpha
        return mask
