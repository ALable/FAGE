import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
from loss.vgg_face import ImagePyramide

# PatchGan from sted 
class PatchGAN(nn.Module):
    def __init__(self, input_nc, ndf=64):
        super(PatchGAN, self).__init__()
        use_bias = False
        kw = 4
        padw = 1
        self.conv1 = nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw)

        self.conv2 = nn.Conv2d(ndf, ndf * 2, kernel_size=kw, stride=2, padding=padw, bias=use_bias)
        self.norm1 = nn.BatchNorm2d(ndf * 2)
        self.act = nn.LeakyReLU(0.2, True)

        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, kernel_size=kw, stride=2, padding=padw, bias=use_bias)
        self.norm2 = nn.BatchNorm2d(ndf * 4)

        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, kernel_size=kw, stride=1, padding=padw, bias=use_bias)
        self.norm3 = nn.BatchNorm2d(ndf * 8)

        self.conv5 = nn.Conv2d(ndf * 8, 1, kernel_size=kw, stride=1, padding=padw)  # output 1 channel prediction map

    def forward(self, input):
        """Standard forward."""

        input = self.conv1(input)
        input = self.act(input)

        input = self.conv2(input)
        input = self.norm1(input)
        input = self.act(input)

        input = self.conv3(input)
        input = self.norm2(input)
        input = self.act(input)

        input = self.conv4(input)
        input = self.norm3(input)
        input = self.act(input)

        input = self.conv5(input)

        return input

# class PatchGAN(nn.Module):
#     """
#     SOTA PatchGAN for Inpainting
#     - Removed BatchNorm (Prevents color shifting/washing out)
#     - Added Spectral Normalization (Stabilizes training)
#     - Fixed Kernel Size (4x4 is standard for geometric alignment)
#     - Supports variable input sizes (256×256 for full face, 64×64 for eye patches)
#     """
#     def __init__(self, input_nc=3, ndf=64):
#         super(PatchGAN, self).__init__()
        
#         # 使用 4x4 卷积核，Stride 2，Padding 1 (经典的偶数下采样配置)
#         kw = 4
#         padw = 1
#         sequence = [
#             # Layer 1: Input -> 64
#             # 判别器第一层通常不加 Norm，也不加 SN (这是经验法则)
#             nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
#             nn.LeakyReLU(0.2, True),

#             # Layer 2: 64 -> 128
#             # 使用 Spectral Norm 代替 Batch Norm
#             spectral_norm(nn.Conv2d(ndf, ndf * 2, kernel_size=kw, stride=2, padding=padw)),
#             nn.LeakyReLU(0.2, True),

#             # Layer 3: 128 -> 256
#             spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, kernel_size=kw, stride=2, padding=padw)),
#             nn.LeakyReLU(0.2, True),

#             # Layer 4: 256 -> 512 (Stride=1, 稍微缩小感受野，聚焦局部)
#             spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, kernel_size=kw, stride=1, padding=padw)),
#             nn.LeakyReLU(0.2, True),

#             # Layer 5: Output (1 channel prediction map)
#             # 最后一层不加 Norm，不加 Act
#             nn.Conv2d(ndf * 8, 1, kernel_size=kw, stride=1, padding=padw)
#         ]

#         self.model = nn.Sequential(*sequence)

#     def forward(self, input):
#         return self.model(input)

class DownBlock2d(nn.Module):
    """
    Simple block for processing video (encoder).
    """

    def __init__(self, in_features, out_features, norm=False, kernel_size=4, pool=False, sn=False):
        super(DownBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size)

        if sn:
            self.conv = nn.utils.spectral_norm(self.conv)
        if norm:
            self.norm = nn.InstanceNorm2d(out_features, affine=True)
        else:
            self.norm = None
        self.pool = pool

    def forward(self, x):
        out = x
        out = self.conv(out)
        # Safety check: disable InstanceNorm if feature map too small
        if self.norm and out.size(2) > 1 and out.size(3) > 1:
            out = self.norm(out)
        out = F.leaky_relu(out, 0.2)
        if self.pool:
            out = F.avg_pool2d(out, (2, 2))
        return out


class Discriminator(nn.Module):
    """
    Discriminator similar to Pix2Pix
    """

    def __init__(self, num_channels=3, block_expansion=64, num_blocks=4, max_features=512,
                 sn=False, **kwargs):
        super(Discriminator, self).__init__()
        down_blocks = []
        for i in range(num_blocks):
            down_blocks.append(
                DownBlock2d(num_channels if i == 0 else min(max_features, block_expansion * (2 ** i)),
                            min(max_features, block_expansion * (2 ** (i + 1))),
                            norm=(i != 0), kernel_size=4, pool=(i != num_blocks - 1), sn=sn))

        self.down_blocks = nn.ModuleList(down_blocks)
        self.conv = nn.Conv2d(self.down_blocks[-1].conv.out_channels, out_channels=1, kernel_size=1)
        if sn:
            self.conv = nn.utils.spectral_norm(self.conv)

    def forward(self, x):
        feature_maps = []
        out = x

        for down_block in self.down_blocks:
            feature_maps.append(down_block(out))
            out = feature_maps[-1]
        prediction_map = self.conv(out)

        return feature_maps, prediction_map


class MultiScaleDiscriminator(nn.Module):
    """
    Multi-scale (scale) discriminator
    """

    def __init__(self, scales=(), **kwargs):
        super(MultiScaleDiscriminator, self).__init__()
        self.scales = scales
        discs = {}
        for scale in scales:
            discs[str(scale).replace('.', '-')] = Discriminator(**kwargs)
        self.discs = nn.ModuleDict(discs)

    def forward(self, x):
        """Accept a plain tensor [B, C, H, W], resize per scale internally,
        return concatenated flattened predictions [B, N_total] for API compatibility."""
        predictions = []
        for scale_key, disc in self.discs.items():
            scale = float(scale_key.replace('-', '.'))
            if scale != 1.0:
                x_scaled = F.interpolate(x, scale_factor=scale, mode='bilinear',
                                         align_corners=False, recompute_scale_factor=True)
            else:
                x_scaled = x
            _, prediction_map = disc(x_scaled)          # [B, 1, H', W']
            predictions.append(prediction_map.flatten(1))  # [B, H'*W']
        return torch.cat(predictions, dim=1)               # [B, N_total]


