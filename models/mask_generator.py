"""
FAGE — Dynamic Mask Generator for Personalization (Phase 2)

根据输入眼部特征 + gaze 条件，为 UNet 每个 block 动态生成通道掩码。
使用 Gumbel-Sigmoid 实现可微分的随机二值采样。

Phase 1: 不使用 (mask_generator=None)
Phase 2: 冻结 UNet, 只训练此模块 (~300K params)
"""
import torch
import torch.nn as nn


class DynamicMaskGenerator(nn.Module):
    """输入相关的动态通道掩码生成器

    架构: feat_encoder(Conv→GAP) + trunk(MLP) + per-block mask_heads(Linear→Gumbel-Sigmoid)
    训练时: Gumbel 噪声 + 温度退火 → 随机软掩码, 鼓励探索
    推理时: 无噪声, hard threshold → 二值掩码 {0, 1}
    """
    def __init__(self, in_channels=3, gaze_dim=64,
                 block_channels=None, hidden_dim=128,
                 tau_init=1.0, tau_min=0.1):
        """
        Args:
            in_channels: 输入图像通道数 (3)
            gaze_dim: gaze condition 维度
            block_channels: list[int], 每个 UNetBlock 的输出通道数
            hidden_dim: trunk MLP 隐层维度
            tau_init: Gumbel-Sigmoid 初始温度 (高温→软掩码)
            tau_min: 最低温度 (低温→接近二值)
        """
        super().__init__()
        if block_channels is None:
            # EyeOnlyGazeDiC default: depth=[2,2,4,2,2], mult=[1,2,4,2,1], hidden=32
            block_channels = [32]*2 + [64]*2 + [128]*4 + [64]*2 + [32]*2

        self.tau = tau_init
        self.tau_min = tau_min

        # 轻量特征提取器 (从输入 eye crop 提取全局表示)
        self.feat_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, 2, 1),   # [B,32,32,64]
            nn.GELU(),
            nn.Conv2d(32, 64, 4, 2, 1),             # [B,64,16,32]
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),                 # [B,64,1,1]
            nn.Flatten(),                             # [B,64]
        )

        feat_dim = 64 + gaze_dim

        # 共享 trunk
        self.trunk = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        # 每个 block 独立的 mask head (输出 logits)
        self.mask_heads = nn.ModuleList([
            nn.Linear(hidden_dim, ch) for ch in block_channels
        ])

        # 初始化 bias=2.0 → gumbel_sigmoid(2.0, tau=1)≈0.88
        # 初始状态几乎不改变预训练 UNet 的行为
        for head in self.mask_heads:
            nn.init.zeros_(head.weight)
            nn.init.constant_(head.bias, 2.0)

    def gumbel_sigmoid(self, logits, hard=False):
        """Gumbel-Sigmoid: 可微分的随机二值采样

        Args:
            logits: [B, C] 未归一化分数
            hard: 是否输出硬二值 (straight-through estimator)
        Returns:
            mask: [B, C] ∈ (0,1), hard 时为 {0,1}
        """
        if self.training:
            # 采样 Gumbel(0,1) 噪声
            gumbels = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
            y_soft = torch.sigmoid((logits + gumbels) / self.tau)
        else:
            # eval 无噪声
            y_soft = torch.sigmoid(logits / self.tau)

        if hard:
            y_hard = (y_soft > 0.5).float()
            return y_hard - y_soft.detach() + y_soft
        return y_soft

    def set_tau(self, tau):
        """外部调用设置温度 (退火调度)"""
        self.tau = max(tau, self.tau_min)

    def forward(self, eye_input, gaze_cond_flat):
        """
        Args:
            eye_input: [B, 3, 64, 128] 源眼部 crop
            gaze_cond_flat: [B, gaze_dim] target gaze (展平)
        Returns:
            masks: list of [B, C_l] tensors
        """
        feat = self.feat_encoder(eye_input)
        combined = torch.cat([feat, gaze_cond_flat], dim=1)
        h = self.trunk(combined)

        # eval 自动 hard mask
        use_hard = not self.training

        masks = []
        for head in self.mask_heads:
            logits = head(h)
            mask = self.gumbel_sigmoid(logits, hard=use_hard)
            masks.append(mask)
        return masks
