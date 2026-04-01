"""
GazeDiC 网络深度分析工具

分析方法:
1. 通道相关性分析 (Channel Correlation) - 识别冗余通道
2. 特征图可视化 (Feature Map Visualization) - 理解网络学习内容
3. 激活统计分析 (Activation Statistics) - 检测死神经元
4. 梯度流分析 (Gradient Flow) - 识别梯度消失/爆炸

使用方法:
python analyze_network.py --checkpoint path/to/checkpoint.pth --output_dir ./analysis_results
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from tqdm import tqdm
import os

# 可选导入
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("⚠️  seaborn not installed, using matplotlib only")

from models.unet_dic import GazeDiCWrapper


class NetworkAnalyzer:
    """网络分析器 - 提供多种分析工具"""

    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.activations = {}
        self.gradients = {}

        # 注册 hooks
        self.hooks = []

    def register_hooks(self, layer_names=None):
        """注册前向和反向钩子来捕获激活和梯度"""

        def forward_hook(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    output = output[0]
                self.activations[name] = output.detach().cpu()
            return hook

        def backward_hook(name):
            def hook(module, grad_input, grad_output):
                if isinstance(grad_output[0], torch.Tensor):
                    self.gradients[name] = grad_output[0].detach().cpu()
            return hook

        # 自动注册所有 UNetBlock 和 Down/Upsample
        for name, module in self.model.named_modules():
            if 'enc_blocks' in name or 'dec_blocks' in name or 'lat_blocks' in name:
                if isinstance(module, nn.Conv2d):
                    handle_fwd = module.register_forward_hook(forward_hook(name))
                    handle_bwd = module.register_full_backward_hook(backward_hook(name))
                    self.hooks.append(handle_fwd)
                    self.hooks.append(handle_bwd)

        print(f"✅ 注册了 {len(self.hooks)//2} 个层的 hooks")

    def remove_hooks(self):
        """移除所有钩子"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def analyze_channel_correlation(self, feature_map, layer_name):
        """
        通道相关性分析

        目的: 识别冗余通道
        - 高相关性 (>0.9) → 通道冗余，可以剪枝
        - 低相关性 (<0.3) → 通道独立，信息丰富

        Returns:
            correlation_matrix: [C, C] 通道间相关性矩阵
            redundant_pairs: 高度相关的通道对 (可剪枝)
        """
        # feature_map: [B, C, H, W]
        B, C, H, W = feature_map.shape

        # 展平空间维度: [B, C, H*W] -> [C, B*H*W]
        features = feature_map.view(B, C, -1).permute(1, 0, 2).reshape(C, -1)

        # 计算相关性矩阵
        features_normalized = (features - features.mean(dim=1, keepdim=True)) / (features.std(dim=1, keepdim=True) + 1e-8)
        correlation_matrix = torch.mm(features_normalized, features_normalized.t()) / features.shape[1]

        # 识别高度相关的通道对 (排除对角线)
        correlation_matrix_np = correlation_matrix.numpy()
        np.fill_diagonal(correlation_matrix_np, 0)  # 忽略自相关

        redundant_pairs = []
        threshold = 0.9
        for i in range(C):
            for j in range(i+1, C):
                if abs(correlation_matrix_np[i, j]) > threshold:
                    redundant_pairs.append((i, j, correlation_matrix_np[i, j]))

        return correlation_matrix_np, redundant_pairs

    def analyze_activation_statistics(self, feature_map, layer_name):
        """
        激活统计分析

        目的: 检测死神经元和激活异常
        - Dead neurons: 激活值始终为0或极小
        - Saturated neurons: 激活值始终饱和 (接近max)

        Returns:
            stats: {mean, std, min, max, sparsity, dead_ratio}
        """
        # feature_map: [B, C, H, W]
        B, C, H, W = feature_map.shape

        # 按通道统计
        channel_means = feature_map.mean(dim=[0, 2, 3])  # [C]
        channel_stds = feature_map.std(dim=[0, 2, 3])    # [C]
        channel_mins = feature_map.amin(dim=[0, 2, 3])   # [C]
        channel_maxs = feature_map.amax(dim=[0, 2, 3])   # [C]

        # 稀疏性: 激活值接近0的比例
        sparsity = (feature_map.abs() < 0.01).float().mean().item()

        # 死神经元: 均值和标准差都极小
        dead_threshold = 0.01
        dead_neurons = ((channel_means.abs() < dead_threshold) & (channel_stds < dead_threshold)).sum().item()
        dead_ratio = dead_neurons / C

        stats = {
            'layer_name': layer_name,
            'num_channels': C,
            'mean': channel_means.numpy(),
            'std': channel_stds.numpy(),
            'min': channel_mins.numpy(),
            'max': channel_maxs.numpy(),
            'sparsity': sparsity,
            'dead_neurons': dead_neurons,
            'dead_ratio': dead_ratio,
        }

        return stats

    def visualize_feature_maps(self, feature_map, layer_name, save_path, num_channels=16):
        """
        特征图可视化

        显示前 num_channels 个通道的激活模式
        """
        # feature_map: [B, C, H, W]
        B, C, H, W = feature_map.shape

        # 取第一个样本的前 num_channels 个通道
        num_channels = min(num_channels, C)
        feature_map_vis = feature_map[0, :num_channels].numpy()

        # 创建网格可视化
        grid_size = int(np.ceil(np.sqrt(num_channels)))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
        axes = axes.flatten()

        for i in range(num_channels):
            ax = axes[i]
            im = ax.imshow(feature_map_vis[i], cmap='viridis')
            ax.set_title(f'Ch {i}', fontsize=8)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)

        # 隐藏多余的子图
        for i in range(num_channels, len(axes)):
            axes[i].axis('off')

        plt.suptitle(f'Feature Maps: {layer_name}', fontsize=16)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def visualize_channel_correlation(self, correlation_matrix, layer_name, save_path):
        """可视化通道相关性矩阵"""
        plt.figure(figsize=(12, 10))

        if HAS_SEABORN:
            sns.heatmap(correlation_matrix, cmap='RdBu_r', center=0,
                        vmin=-1, vmax=1, square=True, cbar_kws={'label': 'Correlation'})
        else:
            # Fallback: 使用 matplotlib的 imshow
            im = plt.imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
            plt.colorbar(im, label='Correlation')

        plt.title(f'Channel Correlation Matrix: {layer_name}')
        plt.xlabel('Channel Index')
        plt.ylabel('Channel Index')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def analyze_gradient_flow(self):
        """
        梯度流分析

        目的: 检测梯度消失/爆炸问题
        - 梯度范数 < 1e-5: 梯度消失
        - 梯度范数 > 100: 梯度爆炸
        """
        grad_stats = {}

        for name, grad in self.gradients.items():
            if grad is not None:
                grad_norm = grad.norm().item()
                grad_mean = grad.mean().item()
                grad_std = grad.std().item()

                grad_stats[name] = {
                    'norm': grad_norm,
                    'mean': grad_mean,
                    'std': grad_std,
                    'vanishing': grad_norm < 1e-5,
                    'exploding': grad_norm > 100,
                }

        return grad_stats

    def visualize_gradient_flow(self, grad_stats, save_path):
        """可视化梯度流"""
        layer_names = list(grad_stats.keys())
        grad_norms = [grad_stats[name]['norm'] for name in layer_names]

        plt.figure(figsize=(15, 6))
        plt.bar(range(len(layer_names)), grad_norms)
        plt.yscale('log')
        plt.xlabel('Layer Index')
        plt.ylabel('Gradient Norm (log scale)')
        plt.title('Gradient Flow Across Layers')
        plt.axhline(y=1e-5, color='r', linestyle='--', label='Vanishing threshold')
        plt.axhline(y=100, color='orange', linestyle='--', label='Exploding threshold')
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def run_analysis(checkpoint_path, output_dir, config_path):
    """运行完整的网络分析"""

    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载模型
    print("🔧 加载模型...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 初始化模型
    model = GazeDiCWrapper(unet_config=config_path, device=device)

    # 加载权重
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # 尝试加载，允许部分不匹配（新旧版本兼容）
        try:
            model.model.load_state_dict(state_dict, strict=True)
            print(f"✅ 加载权重 (strict): {checkpoint_path}")
        except RuntimeError as e:
            print(f"⚠️  Strict加载失败，尝试非严格模式...")
            # 移除 'model.' 前缀（如果存在）
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('model.'):
                    new_state_dict[k[6:]] = v
                else:
                    new_state_dict[k] = v

            missing_keys, unexpected_keys = model.model.load_state_dict(new_state_dict, strict=False)

            print(f"✅ 加载权重 (非严格): {checkpoint_path}")
            if missing_keys:
                print(f"⚠️  缺少的键 ({len(missing_keys)}): {missing_keys[:5]}...")
            if unexpected_keys:
                print(f"⚠️  意外的键 ({len(unexpected_keys)}): {unexpected_keys[:5]}...")
    else:
        print("⚠️  未加载权重，使用随机初始化的模型")

    model.eval()

    # 初始化分析器
    analyzer = NetworkAnalyzer(model.model, device)
    analyzer.register_hooks()

    # 创建测试输入
    print("🎲 生成测试输入...")
    batch_size = 4
    test_input = torch.randn(batch_size, 3, 256, 256).to(device)
    test_gaze = torch.randn(batch_size, 2, 64).to(device)
    # Eye texture region for style injection (if model uses it)
    test_eye_texture = torch.randn(batch_size, 3, 256, 256).to(device)

    # 前向传播
    print("⏩ 前向传播...")
    with torch.enable_grad():
        test_input.requires_grad = True
        # GazeDiC.forward(sample, encoder_hidden_states, eye_texture_region, return_dict)
        output = model.model(test_input, test_gaze, eye_texture_region=test_eye_texture, return_dict=False)

        # 反向传播 (用于梯度分析)
        loss = output.sum()
        loss.backward()

    print(f"✅ 捕获了 {len(analyzer.activations)} 个激活")
    print(f"✅ 捕获了 {len(analyzer.gradients)} 个梯度")

    # ========== 分析1: 通道相关性 ==========
    print("\n" + "="*60)
    print("📊 分析1: 通道相关性分析")
    print("="*60)

    correlation_results = {}
    for layer_name, activation in tqdm(list(analyzer.activations.items())[:10], desc="通道相关性"):
        if activation.dim() == 4 and activation.shape[1] > 1:  # [B, C, H, W]
            corr_matrix, redundant_pairs = analyzer.analyze_channel_correlation(activation, layer_name)

            correlation_results[layer_name] = {
                'correlation_matrix': corr_matrix,
                'redundant_pairs': redundant_pairs,
                'num_redundant': len(redundant_pairs),
            }

            # 可视化
            save_path = output_dir / f'correlation_{layer_name.replace(".", "_")}.png'
            analyzer.visualize_channel_correlation(corr_matrix, layer_name, save_path)

            if redundant_pairs:
                print(f"  {layer_name}: 发现 {len(redundant_pairs)} 对冗余通道")

    # ========== 分析2: 激活统计 ==========
    print("\n" + "="*60)
    print("📊 分析2: 激活统计分析")
    print("="*60)

    activation_stats = []
    for layer_name, activation in tqdm(analyzer.activations.items(), desc="激活统计"):
        if activation.dim() == 4:
            stats = analyzer.analyze_activation_statistics(activation, layer_name)
            activation_stats.append(stats)

            if stats['dead_ratio'] > 0.1:
                print(f"  ⚠️  {layer_name}: {stats['dead_ratio']*100:.1f}% 死神经元")

    # ========== 分析3: 特征图可视化 ==========
    print("\n" + "="*60)
    print("📊 分析3: 特征图可视化")
    print("="*60)

    # 可视化所有encoder, decoder, latent层
    for layer_name, activation in analyzer.activations.items():
        if activation.dim() == 4 and ('enc_blocks' in layer_name or 'dec_blocks' in layer_name or 'lat_blocks' in layer_name):
            save_path = output_dir / f'feature_maps_{layer_name.replace(".", "_")}.png'
            analyzer.visualize_feature_maps(activation, layer_name, save_path, num_channels=16)
            print(f"  ✅ 保存: {save_path}")

    # ========== 分析4: 梯度流 ==========
    print("\n" + "="*60)
    print("📊 分析4: 梯度流分析")
    print("="*60)

    grad_stats = analyzer.analyze_gradient_flow()

    vanishing_layers = [name for name, stats in grad_stats.items() if stats['vanishing']]
    exploding_layers = [name for name, stats in grad_stats.items() if stats['exploding']]

    if vanishing_layers:
        print(f"  ⚠️  梯度消失层 ({len(vanishing_layers)}): {vanishing_layers[:5]}")
    if exploding_layers:
        print(f"  ⚠️  梯度爆炸层 ({len(exploding_layers)}): {exploding_layers[:10]}")
        if len(exploding_layers) > 10:
            print(f"      ... 还有 {len(exploding_layers) - 10} 个层")

    # 可视化梯度流
    save_path = output_dir / 'gradient_flow.png'
    analyzer.visualize_gradient_flow(grad_stats, save_path)
    print(f"  ✅ 保存: {save_path}")

    # ========== 生成报告 ==========
    print("\n" + "="*60)
    print("📝 生成分析报告")
    print("="*60)

    report_path = output_dir / 'analysis_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("GazeDiC 网络分析报告\n")
        f.write("="*60 + "\n\n")

        f.write("## 1. 通道相关性分析\n\n")
        total_redundant = sum(r['num_redundant'] for r in correlation_results.values())
        f.write(f"总冗余通道对: {total_redundant}\n")
        for layer_name, result in correlation_results.items():
            if result['num_redundant'] > 0:
                f.write(f"  - {layer_name}: {result['num_redundant']} 对\n")

        f.write("\n## 2. 激活统计分析\n\n")
        total_dead = sum(s['dead_neurons'] for s in activation_stats)
        f.write(f"总死神经元: {total_dead}\n")
        for stats in activation_stats:
            if stats['dead_ratio'] > 0.1:
                f.write(f"  - {stats['layer_name']}: {stats['dead_neurons']}/{stats['num_channels']} ({stats['dead_ratio']*100:.1f}%)\n")

        f.write("\n## 3. 梯度流分析\n\n")
        f.write(f"梯度消失层: {len(vanishing_layers)}\n")
        if vanishing_layers:
            f.write("\n详细列表（梯度消失层）:\n")
            for layer in vanishing_layers:
                norm = grad_stats[layer]['norm']
                f.write(f"  - {layer}: grad_norm={norm:.6f}\n")

        f.write(f"\n梯度爆炸层: {len(exploding_layers)}\n")
        if exploding_layers:
            f.write("\n详细列表（梯度爆炸层）:\n")
            # 按梯度范数降序排列
            sorted_exploding = sorted(exploding_layers,
                                     key=lambda x: grad_stats[x]['norm'],
                                     reverse=True)
            for layer in sorted_exploding:
                norm = grad_stats[layer]['norm']
                f.write(f"  - {layer}: grad_norm={norm:.2f}\n")

        f.write("\n## 4. 优化建议\n\n")
        if total_redundant > 10:
            f.write("- ⚠️  发现大量冗余通道，建议进行通道剪枝\n")
        if total_dead > 50:
            f.write("- ⚠️  发现大量死神经元，建议检查激活函数或初始化\n")
        if vanishing_layers:
            f.write("- ⚠️  存在梯度消失，建议使用ResNet连接或调整学习率\n")
        if exploding_layers:
            f.write("- ⚠️  存在梯度爆炸，建议使用梯度裁剪或降低学习率\n")

    print(f"✅ 报告保存: {report_path}")

    # 清理
    analyzer.remove_hooks()

    print("\n" + "="*60)
    print("🎉 分析完成！")
    print(f"📁 结果保存在: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GazeDiC Network Analysis')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str,
                        default='configs/training/dic_stage1.yaml',
                        help='Path to config file')
    parser.add_argument('--output_dir', type=str, default='./analysis_results',
                        help='Output directory for analysis results')

    args = parser.parse_args()

    run_analysis(args.checkpoint, args.output_dir, args.config)
