"""
FAGE — Mask Generator 可视化与分析

可视化 DynamicMaskGenerator 为不同输入/用户生成的通道掩码分布。
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from models.gaze_dic import EyeOnlyWrapper
from models.mask_generator import DynamicMaskGenerator


def visualize_user_masks(model, dataloader, save_path, num_samples=50, device='cuda'):
    """可视化 mask 分布

    Args:
        model: EyeOnlyWrapper (with mask_generator)
        dataloader: 数据加载器
        save_path: 保存路径
        num_samples: 采样数
        device: 设备
    """
    os.makedirs(save_path, exist_ok=True)
    model.eval()
    model.to(device)

    all_masks = []  # list of [num_blocks, C_l]
    all_subjects = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i * batch['source_eye_crops'].shape[0] >= num_samples:
                break

            src = batch['source_eye_crops'].to(device)
            tgt_gaze = batch['target_gaze'].to(device)

            # 简单展平 gaze 作为条件 (实际训练中用 gaze_mlp)
            gaze_flat = tgt_gaze  # [B, 2] — 真实场景用 gaze_mlp 输出

            if model.mask_generator is not None:
                masks = model.mask_generator(src, gaze_flat)
                for b in range(src.shape[0]):
                    sample_masks = [m[b].cpu().numpy() for m in masks]
                    all_masks.append(sample_masks)
                    all_subjects.append(batch['subject_idx'][b].item())

    if not all_masks:
        print("No masks collected. Check model.mask_generator is not None.")
        return

    num_blocks = len(all_masks[0])

    # === 1. 每层 mask 均值 heatmap ===
    fig, ax = plt.subplots(figsize=(14, 6))
    mask_means = np.array([[m.mean() for m in sample] for sample in all_masks])  # [N, num_blocks]
    im = ax.imshow(mask_means.T, aspect='auto', cmap='RdYlBu_r', vmin=0, vmax=1)
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Block Index')
    ax.set_title('Mask Mean per Block (across samples)')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'mask_means_heatmap.png'), dpi=150)
    plt.close()

    # === 2. 每层 mask 方差 ===
    fig, ax = plt.subplots(figsize=(10, 4))
    mask_vars = mask_means.var(axis=0)  # 跨样本的方差
    ax.bar(range(num_blocks), mask_vars)
    ax.set_xlabel('Block Index')
    ax.set_ylabel('Variance (across samples)')
    ax.set_title('Mask Variance per Block (higher = more input-sensitive)')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'mask_variance_per_block.png'), dpi=150)
    plt.close()

    # === 3. 按用户分组对比 ===
    unique_subjects = sorted(set(all_subjects))
    if len(unique_subjects) > 1:
        fig, axes = plt.subplots(min(len(unique_subjects), 5), 1,
                                 figsize=(12, 3*min(len(unique_subjects), 5)))
        if not isinstance(axes, np.ndarray):
            axes = [axes]
        for i, subj in enumerate(unique_subjects[:5]):
            subj_masks = [all_masks[j] for j in range(len(all_masks)) if all_subjects[j] == subj]
            subj_means = np.array([[m.mean() for m in s] for s in subj_masks])
            avg = subj_means.mean(axis=0)
            axes[i].bar(range(num_blocks), avg)
            axes[i].set_title(f'Subject {subj} (avg mask per block)')
            axes[i].set_ylim(0, 1)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'mask_per_subject.png'), dpi=150)
        plt.close()

    # === 4. 通道级 mask 详情 (latent blocks) ===
    # 显示 latent block 的通道掩码分布
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for idx, block_idx in enumerate([0, num_blocks//2, num_blocks-2, num_blocks-1]):
        ax = axes[idx//2][idx%2]
        ch_values = np.array([all_masks[i][block_idx] for i in range(min(20, len(all_masks)))])
        im = ax.imshow(ch_values, aspect='auto', cmap='RdYlBu_r', vmin=0, vmax=1)
        ax.set_xlabel('Channel')
        ax.set_ylabel('Sample')
        ax.set_title(f'Block {block_idx} Channel Masks')
        plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'channel_masks_detail.png'), dpi=150)
    plt.close()

    print(f"Visualizations saved to {save_path}")
    print(f"  - mask_means_heatmap.png")
    print(f"  - mask_variance_per_block.png")
    print(f"  - mask_per_subject.png")
    print(f"  - channel_masks_detail.png")


if __name__ == "__main__":
    import argparse
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Phase 2 checkpoint path")
    parser.add_argument("--config", type=str, required=True, help="Config yaml path")
    parser.add_argument("--output_dir", type=str, default="./mask_analysis")
    parser.add_argument("--num_samples", type=int, default=100)
    args = parser.parse_args()

    from omegaconf import OmegaConf
    cfg = OmegaConf.load(args.config)

    # 创建模型 (带 mask generator)
    unet_config = OmegaConf.to_container(cfg.dic_unet_params, resolve=True)
    mask_gen_config = OmegaConf.to_container(cfg.personalization.mask_generator, resolve=True)
    model = EyeOnlyWrapper(unet_config, mask_generator_config=mask_gen_config)

    # 加载 checkpoint
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    print(f"Loaded checkpoint from {args.checkpoint}")

    # 数据
    from dataset.gaze_capture import HDFDataset
    dataset = HDFDataset(
        cfg.data.hdf_path, split="val",
        split_ratio=cfg.data.split_ratio, seed=cfg.seed,
    )
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=2)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    visualize_user_masks(model, dataloader, args.output_dir, args.num_samples, device)
