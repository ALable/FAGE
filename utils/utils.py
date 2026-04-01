import os
import numpy as np
import torch
from typing import Union, List
import torch.nn.functional as F
import shutil
import os.path as osp
import torch
import torch.nn.functional as F

# from models.vae import VAE
# from models.custom_unet import UNet
# from models.unet import UNet


# def load_all_model(
#     unet_model_path=os.path.join("models", "musetalkV15", "unet.pth"),
#     vae_type="sd-vae",
#     unet_config=os.path.join("models", "musetalkV15", "musetalk.json"),
#     device=None,
#     use_depthwise_separable_conv=False,
# ):
#     vae = VAE(
#         model_path = os.path.join("models", vae_type),
#     )
#     print(f"load unet model from {unet_model_path}")
#     unet = UNet(
#         unet_config=unet_config,
#         model_path=unet_model_path,
#         device=device,
#         use_depthwise_separable_conv=use_depthwise_separable_conv
#     )
#     # pe = PositionalEncoding(d_model=384)
#     return vae, unet

def get_file_type(video_path):
    _, ext = os.path.splitext(video_path)

    if ext.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
        return 'image'
    elif ext.lower() in ['.avi', '.mp4', '.mov', '.flv', '.mkv']:
        return 'video'
    else:
        return 'unsupported'

def cast_training_params(
    model: Union[torch.nn.Module, List[torch.nn.Module]],
    dtype=torch.float32,
):
    if not isinstance(model, list):
        model = [model]
    for m in model:
        for param in m.parameters():
            # only upcast trainable parameters into fp32
            if param.requires_grad:
                param.data = param.to(dtype)

def rand_log_normal(
    shape,
    loc=0.,
    scale=1.,
    device='cpu',
    dtype=torch.float32,
    generator=None
):
    """Draws samples from an lognormal distribution."""
    rnd_normal = torch.randn(
        shape, device=device, dtype=dtype, generator=generator)  # N(0, I)
    sigma = (rnd_normal * scale + loc).exp()
    return sigma

def save_models(cfg, accelerator, net, save_dir, global_step, gaze_mlp=None, logger=None, optimizer=None, lr_scheduler=None,
                discriminator=None, eye_discriminator=None, optimizer_D=None, eye_optimizer_D=None,
                scheduler_D=None, eye_scheduler_D=None, **kwargs):
    """Save model checkpoints including UNet, GazeNet MLP, and Discriminators"""
    import datetime

    unwarp_net = accelerator.unwrap_model(net)

    # Create combined checkpoint with essential information
    combined_state_dict = {
        # Model weights
        'model_state_dict': unwarp_net.state_dict(),
        'global_step': global_step,

        # Training state
        'training_config': {
            'max_train_steps': cfg.solver.max_train_steps,
            'learning_rate': cfg.solver.learning_rate,
            'batch_size': cfg.data.train_bs,
        },

        # Metadata
        'checkpoint_metadata': {
            'save_timestamp': datetime.datetime.now().isoformat(),
            'exp_name': cfg.exp_name,
        }
    }

    # Add GazeNet MLP if available
    if gaze_mlp is not None:
        combined_state_dict['gaze_mlp_state_dict'] = accelerator.unwrap_model(gaze_mlp).state_dict()

    # Add optimizer and scheduler states if available
    if optimizer is not None:
        try:
            combined_state_dict['optimizer_state_dict'] = optimizer.state_dict()
            if logger:
                logger.info("✅ Optimizer state included in combined checkpoint")
        except Exception as e:
            if logger:
                logger.warning(f"⚠️ Failed to save optimizer state: {e}")

    if lr_scheduler is not None:
        try:
            combined_state_dict['lr_scheduler_state_dict'] = lr_scheduler.state_dict()
            if logger:
                logger.info("✅ LR Scheduler state included in combined checkpoint")
        except Exception as e:
            if logger:
                logger.warning(f"⚠️ Failed to save scheduler state: {e}")

    # Add discriminator states if available
    if discriminator is not None:
        try:
            combined_state_dict['discriminator_state_dict'] = accelerator.unwrap_model(discriminator).state_dict()
            if logger:
                logger.info("✅ Discriminator weights included in combined checkpoint")
        except Exception as e:
            if logger:
                logger.warning(f"⚠️ Failed to save discriminator state: {e}")

    if eye_discriminator is not None:
        try:
            combined_state_dict['eye_discriminator_state_dict'] = accelerator.unwrap_model(eye_discriminator).state_dict()
            if logger:
                logger.info("✅ Eye Discriminator weights included in combined checkpoint")
        except Exception as e:
            if logger:
                logger.warning(f"⚠️ Failed to save eye discriminator state: {e}")

    if optimizer_D is not None:
        try:
            combined_state_dict['optimizer_D_state_dict'] = optimizer_D.state_dict()
            if logger:
                logger.info("✅ Discriminator optimizer state included in combined checkpoint")
        except Exception as e:
            if logger:
                logger.warning(f"⚠️ Failed to save discriminator optimizer state: {e}")

    if eye_optimizer_D is not None:
        try:
            combined_state_dict['eye_optimizer_D_state_dict'] = eye_optimizer_D.state_dict()
            if logger:
                logger.info("✅ Eye discriminator optimizer state included in combined checkpoint")
        except Exception as e:
            if logger:
                logger.warning(f"⚠️ Failed to save eye discriminator optimizer state: {e}")

    if scheduler_D is not None:
        try:
            combined_state_dict['scheduler_D_state_dict'] = scheduler_D.state_dict()
            if logger:
                logger.info("✅ Discriminator scheduler state included in combined checkpoint")
        except Exception as e:
            if logger:
                logger.warning(f"⚠️ Failed to save discriminator scheduler state: {e}")

    if eye_scheduler_D is not None:
        try:
            combined_state_dict['eye_scheduler_D_state_dict'] = eye_scheduler_D.state_dict()
            if logger:
                logger.info("✅ Eye discriminator scheduler state included in combined checkpoint")
        except Exception as e:
            if logger:
                logger.warning(f"⚠️ Failed to save eye discriminator scheduler state: {e}")

    # Save combined checkpoint
    combined_path = os.path.join(save_dir, f"combined_checkpoint-{global_step}.pth")
    torch.save(combined_state_dict, combined_path)

    if logger:
        components = ["model weights"]
        if gaze_mlp is not None:
            components.append("gaze_mlp")
        if optimizer is not None:
            components.append("optimizer")
        if lr_scheduler is not None:
            components.append("scheduler")
        if discriminator is not None:
            components.append("discriminator")
        if eye_discriminator is not None:
            components.append("eye_discriminator")
        logger.info(f"Saved combined checkpoint: {' + '.join(components)}")

    # Clean up old combined checkpoints
    if cfg.total_limit is not None:
        checkpoints = os.listdir(save_dir)
        checkpoints = [d for d in checkpoints if d.startswith("combined_checkpoint-") and d.endswith(".pth")]
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1].split(".")[0]))

        if len(checkpoints) > cfg.total_limit:
            num_to_remove = len(checkpoints) - cfg.total_limit
            removing_checkpoints = checkpoints[0:num_to_remove]
            for removing_checkpoint in removing_checkpoints:
                removing_checkpoint_path = os.path.join(save_dir, removing_checkpoint)
                os.remove(removing_checkpoint_path)


def load_checkpoint(checkpoint_path, model_dict, loss_dict, accelerator, logger=None):
    """
    Load checkpoint including generator, discriminators, optimizers, and schedulers

    Args:
        checkpoint_path: Path to checkpoint file
        model_dict: Dictionary containing 'net', 'gaze_mlp', 'optimizer', 'lr_scheduler'
        loss_dict: Dictionary containing discriminator models and optimizers
        accelerator: Accelerator instance for distributed training
        logger: Optional logger

    Returns:
        global_step: Training step number from checkpoint
    """
    if logger:
        logger.info(f"📂 Loading checkpoint: {checkpoint_path}")
    else:
        accelerator.print(f"📂 Loading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Load generator (DiC UNet)
    if 'model_state_dict' in checkpoint:
        model_dict['net'].load_state_dict(checkpoint['model_state_dict'])
        if logger:
            logger.info("✅ DiC UNet weights loaded")
        else:
            accelerator.print("✅ DiC UNet weights loaded")

    # Load GazeNet MLP
    if 'gaze_mlp_state_dict' in checkpoint and 'gaze_mlp' in model_dict:
        model_dict['gaze_mlp'].load_state_dict(checkpoint['gaze_mlp_state_dict'])
        if logger:
            logger.info("✅ GazeNet MLP weights loaded")
        else:
            accelerator.print("✅ GazeNet MLP weights loaded")

    # Load generator optimizer
    if 'optimizer_state_dict' in checkpoint and 'optimizer' in model_dict:
        model_dict['optimizer'].load_state_dict(checkpoint['optimizer_state_dict'])
        if logger:
            logger.info("✅ Optimizer state loaded")
        else:
            accelerator.print("✅ Optimizer state loaded")

    # Load generator scheduler
    if 'lr_scheduler_state_dict' in checkpoint and 'lr_scheduler' in model_dict:
        model_dict['lr_scheduler'].load_state_dict(checkpoint['lr_scheduler_state_dict'])
        if logger:
            logger.info("✅ LR Scheduler state loaded")
        else:
            accelerator.print("✅ LR Scheduler state loaded")

    # Load discriminator states
    if 'discriminator_state_dict' in checkpoint and 'discriminator' in loss_dict:
        try:
            loss_dict['discriminator'].load_state_dict(checkpoint['discriminator_state_dict'])
            if logger:
                logger.info("✅ Discriminator weights loaded")
            else:
                accelerator.print("✅ Discriminator weights loaded")
        except Exception as e:
            msg = f"⚠️ Failed to load discriminator weights: {e}"
            if logger:
                logger.warning(msg)
            else:
                accelerator.print(msg)

    if 'eye_discriminator_state_dict' in checkpoint and 'eye_discriminator' in loss_dict:
        try:
            loss_dict['eye_discriminator'].load_state_dict(checkpoint['eye_discriminator_state_dict'])
            if logger:
                logger.info("✅ Eye Discriminator weights loaded")
            else:
                accelerator.print("✅ Eye Discriminator weights loaded")
        except Exception as e:
            msg = f"⚠️ Failed to load eye discriminator weights: {e}"
            if logger:
                logger.warning(msg)
            else:
                accelerator.print(msg)

    if 'optimizer_D_state_dict' in checkpoint and 'optimizer_D' in loss_dict:
        try:
            loss_dict['optimizer_D'].load_state_dict(checkpoint['optimizer_D_state_dict'])
            if logger:
                logger.info("✅ Discriminator optimizer state loaded")
            else:
                accelerator.print("✅ Discriminator optimizer state loaded")
        except Exception as e:
            msg = f"⚠️ Failed to load discriminator optimizer state: {e}"
            if logger:
                logger.warning(msg)
            else:
                accelerator.print(msg)

    if 'eye_optimizer_D_state_dict' in checkpoint and 'eye_optimizer_D' in loss_dict:
        try:
            loss_dict['eye_optimizer_D'].load_state_dict(checkpoint['eye_optimizer_D_state_dict'])
            if logger:
                logger.info("✅ Eye discriminator optimizer state loaded")
            else:
                accelerator.print("✅ Eye discriminator optimizer state loaded")
        except Exception as e:
            msg = f"⚠️ Failed to load eye discriminator optimizer state: {e}"
            if logger:
                logger.warning(msg)
            else:
                accelerator.print(msg)

    if 'scheduler_D_state_dict' in checkpoint and 'scheduler_D' in loss_dict:
        try:
            loss_dict['scheduler_D'].load_state_dict(checkpoint['scheduler_D_state_dict'])
            if logger:
                logger.info("✅ Discriminator scheduler state loaded")
            else:
                accelerator.print("✅ Discriminator scheduler state loaded")
        except Exception as e:
            msg = f"⚠️ Failed to load discriminator scheduler state: {e}"
            if logger:
                logger.warning(msg)
            else:
                accelerator.print(msg)

    if 'eye_scheduler_D_state_dict' in checkpoint and 'eye_scheduler_D' in loss_dict:
        try:
            loss_dict['eye_scheduler_D'].load_state_dict(checkpoint['eye_scheduler_D_state_dict'])
            if logger:
                logger.info("✅ Eye discriminator scheduler state loaded")
            else:
                accelerator.print("✅ Eye discriminator scheduler state loaded")
        except Exception as e:
            msg = f"⚠️ Failed to load eye discriminator scheduler state: {e}"
            if logger:
                logger.warning(msg)
            else:
                accelerator.print(msg)

    # Get global step
    if 'global_step' in checkpoint:
        global_step = checkpoint['global_step']
    else:
        # Fallback: extract from filename
        import os
        filename = os.path.basename(checkpoint_path)
        global_step = int(filename.split("-")[1].split(".")[0])

    return global_step



def log_training_images_to_tensorboard(
    image, image_pred, mask, norm_head, norm_gaze,
    accelerator, global_step, max_samples=4, eye_region=None
):
    """Log training images to tensorboard for visualization"""
    import matplotlib.pyplot as plt
    import numpy as np
    from torch.utils.tensorboard import SummaryWriter

    # Convert tensors to cpu and numpy
    image = image.detach().cpu()
    image_pred = image_pred.detach().cpu()
    mask = mask.detach().cpu()
    norm_head = norm_head.detach().cpu().numpy()
    norm_gaze = norm_gaze.detach().cpu().numpy()

    # Normalize images to [0, 1] range for visualization
    image = torch.clamp((image + 1.0) / 2.0, 0, 1)
    image_pred = torch.clamp((image_pred + 1.0) / 2.0, 0, 1)

    # Process eye_texture_region if available
    has_eye_texture = eye_region is not None
    if has_eye_texture:
        eye_region = eye_region.detach().cpu()
        eye_region = torch.clamp((eye_region + 1.0) / 2.0, 0, 1)

    batch_size = min(max_samples, image.shape[0])

    # Create a grid of images for comparison (4 rows if eye_texture available, else 3)
    num_rows = 4 if has_eye_texture else 3
    fig, axes = plt.subplots(num_rows, batch_size, figsize=(4*batch_size, 4*num_rows))
    if batch_size == 1:
        axes = axes.reshape(-1, 1)

    for i in range(batch_size):
        # Ground truth
        gt_img = image[i].permute(1, 2, 0).numpy()
        axes[0, i].imshow(gt_img)
        axes[0, i].set_title(f"GT - Head:[{norm_head[i, 0]:.2f},{norm_head[i, 1]:.2f}]\nGaze:[{norm_gaze[i, 0]:.2f},{norm_gaze[i, 1]:.2f}]", fontsize=8)
        axes[0, i].axis('off')

        # Prediction
        pred_img = image_pred[i].permute(1, 2, 0).numpy()
        axes[1, i].imshow(pred_img)
        axes[1, i].set_title("Prediction", fontsize=8)
        axes[1, i].axis('off')

        # Masked input (eyes occluded)
        mask_expanded = mask[i].unsqueeze(0).repeat(3, 1, 1)
        masked_input = image[i] * (1.0 - mask_expanded)
        masked_img = masked_input.permute(1, 2, 0).numpy()
        axes[2, i].imshow(masked_img)
        axes[2, i].set_title("Input (Eyes Occluded)", fontsize=8)
        axes[2, i].axis('off')

        # Reference Eye Texture (if available)
        if has_eye_texture:
            eye_texture_img = eye_region[i].permute(1, 2, 0).numpy()
            axes[3, i].imshow(eye_texture_img)
            axes[3, i].set_title("Reference Eye Texture", fontsize=8)
            axes[3, i].axis('off')

    plt.tight_layout()

    # Log to tensorboard
    if hasattr(accelerator, 'trackers'):
        for tracker in accelerator.trackers:
            if tracker.name == "tensorboard":
                tracker.writer.add_figure("train_samples/eye_inpainting", fig, global_step=global_step)

    plt.close(fig)



def normalize_image_for_perception(image, input_range=(-1, 1), clamp=True):
    """
    Normalize image tensor for perceptual loss models

    Args:
        image: Image tensor [B, C, H, W]
        input_range: Tuple (min, max) of input image range. Default: (-1, 1)
        clamp: Whether to clamp output to [0, 1]. Default: True

    Returns:
        Normalized image tensor in [0, 1] range

    Examples:
        >>> # For images in [-1, 1] range (common for diffusion models)
        >>> normalized = normalize_image_for_perception(image)

        >>> # For images already in [0, 1] range
        >>> normalized = normalize_image_for_perception(image, input_range=(0, 1))
    """
    min_val, max_val = input_range
    # Linear normalization: (x - min) / (max - min):
    normalized = (image - min_val) / (max_val - min_val)

    if clamp:
        normalized = torch.clamp(normalized, 0, 1)

    return normalized


def denormalize_image(image, output_range=(-1, 1), input_range=(0, 1)):
    """
    Denormalize image tensor from [0, 1] back to original range

    Args:
        image: Normalized image tensor [B, C, H, W] in [0, 1] range
        output_range: Tuple (min, max) of desired output range. Default: (-1, 1)
        input_range: Tuple (min, max) of input range. Default: (0, 1)

    Returns:
        Denormalized image tensor
    """
    in_min, in_max = input_range
    out_min, out_max = output_range

    # Linear denormalization: x * (out_max - out_min) + out_min
    denormalized = (image - in_min) / (in_max - in_min) * (out_max - out_min) + out_min

    return denormalized




def crop_eyes(images, masks, output_size=(64, 64)):
    """
    Crop left and right eyes from images using masks and concatenate along channel dimension.

    1. 计算 mask 的行列投影 (rows/cols)
    2. 找到整体的 bounding box
    3. 根据 x_indices 的中点 (x_mid) 区分左右眼
    4. 沿 channel 维度拼接左右眼

    Args:
        images: [B, 3, 256, 256]
        masks: [B, 256, 256] (Dataset 已经保证这是包含眼周的正方形 Mask)
        output_size: crop single eye region size (默认64x64)

    Returns:
        eye_crops: [B, 6, 64, 64] - 左眼(前3通道) + 右眼(后3通道)
    """
    batch_size = images.shape[0]
    device = images.device
    eye_crops_list = []

    for b in range(batch_size):
        img = images[b]   # [3, 256, 256]
        mask = masks[b]   # [256, 256]

        # 1. 寻找非零区域
        # rows: 投影到 Y 轴 (看哪一行有像素)
        # cols: 投影到 X 轴 (看哪一列有像素)
        rows = torch.any(mask > 0.5, dim=1)
        cols = torch.any(mask > 0.5, dim=0)

        # 如果全是黑的，返回全黑
        if not torch.any(rows) or not torch.any(cols):
            eye_crop = torch.zeros(6, output_size[0], output_size[1], device=device)
            eye_crops_list.append(eye_crop)
            continue

        # 2. 获取整体 Bounding Box (复刻原始逻辑: y_min, y_max 是全局的)
        y_indices = torch.where(rows)[0]
        y_min, y_max = y_indices[0], y_indices[-1]
        
        x_indices = torch.where(cols)[0]
        
        # 3. 计算中点进行左右分割 (复刻原始逻辑: x_mid)
        x_mid = (x_indices[0] + x_indices[-1]) // 2

        # --- 左眼处理 (x < x_mid) ---
        # 找到属于左眼的列索引
        left_cols_mask = (x_indices < x_mid)
        
        if torch.any(left_cols_mask):
            left_valid_indices = x_indices[left_cols_mask]
            left_x_min = left_valid_indices[0]
            left_x_max = left_valid_indices[-1]
            left_eye_region = img[:, y_min:y_max+1, left_x_min:left_x_max+1]
            
            left_eye_resized = F.interpolate(
                left_eye_region.unsqueeze(0),
                size=output_size,
                mode='bilinear',
                align_corners=False
            ).squeeze(0)  # [3, 64, 64]
        else:
            raise ValueError("left_cols_mask not exist")

        # --- 右眼处理 (x >= x_mid) ---
        right_cols_mask = (x_indices >= x_mid)

        if torch.any(right_cols_mask):
            right_valid_indices = x_indices[right_cols_mask]
            right_x_min = right_valid_indices[0]
            right_x_max = right_valid_indices[-1]

            # 裁剪
            right_eye_region = img[:, y_min:y_max+1, right_x_min:right_x_max+1]

            right_eye_resized = F.interpolate(
                right_eye_region.unsqueeze(0),
                size=output_size,
                mode='bilinear',
                align_corners=False
            ).squeeze(0)  # [3, 64, 64]
        else:
             raise ValueError("right_cols_mask not exist")

        # 4. 沿 channel 维度拼接 (Left + Right) -> [6, 64, 64]
        # 前3个通道是左眼，后3个通道是右眼
        eye_crop = torch.cat([left_eye_resized, right_eye_resized], dim=0)
        eye_crops_list.append(eye_crop)

    return torch.stack(eye_crops_list)  # [B, 6, 64, 64]
