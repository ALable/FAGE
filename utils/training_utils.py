import random
import os
import json
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import lpips

import torchvision
from piq import psnr, ssim
from omegaconf import OmegaConf
from diffusers.optimization import get_scheduler

from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset.eth_xgaze import ETHXGazeLMDBDataset
from dataset.gaze_capture import HDFDataset

from models.gazenet import MLPNetwork


from loss.vgg_eye import GazePerceptualLoss
from loss.basic_loss import Interpolate
from loss.basic_loss import gaze_angular_loss as gaze_angular_error
from loss.basic_loss import IDLoss
from loss.discriminator import MultiScaleDiscriminator, PatchGAN


def seed_everything(seed):
    import random
    import numpy as np
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed % (2**32))
    random.seed(seed)

logger = logging.getLogger(__name__)


def initialize_models_and_optimizers_dic(cfg, accelerator, weight_dtype):
    """Initialize DiC UNet models and optimizers"""
    model_dict = {
        'net': None,  # GazeDiCWrapper (direct usage, no extra wrapper needed)
        'gaze_mlp': None,
        'optimizer': None,
        'lr_scheduler': None,
        'scheduler_max_steps': None,
        'trainable_params': None,
        'gaze_estimator_eval': None,
    }

    # Load DiC UNet configuration
    # Priority: YAML dic_unet_params > JSON file (backward compatibility)
    if hasattr(cfg, 'dic_unet_params'):
        # Modern approach: All DiC UNet config unified in YAML dic_unet_params section
        unet_config = OmegaConf.to_container(cfg.dic_unet_params, resolve=True)
    else:
        # Legacy approach: Read from separate JSON file
        unet_config_file = os.path.join(
            cfg.pretrained_model_name_or_path,
            cfg.unet_sub_folder,
            cfg.unet_config_file
        )
        print(f"📄 Loading DiC UNet config from JSON: {unet_config_file}")
        print(f"   (Consider migrating to YAML for easier management)")
        with open(unet_config_file, 'r') as f:
            unet_config = json.load(f)

    # Initialize GazeDiC (pure convolution, no attention)
    # Directly use GazeDiCWrapper - no extra wrapper needed
    if cfg.model_name =="GazeInpainting":
        model_dict['net'] = GazeInpaitingWrapper(
        unet_config=unet_config,
        use_float16=False,  # Will be converted later
        device=accelerator.device
    )
        print("Initinal GazeInpainting")
    else:
        model_dict['net'] = GazeDiCWrapper(
        unet_config=unet_config,
        use_float16=False,  # Will be converted later
        device=accelerator.device
        )
    # Note: Access underlying GazeDiC model via model_dict['net'].model if needed
    # Initialize GazeNet MLP
    gazenet_params = getattr(cfg.model_params, 'gazenet_params', {})
    # Get condition_dim from config
    condition_dim = unet_config.get('condition_dim', 64)
    print(f"📐 GazeNet output dimension: {condition_dim} (matching DiC UNet condition_dim)")

    model_dict['gaze_mlp'] = MLPNetwork(
        num_layers=gazenet_params.get('num_layers', 4),
        num_in=gazenet_params.get('num_in', 2),  # pitch, yaw
        num_hidden=gazenet_params.get('num_hidden', 128),
        num_out=condition_dim,
    )

    # Load checkpoint if specified
    if hasattr(cfg, 'oneshot_params') and cfg.oneshot_params.checkpoint_path:
        checkpoint_path = cfg.oneshot_params.checkpoint_path
        if os.path.exists(checkpoint_path):
            print(f"📂 Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=accelerator.device)

            if 'model_state_dict' in checkpoint:
                try:
                    model_dict['net'].load_state_dict(checkpoint['model_state_dict'])
                    print("✅ DiC UNet weights loaded from checkpoint")
                except Exception as e:
                    print(f"⚠️ Failed to load UNet weights: {e}")
                    print("   Using random initialization")
            else:
                print("⚠️ UNet weights not found in checkpoint, using random initialization")

            if 'gaze_mlp_state_dict' in checkpoint:
                try:
                    model_dict['gaze_mlp'].load_state_dict(checkpoint['gaze_mlp_state_dict'])
                    print("✅ GazeNet MLP weights loaded from checkpoint")
                except Exception as e:
                    print(f"⚠️ Failed to load GazeNet weights: {e}")
                    print("   Using random initialization")
            else:
                print("⚠️ GazeNet MLP weights not found in checkpoint, using random initialization")
        else:
            print(f"⚠️ Checkpoint not found at {checkpoint_path}, using random initialization")

    # Set model states
    model_dict['net'].train()  # GazeDiCWrapper has .train() method
    model_dict['gaze_mlp'].to(accelerator.device, dtype=weight_dtype)
    model_dict['gaze_mlp'].train()

    # Get trainable parameters
    model_dict['trainable_params'] = list(
        filter(lambda p: p.requires_grad, model_dict['net'].parameters()))

    if accelerator.is_main_process:
        print('=== DiC UNet Trainable Parameters Debug ===')
        total_params = sum(p.numel() for p in model_dict['net'].parameters())
        trainable_params_count = sum(p.numel() for p in model_dict['trainable_params'])

        print(f"GazeDiCWrapper total parameters: {total_params:,}")
        print(f"Trainable parameters count: {trainable_params_count:,}")
        print(f"Trainable parameter objects: {len(model_dict['trainable_params'])}")
        print('=== End Debug ===')

    # Initialize optimizer
    if cfg.solver.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer_cls = bnb.optim.AdamW8bit
        except ImportError:
            raise ImportError("Please install bitsandbytes to use 8bit Adam optimizer")
    else:
        optimizer_cls = torch.optim.AdamW

    model_dict['optimizer'] = optimizer_cls(
        model_dict['trainable_params'],
        lr=cfg.solver.learning_rate,
        betas=(cfg.solver.adam_beta1, cfg.solver.adam_beta2),
        weight_decay=cfg.solver.adam_weight_decay,
        eps=cfg.solver.adam_epsilon,
    )

    # Initialize learning rate scheduler
    model_dict['scheduler_max_steps'] = cfg.solver.max_train_steps * cfg.solver.gradient_accumulation_steps
    model_dict['lr_scheduler'] = get_scheduler(
        cfg.solver.lr_scheduler,
        optimizer=model_dict['optimizer'],
        num_warmup_steps=cfg.solver.lr_warmup_steps * cfg.solver.gradient_accumulation_steps,
        num_training_steps=model_dict['scheduler_max_steps'],
    )

    # Initialize Gaze Estimator for validation (ResNet-based)
    print("🔍 Initializing Gaze Estimator (ResNet-based) for validation...")
    from gaze_estimation.baseline_resnet import GazeHeadResNet
    model_dict['gaze_estimator_eval'] = GazeHeadResNet().to(accelerator.device)

    # Load pre-trained weights if available
    gaze_estimator_checkpoint = cfg.pretrained.gaze_eval_checkpoint_path
    if os.path.exists(gaze_estimator_checkpoint):
        checkpoint = torch.load(gaze_estimator_checkpoint, map_location=accelerator.device)
        model_dict['gaze_estimator_eval'].load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Gaze Estimator loaded from: {gaze_estimator_checkpoint}")
    else:
        raise ValueError("resnet gaze estimator checkpoint is not exist")

    model_dict['gaze_estimator_eval'].eval()
    print(f"✅ Gaze Estimator initialized for validation")

    return model_dict


def get_dataset(cfg, split="train"):
    """Get dataset based on configuration"""
    if hasattr(cfg.data, 'dataset_type') and cfg.data.dataset_type == "eth_xgaze":
        return ETHXGazeLMDBDataset(
            lmdb_path=cfg.data.lmdb_path,
            split=split,
            resolution=getattr(cfg.data, 'image_size', 256),
            split_ratio=getattr(cfg.data, 'split_ratio', (0.8, 0.1, 0.1)),
            seed=getattr(cfg, 'seed', 41),
            convert_to_square=getattr(cfg.data, 'convert_to_square', False),
            eye_scale_factor=getattr(cfg.data, 'eye_scale_factor', 1.2),
            return_reference=getattr(cfg.data, 'return_reference', False),
            identity_model=getattr(cfg.data, 'identity_model', 'ArcFace')
        )
    elif hasattr(cfg.data, 'dataset_type') and cfg.data.dataset_type == "gaze_hdf":
        return HDFDataset(
            hdf_file_path=getattr(cfg.data, 'hdf_path', '/mnt/data/xhy/preprocess_sted/output_msted256/GazeCapture.h5'),
            split=split,
            return_original_eye_reference=getattr(cfg.data,"return_original_eye_texture",False),
            return_eye_reference=getattr(cfg.data,'return_eye_texture',False),
            reference_search_radius=getattr(cfg.data,'reference_search_radius',20)
        )
    else:
        raise ValueError(
            f"Unknown dataset_type: {getattr(cfg.data, 'dataset_type', None)}")


def initialize_dataloaders_dic(cfg):
    """Initialize dataloaders for DiC training"""
    train_dataset = get_dataset(cfg, split="train")
    val_dataset = get_dataset(cfg, split="val")
    # test_dataset = get_dataset(cfg, split="test")

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.data.train_bs,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.data.train_bs,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # test_dataloader = torch.utils.data.DataLoader(
    #     test_dataset,
    #     batch_size=cfg.data.train_bs,
    #     shuffle=False,
    #     num_workers=cfg.data.num_workers,
    #     pin_memory=True,
    #     drop_last=False,
    # )

    return {
        'train': train_dataloader,
        'val': val_dataloader,
        # 'test': test_dataloader
    }


def initialize_loss_functions(cfg, accelerator, scheduler_max_steps):
    """Initialize loss functions and discriminators for DiC training"""
    loss_dict = {
        'L1_loss': nn.L1Loss()
    }
    if cfg.loss_params.gaze_perceptual_loss > 0:
        loss_dict["angular_loss"] = GazePerceptualLoss(resize=True, 
        device=accelerator.device,
        path=cfg.pretrained.gaze_train_checkpoint_path
        )

    # Get discriminator type from config (default: patchgan)
    discriminator_type = getattr(cfg.discriminator_train_params, 'discriminator_type', 'patchgan')
    loss_dict['discriminator_type'] = discriminator_type

    # Initialize eye discriminator if eye GAN loss is enabled
    if cfg.loss_params.eye_gan_loss > 0:
        if discriminator_type == 'patchgan':
            # PatchGAN for eye region (6 channels: left eye 3ch + right eye 3ch)
            print("📦 Initializing PatchGAN eye discriminator (6-channel input for cropped eyes)...")
            eye_disc = PatchGAN(
                input_nc=6,  # 左眼3通道 + 右眼3通道
                ndf=getattr(cfg.discriminator_train_params, 'ndf', 64)
            ).to(accelerator.device)
            loss_dict['eye_discriminator'] = eye_disc
            # Wrap with masked loss calculator
            # loss_dict['eye_discriminator_full'] = PatchGANWithMaskLoss(eye_disc)

        elif discriminator_type == 'multi-scale':
            # Multi-Scale eye discriminator (6 channels: left eye 3ch + right eye 3ch)
            print("📦 Initializing Multi-Scale eye discriminator (6-channel input)...")
            eye_disc_params = dict(cfg.model_params.discriminator_params)
            # 修改参数名：image_channel -> num_channels (Discriminator 构造函数的参数名)
            eye_disc_params.pop('image_channel', None)  # 移除旧参数
            eye_disc_params['num_channels'] = 6  # 左眼3通道 + 右眼3通道
            loss_dict['eye_discriminator'] = MultiScaleDiscriminator(
                **eye_disc_params).to(accelerator.device)
        else:
            raise ValueError(f"Unknown discriminator_type: {discriminator_type}")

        loss_dict['eye_optimizer_D'] = optim.AdamW(
            loss_dict['eye_discriminator'].parameters(),
            lr=cfg.discriminator_train_params.lr,
            betas=cfg.discriminator_train_params.betas,
            weight_decay=cfg.discriminator_train_params.weight_decay,
            eps=cfg.discriminator_train_params.eps
        )
        loss_dict['eye_scheduler_D'] = CosineAnnealingLR(
            loss_dict['eye_optimizer_D'],
            T_max=scheduler_max_steps,
            eta_min=cfg.discriminator_train_params.lr * 0.01
        )

    return loss_dict


# def initialize_vgg(cfg, accelerator):
#     if cfg.loss_params.vgg_loss > 0:
#         vgg_IN = vgg_face.Vgg19().to(accelerator.device)
#         pyramid = vgg_face.ImagePyramide(
#             cfg.loss_params.pyramid_scale, 3).to(accelerator.device)
#         vgg_IN.eval()
#         downsampler = Interpolate(
#             size=(224, 224), mode='bilinear', align_corners=False)
#         return vgg_IN, pyramid, downsampler
#     else:
#         return None, None, None


def prepare_model_input(image, mask, cfg, batch=None):
    """Prepare model input based on in_channels configuration

    Args:
        image: [B, 3, H, W] input image
        mask: [B, 1, H, W] or [B, H, W] eye mask
        cfg: configuration object
        batch: optional batch dict for reference image

    Returns:
        model_input: [B, C, H, W] where C depends on in_channels config
    """
    # Ensure mask has correct dimensions
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)  # [B, H, W] -> [B, 1, H, W]

    # Get model in_channels configuration
    model_in_channels = cfg.dic_unet_params.get('in_channels', 3)

    if model_in_channels == 6:
        # DiC V3: 6-channel input (Masked RGB + Reference RGB)
        if batch is not None and 'reference_image' in batch:
            reference_image = batch['reference_image']  # [B, 3, H, W]
            masked_image = image * (1 - mask)  # [B, 3, H, W]
            model_input = torch.cat([masked_image, reference_image], dim=1)  # [B, 6, H, W]
        else:
            raise ValueError("6-channel input requires 'reference_image' in batch. Set return_reference=True in dataset config.")
    elif model_in_channels == 4:
        # DiC V2: 4-channel input (RGB + Mask concat)
        masked_image = image * (1 - mask)  # [B, 3, H, W]
        model_input = torch.cat([masked_image, mask], dim=1)  # [B, 4, H, W]
    else:
        # DiC V1: 3-channel input (Masked RGB)
        model_input = image * (1 - mask)  # [B, 3, H, W]

    return model_input


def validation(cfg, val_dataloader, accelerator, save_dir, global_step, weight_dtype, model_dict):
    """Validation function with LPIPS, Gaze Error, ID Distance, SSIM, and PSNR metrics"""


    # Extract models from model_dict
    net = model_dict['net']
    gaze_mlp = model_dict['gaze_mlp']
    gaze_estimator_eval = model_dict['gaze_estimator_eval']

    net.eval()
    gaze_mlp.eval()

    lpips_fn = lpips.LPIPS(net='alex').to(accelerator.device)
    lpips_fn.eval()

    # Initialize IDLoss (ArcFace-based) for validation identity distance
    id_loss_module = IDLoss(multiscale=True).to(accelerator.device)
    id_loss_module.eval()
    for param in id_loss_module.parameters():
        param.requires_grad = False

    # Initialize SSIM and PSNR metrics (using piq - already imported at top)
    # Note: piq functions are called directly, no need to instantiate

    L1_score = 0.0
    lpips_score = 0.0
    gaze_error = 0.0
    id_distance_score = 0.0
    ssim_score = 0.0
    psnr_score = 0.0
    eye_lpips_score = 0.0 
    eye_L1_score = 0.0
    num_batches = 0

    # Store selected batch images for TensorBoard
    selected_batch_gt = None
    selected_batch_pred = None
    selected_batch_masked_input = None

    # Transform for tensors (initialized ONCE outside loop)
    eval_trans = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Get validation batch limit from config (default: 5)
    val_batch_limit = getattr(cfg, 'val_batch_limit', 5)

    # Number of samples to visualize from ALL batches
    num_vis_samples = getattr(cfg, 'num_vis_samples', 4)

    # Storage for all validation samples (for random selection later)
    all_samples = []  # List of dicts containing sample data

    # Calculate sampling interval to ensure diverse subject coverage
    total_val_batches = len(val_dataloader)
    sample_interval = max(1, total_val_batches // val_batch_limit)  # Sample every N batches

    if accelerator.is_main_process:
        print(f"📊 Validation: Sampling {val_batch_limit} batches from {total_val_batches} total batches (interval={sample_interval})")
        print(f"📊 Will randomly select {num_vis_samples} samples from sampled batches")

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dataloader):
            # Skip batches based on sampling interval to ensure diverse subject coverage
            if batch_idx % sample_interval != 0:
                continue

            # Extract data from batch
            image = batch['image'].to(accelerator.device, dtype=weight_dtype)
            norm_head = batch['norm_head'].to(accelerator.device, dtype=weight_dtype)
            norm_gaze = batch['norm_gaze'].to(accelerator.device, dtype=weight_dtype)
            mask = batch['mask'].to(accelerator.device, dtype=weight_dtype)


            eye_region = batch.get('eye_reference', None)
            eye_bbox = batch.get('eye_bbox', None)
            if eye_region is not None:
                eye_region = eye_region.to(accelerator.device, dtype=weight_dtype)
            if eye_bbox is not None:
                eye_bbox = eye_bbox.to(accelerator.device, dtype=weight_dtype)

            # Expand mask dimension
            mask = mask.unsqueeze(1)
            # Generate gaze embeddings
            head_embedding, gaze_embedding = gaze_mlp(norm_head, norm_gaze)
            head_embedding = head_embedding.unsqueeze(1)
            gaze_embedding = gaze_embedding.unsqueeze(1)
            gaze_prompt = torch.cat([head_embedding, gaze_embedding], dim=1)
            # gaze_prompt=gaze_embedding
            # Prepare input based on model in_channels configuration
            model_in_channels = cfg.dic_unet_params.get('in_channels', 3)

            if model_in_channels == 6:
                # DiC V3: 6-channel input (Masked RGB + Reference RGB)
                if 'reference_image' in batch:
                    reference_image = batch['reference_image'].to(accelerator.device, dtype=weight_dtype)  # [B, 3, H, W]
                    masked_image = image * (1 - mask)  # [B, 3, H, W]
                    model_input = torch.cat([masked_image, reference_image], dim=1)  # [B, 6, H, W]
                else:
                    raise ValueError("6-channel input requires 'reference_image' in batch. Set return_reference=True in dataset config.")
            elif model_in_channels == 4:
                # DiC V2: 4-channel input (RGB + Mask concat)
                model_input = torch.cat([image, mask], dim=1)  # [B, 4, H, W]
            else:
                # DiC V1: 3-channel input (Masked RGB)
                model_input = image * (1 - mask)  # [B, 3, H, W]

            # Forward pass through DiC UNet (no timestep, no identity)

            model_output = net(model_input, encoder_hidden_states=gaze_prompt, eye_region=eye_region, eye_bbox=eye_bbox)
            image_pred = model_output.sample
            # Calculate validation loss (L1)
            loss = nn.functional.l1_loss(image, image_pred)
            L1_score += loss.item()
            lpips_value = lpips_fn(image, image_pred)
            lpips_score += torch.mean(lpips_value)

            # Calculate SSIM and PSNR for full image (using piq functions)
            # piq expects values in [0, data_range], so shift from [-1, 1] to [0, 2]
            image_pred_shifted = (image_pred + 1.0)  # [-1, 1] -> [0, 2]
            image_shifted = (image + 1.0)  # [-1, 1] -> [0, 2]
            ssim_value = ssim(image_pred_shifted, image_shifted, data_range=2.0)
            ssim_score += ssim_value.item()
            psnr_value = psnr(image_pred_shifted, image_shifted, data_range=2.0)
            psnr_score += psnr_value.item()

            # 计算眼睛区域的 L1 和 LPIPS
            # mask: [B, 1, H, W], 1表示眼睛区域
            eye_region_gt = image * mask+ 1-mask
            eye_region_pred = image_pred * mask+1-mask

            # 眼睛区域 L1 loss (只计算眼睛区域的像素)
            eye_l1 = nn.functional.l1_loss(eye_region_gt, eye_region_pred, reduction='mean')
            eye_L1_score += eye_l1.item()

            eye_lpips_value = lpips_fn(eye_region_gt, eye_region_pred)
            eye_lpips_score += torch.mean(eye_lpips_value)

            from utils.utils import normalize_image_for_perception
            image_pred_normalized = normalize_image_for_perception(image_pred)
            image_pred_normalized = eval_trans(image_pred_normalized)
            gaze_pred, _ = gaze_estimator_eval(image_pred_normalized)
            gaze_error += gaze_angular_error(norm_gaze, gaze_pred)

            # Calculate ID Distance using IDLoss features (cosine distance)
            feats_real = id_loss_module(image)          # [B, 512]
            feats_pred = id_loss_module(image_pred)     # [B, 512]
            cosine_sim = F.cosine_similarity(feats_real, feats_pred, eps=1e-6, dim=1)
            id_distance = (1.0 - cosine_sim).mean().item()
            id_distance_score += id_distance

            num_batches += 1

            # Store all samples from this batch for potential visualization
            batch_size = image.shape[0]
            for sample_idx in range(batch_size):
                sample_dict = {
                    'batch_idx': batch_idx,
                    'sample_idx': sample_idx,
                    'image': image[sample_idx].cpu(),
                    'image_pred': image_pred[sample_idx].cpu(),
                    'mask': mask[sample_idx].cpu(),
                    'norm_head': norm_head[sample_idx].cpu(),
                    'norm_gaze': norm_gaze[sample_idx].cpu(),
                    'gaze_pred': gaze_pred[sample_idx].cpu(),
                }
                # Add eye texture region if available
                if eye_region is not None:
                    sample_dict['eye_reference'] = eye_region[sample_idx].cpu()

                all_samples.append(sample_dict)

            # Stop after collecting enough batches
            if num_batches >= val_batch_limit:
                break

    avg_L1 = L1_score / num_batches
    avg_lpips = lpips_score / num_batches
    avg_gaze_error = gaze_error / num_batches
    avg_id_distance = id_distance_score / num_batches
    avg_ssim = ssim_score / num_batches
    avg_psnr = psnr_score / num_batches
    # 新增: 眼睛区域指标平均值
    avg_eye_L1 = eye_L1_score / num_batches
    avg_eye_lpips = eye_lpips_score / num_batches

    # Randomly select samples from all collected samples for visualization
    selected_batch_gt = None
    selected_batch_pred = None
    selected_batch_masked_input = None

    if len(all_samples) > 0 and accelerator.is_main_process:
        # Randomly select num_vis_samples from all samples
        num_available = len(all_samples)
        num_to_select = min(num_vis_samples, num_available)
        selected_sample_objects = random.sample(all_samples, num_to_select)

        print(f"✅ Randomly selected {num_to_select} samples from {num_available} total samples")
        selected_info = [(s['batch_idx'], s['sample_idx']) for s in selected_sample_objects]
        print(f"   Selected samples (batch_idx, sample_idx): {selected_info}")

        # Stack selected samples for TensorBoard
        selected_batch_gt = torch.stack([s['image'] for s in selected_sample_objects])
        selected_batch_pred = torch.stack([s['image_pred'] for s in selected_sample_objects])
        selected_batch_mask = torch.stack([s['mask'] for s in selected_sample_objects])
        selected_batch_masked_input = selected_batch_gt * (1 - selected_batch_mask)

        # Save validation samples to disk
        for vis_idx, sample in enumerate(selected_sample_objects):
            # Use original batch_idx and sample_idx for meaningful filenames
            original_batch_idx = sample['batch_idx']
            original_sample_idx = sample['sample_idx']

            # Get eye texture region if available
            eye_texture_for_vis = sample.get('eye_reference', None)
            if eye_texture_for_vis is not None:
                eye_texture_for_vis = eye_texture_for_vis.unsqueeze(0)  # Add batch dimension

            save_validation_samples(
                sample['image'].unsqueeze(0),      # Add batch dimension
                sample['image_pred'].unsqueeze(0),
                sample['mask'].unsqueeze(0),
                sample['norm_gaze'].unsqueeze(0),  # norm_gaze (not norm_head)
                save_dir, global_step,
                batch_idx=original_batch_idx,      # Use original batch_idx
                gaze_pred=sample['gaze_pred'].unsqueeze(0),
                reference_images=eye_texture_for_vis,  # Pass eye texture region as reference
                sample_idx=original_sample_idx     # Pass original sample_idx for filename
            )

    # Log to TensorBoard
    log_dict = {
        "val/L1": avg_L1,
        "val/lpips": avg_lpips,
        "val/gaze_error": avg_gaze_error,
        "val/id_distance": avg_id_distance,
        "val/ssim": avg_ssim,
        "val/psnr": avg_psnr,
        "val/eye_L1": avg_eye_L1,
        "val/eye_lpips": avg_eye_lpips,
    }

    # Add validation images to TensorBoard
    if accelerator.is_main_process and selected_batch_gt is not None:
        # Normalize images to [0, 1] for display
        from utils.utils import normalize_image_for_perception
        gt_display = normalize_image_for_perception(selected_batch_gt)
        pred_display = normalize_image_for_perception(selected_batch_pred)
        masked_input_display = normalize_image_for_perception(selected_batch_masked_input)

        # Create comparison grid
        comparison = torch.cat([
            masked_input_display,  # Masked input
            pred_display,          # Prediction
            gt_display,            # Ground truth
        ], dim=0)  # Stack vertically

        # Create grid
        grid = torchvision.utils.make_grid(comparison, nrow=selected_batch_gt.shape[0], padding=2)
        # Log to TensorBoard
        log_dict["val/comparison"] = grid
        # Also log individual images
        log_dict["val/ground_truth"] = torchvision.utils.make_grid(gt_display, nrow=4, padding=2)
        log_dict["val/prediction"] = torchvision.utils.make_grid(pred_display, nrow=4, padding=2)
        log_dict["val/masked_input"] = torchvision.utils.make_grid(masked_input_display, nrow=4, padding=2)

    accelerator.log(log_dict, step=global_step)

    # Print validation metrics
    if accelerator.is_main_process:
        # Write metrics to file
        metrics_file = os.path.join(save_dir, "metrics.txt")
        with open(metrics_file, 'a', encoding='utf-8') as f:
            f.write(f"Step {global_step}:\n")
            f.write(f"  L1 Loss: {avg_L1:.6f}\n")
            f.write(f"  LPIPS: {avg_lpips:.6f}\n")
            f.write(f"  SSIM: {avg_ssim:.6f}\n")
            f.write(f"  PSNR: {avg_psnr:.4f} dB\n")
            f.write(f"  Gaze Error: {avg_gaze_error:.4f}°\n")
            f.write(f"  ID Distance: {avg_id_distance:.6f}\n")
            f.write(f"  Eye L1 Loss: {avg_eye_L1:.6f}\n")
            f.write(f"  Eye LPIPS: {avg_eye_lpips:.6f}\n")
            f.write("\n")
        print(f"📝 Validation metrics saved to: {metrics_file}")

    net.train()
    gaze_mlp.train()
    # Return validation metrics
    metrics = {
        'L1': avg_L1,
        'LPIPS': avg_lpips,
        'SSIM': avg_ssim,
        'PSNR': avg_psnr,
        'Gaze_Error': avg_gaze_error,
        'ID_Distance': avg_id_distance,
        'Eye_L1': avg_eye_L1,
        'Eye_LPIPS': avg_eye_lpips
    }
    return metrics


def save_validation_samples(gt_images, pred_images, masks, norm_gaze, save_dir, global_step, batch_idx, gaze_pred=None, reference_images=None, sample_idx=None):
    """Save validation samples for DiC UNet with gaze arrows

    Args:
        reference_images: Optional reference images [B, 3, H, W]. If provided, will be displayed instead of Absolute Difference.
        sample_idx: Optional original sample index within batch for filename. If None, uses loop index.
    """
    import matplotlib.pyplot as plt
    import os
    from utils.draw_gaze_arrow import draw_gaze_arrow

    os.makedirs(os.path.join(save_dir, "validation_samples"), exist_ok=True)

    batch_size = gt_images.shape[0]
    num_samples = min(batch_size, 4)

    for i in range(num_samples):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Convert tensors to numpy
        gt_img = gt_images[i].permute(1, 2, 0).numpy()
        pred_img = pred_images[i].permute(1, 2, 0).numpy()
        mask_img = masks[i, 0].numpy()

        # Normalize images to [0, 1]
        gt_img = (gt_img + 1.0) / 2.0
        pred_img = (pred_img + 1.0) / 2.0
        gt_img = np.clip(gt_img, 0, 1)
        pred_img = np.clip(pred_img, 0, 1)

        # Ground truth
        axes[0, 0].imshow(gt_img)
        axes[0, 0].set_title("Ground Truth", fontsize=14)
        axes[0, 0].axis('off')

        # Predicted image with gaze arrows
        pred_img_with_arrows = (pred_img * 255).astype(np.uint8).copy()
        
        # Draw target gaze arrow (green)
        target_gaze = norm_gaze[i].numpy()
        # pred_img_with_arrows = draw_gaze_arrow(
        #     pred_img_with_arrows, 
        #     target_gaze, 
        #     thickness=3, 
        #     color=(0, 255, 0)  # Green for target
        # )
        
        # Draw predicted gaze arrow (red) if available
        if gaze_pred is not None:
            predicted_gaze = gaze_pred[i].numpy()
            # pred_img_with_arrows = draw_gaze_arrow(
            #     pred_img_with_arrows, 
            #     predicted_gaze, 
            #     thickness=3, 
            #     color=(0, 0, 255)  # Red for prediction
            # )
        
        # Image is already in RGB format, no conversion needed
        pred_img_with_arrows = pred_img_with_arrows.astype(np.float32) / 255.0
        
        axes[0, 1].imshow(pred_img_with_arrows)
        title = "Prediction (Green=Target"
        if gaze_pred is not None:
            title += ", Red=Predicted)"
        else:
            title += ")"
        axes[0, 1].set_title(title, fontsize=14)
        axes[0, 1].axis('off')

        # Mask
        axes[0, 2].imshow(mask_img, cmap='gray')
        axes[0, 2].set_title("Eye Mask", fontsize=14)
        axes[0, 2].axis('off')

        # Second row, first position: Reference image (if available) or Absolute Difference
        if reference_images is not None:
            # Display reference image
            ref_img = reference_images[i].permute(1, 2, 0).numpy()
            ref_img = (ref_img + 1.0) / 2.0
            ref_img = np.clip(ref_img, 0, 1)
            axes[1, 0].imshow(ref_img)
            axes[1, 0].set_title("Reference Image", fontsize=14)
            axes[1, 0].axis('off')
        else:
            # Display difference map
            diff = np.abs(gt_img - pred_img)
            axes[1, 0].imshow(diff)
            axes[1, 0].set_title("Absolute Difference", fontsize=14)
            axes[1, 0].axis('off')

        # Masked input
        masked_input = gt_img * (1 - np.expand_dims(mask_img, axis=2))
        axes[1, 1].imshow(masked_input)
        axes[1, 1].set_title("Masked Input", fontsize=14)
        axes[1, 1].axis('off')

        # Gaze information with numerical values and error
        y_pos = 0.9
        axes[1, 2].text(0.1, y_pos, f"Target Gaze:", fontsize=12,
                       fontweight='bold', transform=axes[1, 2].transAxes, color='green')
        y_pos -= 0.08
        axes[1, 2].text(0.1, y_pos, f"Pitch: {norm_gaze[i, 0]:.3f} rad", fontsize=10,
                       transform=axes[1, 2].transAxes)
        y_pos -= 0.08
        axes[1, 2].text(0.1, y_pos, f"Yaw: {norm_gaze[i, 1]:.3f} rad", fontsize=10,
                       transform=axes[1, 2].transAxes)

        # Add predicted gaze info if available
        if gaze_pred is not None:
            y_pos -= 0.12
            axes[1, 2].text(0.1, y_pos, f"Predicted Gaze:", fontsize=12,
                           fontweight='bold', transform=axes[1, 2].transAxes, color='red')
            y_pos -= 0.08
            axes[1, 2].text(0.1, y_pos, f"Pitch: {gaze_pred[i, 0]:.3f} rad", fontsize=10,
                           transform=axes[1, 2].transAxes)
            y_pos -= 0.08
            axes[1, 2].text(0.1, y_pos, f"Yaw: {gaze_pred[i, 1]:.3f} rad", fontsize=10,
                           transform=axes[1, 2].transAxes)
            
            # Calculate and display error
            y_pos -= 0.12
            gaze_target = torch.tensor([[norm_gaze[i, 0], norm_gaze[i, 1]]])
            gaze_p = torch.tensor([[gaze_pred[i, 0], gaze_pred[i, 1]]])
            error = gaze_angular_error(gaze_target, gaze_p).item()
            axes[1, 2].text(0.1, y_pos, f"Angular Error:", fontsize=12,
                           fontweight='bold', transform=axes[1, 2].transAxes, color='orange')
            y_pos -= 0.08
            axes[1, 2].text(0.1, y_pos, f"{error:.2f}°", fontsize=11,
                           fontweight='bold', transform=axes[1, 2].transAxes, color='orange')

        axes[1, 2].set_xlim(0, 1)
        axes[1, 2].set_ylim(0, 1)
        axes[1, 2].axis('off')

        # Use provided sample_idx or default to loop index i
        actual_sample_idx = sample_idx if sample_idx is not None else i

        # Title
        fig.suptitle(
            f"DiC UNet Validation - Step: {global_step}, Batch: {batch_idx}, Sample: {actual_sample_idx}",
            fontsize=16
        )

        # Save
        save_path = os.path.join(save_dir, "validation_samples",
                                f"dic_val_step{global_step}_batch{batch_idx}_sample{actual_sample_idx}.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
