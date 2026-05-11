"""
DualStreamGazeNet — Training Script (Phase 1: Joint Pre-training)

Train all components jointly on multi-user gaze redirection data:
  - AppearanceEncoder: multi-scale identity feature extraction
  - GazeTransformDecoder: gaze-conditioned generation with cross-attention
  - DeltaGazeEmbedder: delta-gaze angular embedding
  - PersonalizationHead: per-user attention bias injection

Key differences from train.py (FAGE EyeOnly):
  1. No separate GazeMLP — delta-gaze embedding is internal to the model
  2. Forward pass takes (source_gaze, target_gaze, head_pose) instead of pre-computed embeddings
  3. Delta-zero consistency loss: when Δgaze=0, output should match input (identity mapping)
  4. Cross-attention between appearance and gaze streams adds ~15% compute

Usage:
    accelerate launch train_dual_stream.py --config configs/training/dual_stream_gaze_net.yaml
"""
import os
import random
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from tqdm.auto import tqdm
from omegaconf import OmegaConf
from datetime import timedelta, datetime
from accelerate.utils import DistributedDataParallelKwargs, LoggerType
from accelerate.logging import get_logger
from accelerate import InitProcessGroupKwargs, Accelerator
import warnings
import torch
import torch.nn.functional as F
import math
import logging
import argparse
import shutil

from models.dual_stream_gaze_net import DualStreamGazeNet
from dataset.gaze_capture import HDFDataset
from dataset.eth_xgaze_paired import ETHXGazePairedDataset
from torch.utils.data import DataLoader
from utils.training_utils import initialize_loss_functions
from loss.basic_loss import discriminator_loss, generator_loss, gaze_angular_loss
from torchvision.utils import make_grid
import lpips
from piq import psnr as piq_psnr
from torchvision import transforms as T
from gaze_estimation.baseline_resnet import GazeHeadResNet

logger = get_logger(__name__, log_level="INFO")
warnings.filterwarnings("ignore")


def seed_everything(seed):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_model(cfg):
    """Create DualStreamGazeNet from config."""
    model_config = OmegaConf.to_container(cfg.dual_stream_params, resolve=True)
    model = DualStreamGazeNet(model_config)
    return model


def build_g_lr_scheduler(optimizer, cfg):
    """Two-phase G scheduler: warmup → flat until D joins → cosine decay."""
    warmup    = cfg.solver.lr_warmup_steps
    start_gan = cfg.discriminator_train_params.start_gan
    max_steps = cfg.solver.max_train_steps

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup:
            return float(current_step) / float(max(1, warmup))
        if current_step < start_gan:
            return 1.0
        progress = float(current_step - start_gan) / float(max(1, max_steps - start_gan))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _get_tb_writer(accelerator):
    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            return tracker.writer
    return None


def _denorm(t):
    """[-1,1] → [0,1]"""
    return (t * 0.5 + 0.5).clamp(0, 1)


def _eye_to_grid(eye_tensor, n=4):
    imgs = _denorm(eye_tensor[:n].detach().float().cpu())
    return make_grid(imgs, nrow=n, padding=2)


def log_vis(accelerator, global_step, source_eye_large, generated_tight,
            target_eye_crops, source_image, target_image, pasted_face, n=4,
            prefix="train"):
    writer = _get_tb_writer(accelerator)
    if writer is None:
        return

    p = prefix.rstrip("/")

    grid_src = _eye_to_grid(source_eye_large, n)
    grid_gen = _eye_to_grid(generated_tight, n)
    grid_gt  = _eye_to_grid(target_eye_crops, n)
    if grid_src.shape != grid_gen.shape:
        grid_src = F.interpolate(
            grid_src.unsqueeze(0), size=grid_gen.shape[1:], mode='bilinear',
            align_corners=False,
        ).squeeze(0)
    eye_rows = torch.cat([grid_src, grid_gen, grid_gt], dim=1)
    writer.add_image(f"{p}/eyes_compare", eye_rows, global_step)

    def face_grid(t):
        imgs = _denorm(t[:n].detach().float().cpu())
        return make_grid(imgs, nrow=n, padding=2)

    face_rows = torch.cat([
        face_grid(source_image),
        face_grid(pasted_face),
        face_grid(target_image),
    ], dim=1)
    writer.add_image(f"{p}/face_compare", face_rows, global_step)


def main(cfg, config_file_path=None):
    exp_name = cfg.exp_name
    save_dir = f"{cfg.output_dir}/{exp_name}"
    os.makedirs(save_dir, exist_ok=True)

    if config_file_path and os.path.exists(config_file_path):
        shutil.copy2(config_file_path, os.path.join(save_dir, "config.yaml"))
    else:
        OmegaConf.save(cfg, os.path.join(save_dir, "config.yaml"))

    kwargs = DistributedDataParallelKwargs()
    process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=5400))
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.solver.gradient_accumulation_steps,
        log_with=["tensorboard", LoggerType.TENSORBOARD],
        mixed_precision=cfg.solver.mixed_precision,
        project_dir=os.path.join(save_dir, "tensorboard"),
        kwargs_handlers=[kwargs, process_group_kwargs],
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    if cfg.seed is not None:
        seed_everything(cfg.seed + accelerator.process_index)

    weight_dtype = torch.float32

    # === Create model ===
    model = create_model(cfg)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"DualStreamGazeNet: {total_params/1e6:.2f}M total, {trainable_params/1e6:.2f}M trainable")

    # Component-level param counts
    for name, module in [
        ("AppearanceEncoder", model.appearance_encoder),
        ("GazeTransformDecoder", model.gaze_decoder),
        ("DeltaGazeEmbedder", model.delta_gaze_embedder),
    ]:
        n = sum(p.numel() for p in module.parameters())
        logger.info(f"  {name}: {n/1e6:.2f}M params")
    if model.personal_head is not None:
        n = sum(p.numel() for p in model.personal_head.parameters())
        logger.info(f"  PersonalizationHead: {n/1e6:.2f}M params (Phase 2 only: {n/1e3:.1f}K)")

    # === Optimizer ===
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.solver.learning_rate,
        betas=(cfg.solver.adam_beta1, cfg.solver.adam_beta2),
        weight_decay=cfg.solver.adam_weight_decay,
        eps=cfg.solver.adam_epsilon,
    )

    lr_scheduler = build_g_lr_scheduler(optimizer, cfg)

    # === Dataset ===
    _ds_kwargs = dict(
        split_ratio=cfg.data.split_ratio,
        seed=cfg.seed,
        frame_offset_range=cfg.data.get('frame_offset_range', 2),
        eye_crop_size=cfg.data.get('eye_crop_size', [64, 64]),
        eye_expand_ratio=cfg.data.get('eye_expand_ratio', 1.5),
        input_eye_crop_size=cfg.data.get('input_eye_crop_size', None),
        input_eye_expand_ratio=cfg.data.get('input_eye_expand_ratio', None),
        max_head_diff=cfg.data.get('max_head_diff', None),
        max_target_retries=cfg.data.get('max_target_retries', 10),
    )

    dataset_type = cfg.data.get('dataset_type', 'gaze_hdf')

    if dataset_type == 'eth_xgaze':
        logger.info("Using ETH-XGaze LMDB dataset")
        train_dataset = ETHXGazePairedDataset(
            cfg.data.lmdb_path, split='train',
            resolution=cfg.data.get('image_size', 256), aug=False,
            **_ds_kwargs
        )
        val_dataset = ETHXGazePairedDataset(
            cfg.data.lmdb_path, split='val',
            resolution=cfg.data.get('image_size', 256), aug=False,
            **_ds_kwargs
        )
    elif dataset_type == 'gaze_hdf':
        logger.info("Using GazeCapture HDF5 dataset")
        train_dataset = HDFDataset(cfg.data.hdf_path, split='train', **_ds_kwargs)
        val_dataset   = HDFDataset(cfg.data.hdf_path, split='val',   **_ds_kwargs)
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    train_dataloader = DataLoader(
        train_dataset, batch_size=cfg.data.train_bs,
        shuffle=True, num_workers=cfg.data.num_workers,
        pin_memory=True, drop_last=True,
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=cfg.data.train_bs,
        shuffle=False, num_workers=cfg.data.num_workers,
        pin_memory=True,
    )

    # === Prepare with accelerator ===
    model, optimizer, lr_scheduler, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, lr_scheduler, train_dataloader, val_dataloader
    )

    # === Loss functions ===
    loss_dict = initialize_loss_functions(cfg, accelerator, cfg.solver.max_train_steps)

    if 'eye_discriminator' in loss_dict:
        loss_dict['eye_discriminator'], loss_dict['eye_optimizer_D'] = accelerator.prepare(
            loss_dict['eye_discriminator'], loss_dict['eye_optimizer_D']
        )

    # === Val metric tools ===
    lpips_fn = lpips.LPIPS(net='alex').to(accelerator.device)
    lpips_fn.eval()
    for p in lpips_fn.parameters():
        p.requires_grad = False

    gaze_estimator = GazeHeadResNet(pretrained=False).to(accelerator.device)
    _ckpt_path = cfg.pretrained.gaze_eval_checkpoint_path
    if os.path.exists(_ckpt_path):
        _ckpt = torch.load(_ckpt_path, map_location=accelerator.device)
        gaze_estimator.load_state_dict(_ckpt['model_state_dict'])
        logger.info(f"Gaze estimator loaded from {_ckpt_path}")
    gaze_estimator.eval()
    for p in gaze_estimator.parameters():
        p.requires_grad = False

    _gaze_eval_trans = T.Compose([
        T.Resize((224, 224)),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # === Training loop ===
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / cfg.solver.gradient_accumulation_steps
    )
    num_train_epochs = math.ceil(cfg.solver.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        run_time = datetime.now().strftime("%Y%m%d-%H%M")
        accelerator.init_trackers(cfg.exp_name, init_kwargs={"mlflow": {"run_name": run_time}})

    logger.info("***** DualStreamGazeNet Training *****")
    logger.info(f"Epochs: {num_train_epochs}, Max steps: {cfg.solver.max_train_steps}")
    logger.info(f"Batch size: {cfg.data.train_bs}, Train samples: {len(train_dataset)}")

    global_step = 0
    start_epoch = 0
    steps_to_skip = 0

    # === Checkpoint resume ===
    resume_path = cfg.get('resume_from_checkpoint', False)
    if resume_path and isinstance(resume_path, str) and os.path.exists(resume_path):
        logger.info(f"Resuming from checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location='cpu')
        global_step = ckpt.get('global_step', 0)
        resume_strict = cfg.get('resume_strict', True)
        incompatible = accelerator.unwrap_model(model).load_state_dict(
            ckpt['model_state_dict'], strict=resume_strict)
        if not resume_strict and (incompatible.missing_keys or incompatible.unexpected_keys):
            logger.warning(f"  load_state_dict (strict=False): "
                           f"missing={incompatible.missing_keys}, "
                           f"unexpected={incompatible.unexpected_keys}")
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'lr_scheduler_state_dict' in ckpt:
            lr_scheduler.load_state_dict(ckpt['lr_scheduler_state_dict'])
        if 'discriminator_state_dict' in ckpt and 'eye_discriminator' in loss_dict:
            accelerator.unwrap_model(loss_dict['eye_discriminator']).load_state_dict(
                ckpt['discriminator_state_dict'])
        if 'optimizer_D_state_dict' in ckpt and 'eye_optimizer_D' in loss_dict:
            loss_dict['eye_optimizer_D'].load_state_dict(ckpt['optimizer_D_state_dict'])
        if 'scheduler_D_state_dict' in ckpt and 'eye_scheduler_D' in loss_dict:
            loss_dict['eye_scheduler_D'].load_state_dict(ckpt['scheduler_D_state_dict'])
        start_epoch = global_step // num_update_steps_per_epoch
        steps_to_skip = global_step % num_update_steps_per_epoch
        logger.info(f"Resumed at step={global_step} (epoch={start_epoch}, skip {steps_to_skip})")

    best_val_score = float('inf')
    progress_bar = tqdm(range(cfg.solver.max_train_steps),
                        initial=global_step,
                        disable=not accelerator.is_local_main_process)

    for epoch in range(start_epoch, num_train_epochs):
        model.train()

        for step, batch in enumerate(train_dataloader):
            if epoch == start_epoch and step < steps_to_skip:
                continue
            if global_step >= cfg.solver.max_train_steps:
                break

            with accelerator.accumulate(model):
                # === 1. Extract paired frame data ===
                source_eye_crops = batch['source_input_eye_crops'].to(weight_dtype)
                target_eye_crops = batch['target_eye_crops'].to(weight_dtype)
                target_gaze = batch['target_gaze'].to(weight_dtype)
                target_head = batch['target_head'].to(weight_dtype)
                source_gaze = batch['source_gaze'].to(weight_dtype)
                source_head = batch['source_head'].to(weight_dtype)
                source_image = batch['source_image'].to(weight_dtype)
                target_image = batch['target_image'].to(weight_dtype)
                source_eye_bbox = batch['source_eye_bbox'].to(weight_dtype)
                target_eye_bbox = batch['target_eye_bbox'].to(weight_dtype)

                # === 2. Forward: DualStreamGazeNet ===
                # Model takes raw angles, not pre-computed embeddings
                generated_eyes = accelerator.unwrap_model(model)(
                    source_eye_crops,
                    source_gaze=source_gaze,
                    target_gaze=target_gaze,
                    head_pose=target_head,
                    source_face=source_image,
                )

                # Center-crop generated to supervision size
                in_h, in_w   = source_eye_crops.shape[-2:]
                tgt_h, tgt_w = target_eye_crops.shape[-2:]
                if in_h != tgt_h or in_w != tgt_w:
                    pad_h = (in_h - tgt_h) // 2
                    pad_w = (in_w - tgt_w) // 2
                    generated_tight = generated_eyes[:, :, pad_h:pad_h+tgt_h, pad_w:pad_w+tgt_w]
                else:
                    generated_tight = generated_eyes

                # === 3. Loss computation ===
                l1_loss = F.l1_loss(generated_tight, target_eye_crops)
                loss = cfg.loss_params.l1_loss * l1_loss

                # Delta-zero consistency loss: Δgaze=0 should produce identity mapping
                start_delta_zero = cfg.loss_params.get('start_delta_zero', 500)
                delta_zero_weight = cfg.loss_params.get('delta_zero_loss', 0)
                delta_zero_loss_val = None

                if delta_zero_weight > 0 and global_step >= start_delta_zero:
                    # Forward with source_gaze == target_gaze → should reconstruct input
                    with torch.no_grad():
                        # Detach to avoid double backward through appearance encoder
                        identity_output = accelerator.unwrap_model(model)(
                            source_eye_crops,
                            source_gaze=source_gaze,
                            target_gaze=source_gaze,  # same as source → Δ=0
                            head_pose=source_head,
                            source_face=source_image,
                        )
                    # Only use this as a soft regularizer (no grad through identity_output)
                    # Compare against source eye crops (not target)
                    # Crop to tight size for consistency
                    if in_h != tgt_h or in_w != tgt_w:
                        identity_tight = identity_output[:, :, pad_h:pad_h+tgt_h, pad_w:pad_w+tgt_w]
                        source_tight = source_eye_crops[:, :, pad_h:pad_h+tgt_h, pad_w:pad_w+tgt_w]
                    else:
                        identity_tight = identity_output
                        source_tight = source_eye_crops
                    delta_zero_loss_val = F.l1_loss(identity_tight, source_tight)
                    loss = loss + delta_zero_weight * delta_zero_loss_val

                # Gaze perceptual loss (paste back then compare)
                start_perc = cfg.loss_params.get('start_gaze_perceptual', 500)
                start_id   = cfg.loss_params.get('start_id_loss', 1000)
                gaze_perc_loss = None
                id_loss_val    = None

                need_pasted = (
                    (cfg.loss_params.gaze_perceptual_loss > 0 and global_step >= start_perc) or
                    (cfg.loss_params.get('id_loss', 0) > 0 and global_step >= start_id)
                )
                pasted = None
                if need_pasted:
                    pasted = accelerator.unwrap_model(model).paste_eyes(
                        generated_tight, target_image, target_eye_bbox)

                if cfg.loss_params.gaze_perceptual_loss > 0 and global_step >= start_perc:
                    gaze_perc_loss = loss_dict['angular_loss'](pasted, target_image)
                    loss = loss + cfg.loss_params.gaze_perceptual_loss * gaze_perc_loss

                if cfg.loss_params.get('id_loss', 0) > 0 and global_step >= start_id:
                    id_loss_val = loss_dict['id_loss'].loss(pasted, source_image)
                    loss = loss + cfg.loss_params.id_loss * id_loss_val

                # GAN: Generator step
                start_gan = cfg.discriminator_train_params.start_gan
                g_adv = None
                if cfg.loss_params.eye_gan_loss > 0 and global_step >= start_gan:
                    for p in loss_dict['eye_discriminator'].parameters():
                        p.requires_grad_(False)
                    fake_pred = loss_dict['eye_discriminator'](generated_tight)
                    g_adv = generator_loss(fake_pred, fake_pred.device)
                    loss = loss + cfg.loss_params.eye_gan_loss * g_adv
                    for p in loss_dict['eye_discriminator'].parameters():
                        p.requires_grad_(True)

                # === 4. Backward ===
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    trainable = [p for p in model.parameters() if p.requires_grad]
                    accelerator.clip_grad_norm_(trainable, cfg.solver.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # === Discriminator step ===
            if cfg.loss_params.eye_gan_loss > 0 and global_step >= start_gan:
                with accelerator.accumulate(loss_dict['eye_discriminator']):
                    loss_dict['eye_optimizer_D'].zero_grad()
                    B = target_eye_crops.shape[0]
                    combined = torch.cat([target_eye_crops.detach(), generated_tight.detach()], dim=0)
                    combined_pred = loss_dict['eye_discriminator'](combined)
                    real_pred = combined_pred[:B]
                    fake_pred_d = combined_pred[B:]
                    d_loss = discriminator_loss(real_pred, fake_pred_d, real_pred.device)
                    accelerator.backward(d_loss)
                    loss_dict['eye_optimizer_D'].step()
                    if 'eye_scheduler_D' in loss_dict and accelerator.sync_gradients:
                        loss_dict['eye_scheduler_D'].step()

            # === 5. Logging ===
            global_step += 1
            progress_bar.update(1)

            if global_step % 100 == 0 and accelerator.is_main_process:
                log_dict = {
                    "train/l1_loss": l1_loss.item(),
                    "train/total_loss": loss.item(),
                    "train/lr": lr_scheduler.get_last_lr()[0],
                }
                if gaze_perc_loss is not None:
                    log_dict["train/gaze_perc_loss"] = gaze_perc_loss.item()
                if id_loss_val is not None:
                    log_dict["train/id_loss"] = id_loss_val.item()
                if delta_zero_loss_val is not None:
                    log_dict["train/delta_zero_loss"] = delta_zero_loss_val.item()
                if g_adv is not None:
                    log_dict["train/g_adv_loss"] = g_adv.item()
                    log_dict["train/d_loss"] = d_loss.item()
                accelerator.log(log_dict, step=global_step)

            if global_step % 500 == 0:
                logger.info(f"Step {global_step}: l1={l1_loss.item():.4f}, total={loss.item():.4f}")

            # === 6. Validation ===
            if global_step % cfg.val_freq == 0 and accelerator.is_main_process:
                model.eval()
                val_l1 = 0
                val_lpips = 0
                val_psnr = 0
                val_gaze_error = 0
                val_count = 0
                val_vis_data = None

                with torch.no_grad():
                    for vi, vbatch in enumerate(val_dataloader):
                        if vi >= cfg.val_batch_limit:
                            break
                        src = vbatch['source_input_eye_crops'].to(weight_dtype)
                        tgt = vbatch['target_eye_crops'].to(weight_dtype)
                        tgt_gaze = vbatch['target_gaze'].to(weight_dtype)
                        tgt_head = vbatch['target_head'].to(weight_dtype)
                        src_gaze = vbatch['source_gaze'].to(weight_dtype)
                        src_face = vbatch['source_image'].to(weight_dtype)

                        gen = accelerator.unwrap_model(model)(
                            src,
                            source_gaze=src_gaze,
                            target_gaze=tgt_gaze,
                            head_pose=tgt_head,
                            source_face=src_face,
                        )

                        si_h, si_w = src.shape[-2:]
                        tg_h, tg_w = tgt.shape[-2:]
                        if si_h != tg_h or si_w != tg_w:
                            ph = (si_h - tg_h) // 2
                            pw = (si_w - tg_w) // 2
                            gen = gen[:, :, ph:ph+tg_h, pw:pw+tg_w]

                        val_l1 += F.l1_loss(gen, tgt).item()
                        val_lpips += lpips_fn(gen, tgt).mean().item()
                        val_psnr += piq_psnr(
                            (gen.float() + 1).clamp(0, 2),
                            (tgt.float() + 1).clamp(0, 2),
                            data_range=2.0,
                        ).item()

                        gen_left = (gen[:, :, :, :gen.shape[3]//2].float() + 1) / 2
                        gen_left = _gaze_eval_trans(gen_left)
                        gaze_pred, _ = gaze_estimator(gen_left)
                        val_gaze_error += gaze_angular_loss(tgt_gaze.float(), gaze_pred).item()

                        val_count += 1
                        if random.randint(0, val_count - 1) == 0:
                            val_vis_data = {
                                'src': src, 'gen': gen, 'tgt': tgt,
                                'src_img': vbatch['source_image'].to(weight_dtype),
                                'tgt_img': vbatch['target_image'].to(weight_dtype),
                                'bbox':    vbatch['target_eye_bbox'].to(weight_dtype),
                            }

                if val_count > 0:
                    avg_l1    = val_l1 / val_count
                    avg_lpips = val_lpips / val_count
                    avg_psnr  = val_psnr / val_count
                    avg_gaze  = val_gaze_error / val_count

                    accelerator.log({
                        "val/l1_loss":    avg_l1,
                        "val/lpips":      avg_lpips,
                        "val/psnr":       avg_psnr,
                        "val/gaze_error": avg_gaze,
                    }, step=global_step)
                    logger.info(
                        f"Val step {global_step}: "
                        f"l1={avg_l1:.4f}  lpips={avg_lpips:.4f}  "
                        f"psnr={avg_psnr:.2f}dB  gaze={avg_gaze:.4f}deg"
                    )

                    score = avg_l1 / 0.05 + avg_lpips / 0.1 + 1.0 / max(avg_psnr, 1.0) / 0.03 + avg_gaze / 5.0
                    accelerator.log({"val/composite_score": score}, step=global_step)

                    if score < best_val_score:
                        best_val_score = score
                        best_path = os.path.join(save_dir, "best_model.pth")
                        unwrapped_best = accelerator.unwrap_model(model)
                        torch.save({
                            'global_step': global_step,
                            'best_score': best_val_score,
                            'metrics': {
                                'l1': avg_l1, 'lpips': avg_lpips,
                                'psnr': avg_psnr, 'gaze_error': avg_gaze,
                            },
                            'model_state_dict': unwrapped_best.state_dict(),
                        }, best_path)
                        logger.info(
                            f"New best model (score={score:.4f}): "
                            f"l1={avg_l1:.4f} lpips={avg_lpips:.4f} "
                            f"psnr={avg_psnr:.2f} gaze={avg_gaze:.4f}"
                        )

                # TensorBoard visualization (train batch)
                with torch.no_grad():
                    vis_pasted = accelerator.unwrap_model(model).paste_eyes(
                        generated_tight, target_image, target_eye_bbox)
                log_vis(
                    accelerator, global_step,
                    source_eye_large=source_eye_crops,
                    generated_tight=generated_tight,
                    target_eye_crops=target_eye_crops,
                    source_image=source_image,
                    target_image=target_image,
                    pasted_face=vis_pasted,
                    prefix="train",
                )

                if val_vis_data is not None:
                    with torch.no_grad():
                        val_pasted = accelerator.unwrap_model(model).paste_eyes(
                            val_vis_data['gen'], val_vis_data['tgt_img'], val_vis_data['bbox'])
                    log_vis(
                        accelerator, global_step,
                        source_eye_large=val_vis_data['src'],
                        generated_tight=val_vis_data['gen'],
                        target_eye_crops=val_vis_data['tgt'],
                        source_image=val_vis_data['src_img'],
                        target_image=val_vis_data['tgt_img'],
                        pasted_face=val_pasted,
                        prefix="val",
                    )
                model.train()

            # === 7. Save checkpoint ===
            if global_step % cfg.checkpointing_steps == 0 and accelerator.is_main_process:
                ckpt_path = os.path.join(save_dir, f"checkpoint-{global_step}.pth")
                unwrapped = accelerator.unwrap_model(model)
                save_dict = {
                    'global_step': global_step,
                    'model_state_dict': unwrapped.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                }
                if 'eye_discriminator' in loss_dict:
                    save_dict['discriminator_state_dict'] = (
                        accelerator.unwrap_model(loss_dict['eye_discriminator']).state_dict())
                    save_dict['optimizer_D_state_dict'] = (
                        loss_dict['eye_optimizer_D'].state_dict())
                    if 'eye_scheduler_D' in loss_dict:
                        save_dict['scheduler_D_state_dict'] = (
                            loss_dict['eye_scheduler_D'].state_dict())

                torch.save(save_dict, ckpt_path)
                logger.info(f"Saved checkpoint: {ckpt_path}")

                ckpts = sorted(
                    [f for f in os.listdir(save_dir) if f.startswith("checkpoint-") and f.endswith(".pth")],
                    key=lambda x: int(x.split("-")[1].split(".")[0])
                )
                while len(ckpts) > cfg.total_limit:
                    old = os.path.join(save_dir, ckpts.pop(0))
                    os.remove(old)

    # Final save
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        final_path = os.path.join(save_dir, "final_model.pth")
        torch.save({
            'model_state_dict': unwrapped.state_dict(),
        }, final_path)
        logger.info(f"Training complete. Final model: {final_path}")

    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    main(cfg, args.config)
