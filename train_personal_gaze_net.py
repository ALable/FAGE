"""
PersonalGazeNet — Training Script
Phase 1: Shared Pre-training (all parameters, self-supervised with N_ref=1)
Phase 2: Identity Fine-tuning (only IdentityEncoder + IdentityAggregator, N_ref>=2)

Launch:
    accelerate launch train_personal_gaze_net.py \
        --config configs/training/personal_gaze_net.yaml

Key differences from train.py (EyeOnlyWrapper):
  - Model: PersonalGazeWrapper (spatial identity tokens + gaze-delta warping)
  - Forward: passes source_gaze_raw / target_gaze_raw (for warp module)
              and ref_eyes = source_eye_crops (self-supervised Phase 1)
  - Loss: adds optional flow smoothness regularisation
  - Saving: model key is 'net' (not 'eye_unet')
"""
import os
import math
import random
import shutil
import logging
import argparse
import warnings

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from omegaconf import OmegaConf
from datetime import timedelta, datetime

from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import DistributedDataParallelKwargs, LoggerType
from accelerate.logging import get_logger
from torchvision.utils import make_grid
from torchvision import transforms as T
import lpips
from piq import psnr as piq_psnr
from torch.utils.data import DataLoader

from models.personal_gaze_net import PersonalGazeWrapper
from models.gazenet import MLPNetwork
from dataset.gaze_capture import HDFDataset
from dataset.eth_xgaze_paired import ETHXGazePairedDataset
from utils.training_utils import initialize_loss_functions
from loss.basic_loss import discriminator_loss, generator_loss, gaze_angular_loss
from gaze_estimation.baseline_resnet import GazeHeadResNet

logger = get_logger(__name__, log_level="INFO")
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def seed_everything(seed: int):
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _denorm(t: torch.Tensor) -> torch.Tensor:
    """[-1, 1] → [0, 1], safely clamped."""
    return (t * 0.5 + 0.5).clamp(0, 1)


def _eye_to_grid(eye_tensor: torch.Tensor, n: int = 4) -> torch.Tensor:
    imgs = _denorm(eye_tensor[:n].detach().float().cpu())
    return make_grid(imgs, nrow=n, padding=2)


def _get_tb_writer(accelerator):
    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            return tracker.writer
    return None


def log_vis(accelerator, global_step, source_eye_large, generated_tight,
            target_eye_crops, source_image, target_image, pasted_face,
            n: int = 4, prefix: str = "train"):
    """Write comparison images to TensorBoard."""
    writer = _get_tb_writer(accelerator)
    if writer is None:
        return
    p = prefix.rstrip("/")

    grid_src = _eye_to_grid(source_eye_large, n)
    grid_gen = _eye_to_grid(generated_tight,  n)
    grid_gt  = _eye_to_grid(target_eye_crops,  n)
    if grid_src.shape != grid_gen.shape:
        grid_src = F.interpolate(
            grid_src.unsqueeze(0), size=grid_gen.shape[1:],
            mode='bilinear', align_corners=False,
        ).squeeze(0)
    writer.add_image(f"{p}/eyes_compare",
                     torch.cat([grid_src, grid_gen, grid_gt], dim=1),
                     global_step)

    def face_grid(t):
        imgs = _denorm(t[:n].detach().float().cpu())
        return make_grid(imgs, nrow=n, padding=2)

    writer.add_image(f"{p}/face_compare",
                     torch.cat([face_grid(source_image),
                                face_grid(pasted_face),
                                face_grid(target_image)], dim=1),
                     global_step)


def flow_smoothness_loss(flow: torch.Tensor) -> torch.Tensor:
    """Total-variation smoothness on the predicted flow field.

    Penalises spatially abrupt changes in the displacement field, encouraging
    the warp module to produce smooth, physically-plausible deformations.

    Args:
        flow: [B, 2, H, W]  predicted displacement field
    Returns:
        scalar loss
    """
    dy = (flow[:, :, 1:, :] - flow[:, :, :-1, :]).abs().mean()
    dx = (flow[:, :, :, 1:] - flow[:, :, :, :-1]).abs().mean()
    return dx + dy


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------

def create_model(cfg) -> PersonalGazeWrapper:
    pgn_config = OmegaConf.to_container(cfg.pgn_params, resolve=True)
    return PersonalGazeWrapper(pgn_config)


def create_gaze_mlp(cfg) -> MLPNetwork:
    gp = cfg.model_params.gazenet_params
    gaze_dim = cfg.pgn_params.get('gaze_dim', 64)
    return MLPNetwork(
        num_in=gp.num_in,
        num_hidden=gp.num_hidden,
        num_out=gaze_dim,
        num_layers=gp.num_layers,
        cross_condition=gp.get('cross_condition', False),
    )


def build_g_lr_scheduler(optimizer, cfg):
    """Warmup → flat (G-only) → cosine decay (after D joins)."""
    warmup    = cfg.solver.lr_warmup_steps
    start_gan = cfg.discriminator_train_params.start_gan
    max_steps = cfg.solver.max_train_steps

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return float(step) / float(max(1, warmup))
        if step < start_gan:
            return 1.0
        progress = float(step - start_gan) / float(max(1, max_steps - start_gan))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main(cfg, config_file_path=None):
    exp_name = cfg.exp_name
    save_dir = f"{cfg.output_dir}/{exp_name}"
    os.makedirs(save_dir, exist_ok=True)

    if config_file_path and os.path.exists(config_file_path):
        shutil.copy2(config_file_path, os.path.join(save_dir, "config.yaml"))
    else:
        OmegaConf.save(cfg, os.path.join(save_dir, "config.yaml"))

    # ── Accelerator ──────────────────────────────────────────────────────────
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.solver.gradient_accumulation_steps,
        log_with=["tensorboard", LoggerType.TENSORBOARD],
        mixed_precision=cfg.solver.mixed_precision,
        project_dir=os.path.join(save_dir, "tensorboard"),
        kwargs_handlers=[
            DistributedDataParallelKwargs(),
            InitProcessGroupKwargs(timeout=timedelta(seconds=5400)),
        ],
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    if cfg.seed is not None:
        seed_everything(cfg.seed + accelerator.process_index)

    weight_dtype = torch.float32

    # ── Models ───────────────────────────────────────────────────────────────
    model    = create_model(cfg)
    gaze_mlp = create_gaze_mlp(cfg)

    total_p     = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"PersonalGazeNet: {total_p/1e6:.2f}M total, "
                f"{trainable_p/1e6:.2f}M trainable")

    # ── Optimizer ────────────────────────────────────────────────────────────
    opt_params = list(model.parameters()) + list(gaze_mlp.parameters())
    optimizer  = torch.optim.AdamW(
        opt_params,
        lr=cfg.solver.learning_rate,
        betas=(cfg.solver.adam_beta1, cfg.solver.adam_beta2),
        weight_decay=cfg.solver.adam_weight_decay,
        eps=cfg.solver.adam_epsilon,
    )
    lr_scheduler = build_g_lr_scheduler(optimizer, cfg)

    # ── Dataset ──────────────────────────────────────────────────────────────
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
            resolution=cfg.data.get('image_size', 256), aug=False, **_ds_kwargs,
        )
        val_dataset = ETHXGazePairedDataset(
            cfg.data.lmdb_path, split='val',
            resolution=cfg.data.get('image_size', 256), aug=False, **_ds_kwargs,
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

    # ── Prepare with accelerator ─────────────────────────────────────────────
    (model, gaze_mlp, optimizer, lr_scheduler,
     train_dataloader, val_dataloader) = accelerator.prepare(
        model, gaze_mlp, optimizer, lr_scheduler,
        train_dataloader, val_dataloader,
    )

    # ── Loss functions (discriminator + perceptual) ──────────────────────────
    loss_dict = initialize_loss_functions(cfg, accelerator, cfg.solver.max_train_steps)
    if 'eye_discriminator' in loss_dict:
        loss_dict['eye_discriminator'], loss_dict['eye_optimizer_D'] = accelerator.prepare(
            loss_dict['eye_discriminator'], loss_dict['eye_optimizer_D'],
        )

    # ── Validation tools ─────────────────────────────────────────────────────
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

    # ── Training setup ───────────────────────────────────────────────────────
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / cfg.solver.gradient_accumulation_steps
    )
    num_train_epochs = math.ceil(
        cfg.solver.max_train_steps / num_update_steps_per_epoch
    )

    if accelerator.is_main_process:
        run_time = datetime.now().strftime("%Y%m%d-%H%M")
        accelerator.init_trackers(
            cfg.exp_name,
            init_kwargs={"mlflow": {"run_name": run_time}},
        )

    logger.info("***** PersonalGazeNet Training *****")
    logger.info(f"Epochs: {num_train_epochs}, Max steps: {cfg.solver.max_train_steps}")
    logger.info(f"Batch size: {cfg.data.train_bs}, Train samples: {len(train_dataset)}")

    global_step  = 0
    start_epoch  = 0
    steps_to_skip = 0

    # ── Resume from checkpoint ────────────────────────────────────────────────
    resume_path = cfg.get('resume_from_checkpoint', False)
    if resume_path and isinstance(resume_path, str) and os.path.exists(resume_path):
        logger.info(f"Resuming from checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location='cpu')
        global_step   = ckpt.get('global_step', 0)
        resume_strict = cfg.get('resume_strict', True)
        accelerator.unwrap_model(model).load_state_dict(
            ckpt['model_state_dict'], strict=resume_strict)
        accelerator.unwrap_model(gaze_mlp).load_state_dict(
            ckpt['gaze_mlp_state_dict'], strict=resume_strict)
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'lr_scheduler_state_dict' in ckpt:
            lr_scheduler.load_state_dict(ckpt['lr_scheduler_state_dict'])
        if 'discriminator_state_dict' in ckpt and 'eye_discriminator' in loss_dict:
            accelerator.unwrap_model(loss_dict['eye_discriminator']).load_state_dict(
                ckpt['discriminator_state_dict'])
        if 'optimizer_D_state_dict' in ckpt and 'eye_optimizer_D' in loss_dict:
            loss_dict['eye_optimizer_D'].load_state_dict(ckpt['optimizer_D_state_dict'])
        start_epoch   = global_step // num_update_steps_per_epoch
        steps_to_skip = global_step % num_update_steps_per_epoch
        logger.info(f"Resumed at step={global_step} (epoch={start_epoch})")

    best_val_score = float('inf')
    progress_bar   = tqdm(
        range(cfg.solver.max_train_steps),
        initial=global_step,
        disable=not accelerator.is_local_main_process,
    )

    # ── Loss hyper-params shorthand ──────────────────────────────────────────
    lp      = cfg.loss_params
    start_gan  = cfg.discriminator_train_params.start_gan
    start_perc = lp.get('start_gaze_perceptual', 500)
    start_id   = lp.get('start_id_loss', 1000)
    start_flow = lp.get('start_flow_smooth', 0)

    # ── Main loop ────────────────────────────────────────────────────────────
    for epoch in range(start_epoch, num_train_epochs):
        model.train()
        gaze_mlp.train()

        for step, batch in enumerate(train_dataloader):
            if epoch == start_epoch and step < steps_to_skip:
                continue
            if global_step >= cfg.solver.max_train_steps:
                break

            with accelerator.accumulate(model, gaze_mlp):
                # ── 1. Unpack batch ─────────────────────────────────────────
                source_eye_crops = batch['source_input_eye_crops'].to(weight_dtype)
                target_eye_crops = batch['target_eye_crops'].to(weight_dtype)
                target_gaze      = batch['target_gaze'].to(weight_dtype)
                target_head      = batch['target_head'].to(weight_dtype)
                source_gaze_raw  = batch['source_gaze'].to(weight_dtype)   # [B, 2] raw radians
                target_gaze_raw  = batch['target_gaze'].to(weight_dtype)   # [B, 2] raw radians
                source_image     = batch['source_image'].to(weight_dtype)
                target_image     = batch['target_image'].to(weight_dtype)
                source_eye_bbox  = batch['source_eye_bbox'].to(weight_dtype)
                target_eye_bbox  = batch['target_eye_bbox'].to(weight_dtype)

                # ── 2. Gaze embedding via MLPNetwork ────────────────────────
                head_emb, gaze_emb = gaze_mlp(target_head, target_gaze)
                gaze_prompt = torch.cat(
                    [head_emb.unsqueeze(1), gaze_emb.unsqueeze(1)], dim=1
                )  # [B, 2, gaze_dim]

                # ── 3. Forward (Phase 1: ref = source, N_ref = 1) ───────────
                # ref_eyes: [B, 1, 3, H, W] — self-supervised reference
                ref_eyes = source_eye_crops.unsqueeze(1)

                generated_eyes, flow = model(
                    source_eye_crops=source_eye_crops,
                    encoder_hidden_states=gaze_prompt,
                    ref_eyes=ref_eyes,
                    source_gaze_raw=source_gaze_raw,
                    target_gaze_raw=target_gaze_raw,
                )

                # ── 4. Center-crop to supervision resolution ────────────────
                in_h, in_w   = source_eye_crops.shape[-2:]
                tgt_h, tgt_w = target_eye_crops.shape[-2:]
                if in_h != tgt_h or in_w != tgt_w:
                    ph = (in_h - tgt_h) // 2
                    pw = (in_w - tgt_w) // 2
                    generated_tight = generated_eyes[
                        :, :, ph:ph+tgt_h, pw:pw+tgt_w
                    ]
                else:
                    generated_tight = generated_eyes

                # ── 5. Losses ───────────────────────────────────────────────

                # Pixel L1
                l1_loss = F.l1_loss(generated_tight, target_eye_crops)
                loss    = lp.l1_loss * l1_loss

                # Flow smoothness regularisation (from GazeDeltaWarpModule)
                flow_smooth_val = None
                if (flow is not None and
                        lp.get('flow_smooth_loss', 0) > 0 and
                        global_step >= start_flow):
                    flow_smooth_val = flow_smoothness_loss(flow)
                    loss = loss + lp.flow_smooth_loss * flow_smooth_val

                # Perceptual losses requiring pasted face
                need_pasted = (
                    (lp.gaze_perceptual_loss > 0 and global_step >= start_perc) or
                    (lp.get('id_loss', 0)    > 0 and global_step >= start_id)
                )
                pasted           = None
                gaze_perc_loss   = None
                id_loss_val      = None

                if need_pasted:
                    pasted = accelerator.unwrap_model(model).paste_eyes(
                        generated_tight, target_image, target_eye_bbox,
                    )

                if lp.gaze_perceptual_loss > 0 and global_step >= start_perc:
                    gaze_perc_loss = loss_dict['angular_loss'](pasted, target_image)
                    loss = loss + lp.gaze_perceptual_loss * gaze_perc_loss

                if lp.get('id_loss', 0) > 0 and global_step >= start_id:
                    id_loss_val = loss_dict['id_loss'].loss(pasted, source_image)
                    loss = loss + lp.id_loss * id_loss_val

                # Eye GAN: generator step
                g_adv = None
                if lp.eye_gan_loss > 0 and global_step >= start_gan:
                    for p in loss_dict['eye_discriminator'].parameters():
                        p.requires_grad_(False)
                    fake_pred = loss_dict['eye_discriminator'](generated_tight)
                    g_adv = generator_loss(fake_pred, fake_pred.device)
                    loss = loss + lp.eye_gan_loss * g_adv
                    for p in loss_dict['eye_discriminator'].parameters():
                        p.requires_grad_(True)

                # ── 6. Backward ─────────────────────────────────────────────
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    all_trainable = (
                        [p for p in model.parameters()    if p.requires_grad] +
                        [p for p in gaze_mlp.parameters() if p.requires_grad]
                    )
                    accelerator.clip_grad_norm_(all_trainable, cfg.solver.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # ── 6.5 Discriminator step ────────────────────────────────────────
            if lp.eye_gan_loss > 0 and global_step >= start_gan:
                with accelerator.accumulate(loss_dict['eye_discriminator']):
                    loss_dict['eye_optimizer_D'].zero_grad()
                    B_local  = target_eye_crops.shape[0]
                    combined = torch.cat(
                        [target_eye_crops.detach(), generated_tight.detach()], dim=0
                    )
                    combined_pred = loss_dict['eye_discriminator'](combined)
                    real_pred = combined_pred[:B_local]
                    fake_pred_d = combined_pred[B_local:]
                    d_loss = discriminator_loss(real_pred, fake_pred_d, real_pred.device)
                    accelerator.backward(d_loss)
                    loss_dict['eye_optimizer_D'].step()
                    if 'eye_scheduler_D' in loss_dict and accelerator.sync_gradients:
                        loss_dict['eye_scheduler_D'].step()

            # ── 7. Logging ───────────────────────────────────────────────────
            global_step += 1
            progress_bar.update(1)

            if global_step % 100 == 0 and accelerator.is_main_process:
                log_dict = {
                    "train/l1_loss":    l1_loss.item(),
                    "train/total_loss": loss.item(),
                    "train/lr":         lr_scheduler.get_last_lr()[0],
                }
                if flow_smooth_val is not None:
                    log_dict["train/flow_smooth"] = flow_smooth_val.item()
                if gaze_perc_loss is not None:
                    log_dict["train/gaze_perc_loss"] = gaze_perc_loss.item()
                if id_loss_val is not None:
                    log_dict["train/id_loss"] = id_loss_val.item()
                if g_adv is not None:
                    log_dict["train/g_adv_loss"] = g_adv.item()
                    log_dict["train/d_loss"]     = d_loss.item()
                accelerator.log(log_dict, step=global_step)

            if global_step % 500 == 0:
                logger.info(
                    f"Step {global_step}: "
                    f"l1={l1_loss.item():.4f}  total={loss.item():.4f}"
                )

            # ── 8. Validation ────────────────────────────────────────────────
            if global_step % cfg.val_freq == 0 and accelerator.is_main_process:
                model.eval()
                gaze_mlp.eval()

                val_l1 = val_lpips = val_psnr = val_gaze_error = 0.0
                val_count   = 0
                val_vis_data = None

                with torch.no_grad():
                    for vi, vbatch in enumerate(val_dataloader):
                        if vi >= cfg.val_batch_limit:
                            break

                        src          = vbatch['source_input_eye_crops'].to(weight_dtype)
                        tgt          = vbatch['target_eye_crops'].to(weight_dtype)
                        tgt_gaze     = vbatch['target_gaze'].to(weight_dtype)
                        tgt_head     = vbatch['target_head'].to(weight_dtype)
                        src_gaze_raw = vbatch['source_gaze'].to(weight_dtype)
                        tgt_gaze_raw = vbatch['target_gaze'].to(weight_dtype)

                        h_emb, g_emb = gaze_mlp(tgt_head, tgt_gaze)
                        gp = torch.cat([h_emb.unsqueeze(1), g_emb.unsqueeze(1)], dim=1)

                        ref_val = src.unsqueeze(1)  # self-supervised
                        gen, _ = model(
                            source_eye_crops=src,
                            encoder_hidden_states=gp,
                            ref_eyes=ref_val,
                            source_gaze_raw=src_gaze_raw,
                            target_gaze_raw=tgt_gaze_raw,
                        )

                        # Center-crop to supervision size
                        si_h, si_w = src.shape[-2:]
                        tg_h, tg_w = tgt.shape[-2:]
                        if si_h != tg_h or si_w != tg_w:
                            ph = (si_h - tg_h) // 2
                            pw = (si_w - tg_w) // 2
                            gen = gen[:, :, ph:ph+tg_h, pw:pw+tg_w]

                        val_l1    += F.l1_loss(gen, tgt).item()
                        val_lpips += lpips_fn(gen, tgt).mean().item()
                        val_psnr  += piq_psnr(
                            (gen.float() + 1).clamp(0, 2),
                            (tgt.float() + 1).clamp(0, 2),
                            data_range=2.0,
                        ).item()

                        gen_left = (gen[:, :, :, :gen.shape[3]//2].float() + 1) / 2
                        gen_left = _gaze_eval_trans(gen_left)
                        gaze_pred, _ = gaze_estimator(gen_left)
                        val_gaze_error += gaze_angular_loss(
                            tgt_gaze.float(), gaze_pred
                        ).item()

                        val_count += 1
                        if random.randint(0, val_count - 1) == 0:
                            val_vis_data = {
                                'src':     src,
                                'gen':     gen,
                                'tgt':     tgt,
                                'src_img': vbatch['source_image'].to(weight_dtype),
                                'tgt_img': vbatch['target_image'].to(weight_dtype),
                                'bbox':    vbatch['target_eye_bbox'].to(weight_dtype),
                            }

                if val_count > 0:
                    avg_l1    = val_l1         / val_count
                    avg_lpips = val_lpips      / val_count
                    avg_psnr  = val_psnr       / val_count
                    avg_gaze  = val_gaze_error / val_count

                    accelerator.log({
                        "val/l1_loss":    avg_l1,
                        "val/lpips":      avg_lpips,
                        "val/psnr":       avg_psnr,
                        "val/gaze_error": avg_gaze,
                    }, step=global_step)
                    logger.info(
                        f"Val {global_step}: "
                        f"l1={avg_l1:.4f}  lpips={avg_lpips:.4f}  "
                        f"psnr={avg_psnr:.2f}dB  gaze={avg_gaze:.4f}"
                    )

                    # Composite score (lower = better)
                    score = (avg_l1 / 0.05 + avg_lpips / 0.1 +
                             1.0 / max(avg_psnr, 1.0) / 0.03 + avg_gaze / 5.0)
                    accelerator.log({"val/composite_score": score}, step=global_step)

                    if score < best_val_score:
                        best_val_score = score
                        best_path = os.path.join(save_dir, "best_model.pth")
                        unwrapped_best = accelerator.unwrap_model(model)
                        torch.save({
                            'global_step': global_step,
                            'best_score':  best_val_score,
                            'metrics': {
                                'l1': avg_l1, 'lpips': avg_lpips,
                                'psnr': avg_psnr, 'gaze_error': avg_gaze,
                            },
                            'model_state_dict':   unwrapped_best.state_dict(),
                            'gaze_mlp_state_dict': accelerator.unwrap_model(gaze_mlp).state_dict(),
                            'net_state_dict':     unwrapped_best.net.state_dict(),
                        }, best_path)
                        logger.info(
                            f"New best model (score={score:.4f}): "
                            f"l1={avg_l1:.4f} lpips={avg_lpips:.4f} "
                            f"psnr={avg_psnr:.2f} gaze={avg_gaze:.4f}"
                        )

                # TensorBoard visualisation — training batch
                with torch.no_grad():
                    vis_pasted = accelerator.unwrap_model(model).paste_eyes(
                        generated_tight, target_image, target_eye_bbox,
                    )
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
                            val_vis_data['gen'],
                            val_vis_data['tgt_img'],
                            val_vis_data['bbox'],
                        )
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
                gaze_mlp.train()

            # ── 9. Periodic checkpoint ───────────────────────────────────────
            if (global_step % cfg.checkpointing_steps == 0 and
                    accelerator.is_main_process):
                ckpt_path = os.path.join(save_dir, f"checkpoint-{global_step}.pth")
                unwrapped  = accelerator.unwrap_model(model)
                save_dict  = {
                    'global_step':          global_step,
                    'model_state_dict':     unwrapped.state_dict(),
                    'gaze_mlp_state_dict':  accelerator.unwrap_model(gaze_mlp).state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                }
                if 'eye_discriminator' in loss_dict:
                    save_dict['discriminator_state_dict'] = (
                        accelerator.unwrap_model(loss_dict['eye_discriminator']).state_dict()
                    )
                    save_dict['optimizer_D_state_dict'] = (
                        loss_dict['eye_optimizer_D'].state_dict()
                    )
                    if 'eye_scheduler_D' in loss_dict:
                        save_dict['scheduler_D_state_dict'] = (
                            loss_dict['eye_scheduler_D'].state_dict()
                        )
                torch.save(save_dict, ckpt_path)
                logger.info(f"Saved checkpoint: {ckpt_path}")

                # Rotate old checkpoints
                ckpts = sorted(
                    [f for f in os.listdir(save_dir)
                     if f.startswith("checkpoint-") and f.endswith(".pth")],
                    key=lambda x: int(x.split("-")[1].split(".")[0]),
                )
                while len(ckpts) > cfg.total_limit:
                    os.remove(os.path.join(save_dir, ckpts.pop(0)))

    # ── Final save ───────────────────────────────────────────────────────────
    if accelerator.is_main_process:
        unwrapped  = accelerator.unwrap_model(model)
        final_path = os.path.join(save_dir, "final_model.pth")
        torch.save({
            'model_state_dict':    unwrapped.state_dict(),
            'gaze_mlp_state_dict': accelerator.unwrap_model(gaze_mlp).state_dict(),
            'net_state_dict':      unwrapped.net.state_dict(),
        }, final_path)
        logger.info(f"Training complete. Final model: {final_path}")

    accelerator.end_training()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    main(cfg, args.config)
