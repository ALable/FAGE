"""
FAGE — Training Script for Eye-Only Gaze Generation

Phase 1: Train EyeOnlyGazeDiC on all users (shared model)
Phase 2: Freeze UNet, train DynamicMaskGenerator per user
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from tqdm.auto import tqdm
from omegaconf import OmegaConf
from datetime import timedelta, datetime
from accelerate.utils import DistributedDataParallelKwargs, LoggerType
from accelerate.logging import get_logger
from accelerate import InitProcessGroupKwargs, Accelerator
import warnings
import torch
import torch.nn.functional as F
import time
import math
import logging
import argparse
import shutil

from models.gaze_dic import EyeOnlyWrapper
from models.gazenet import MLPNetwork
from dataset.gaze_capture import HDFDataset
from torch.utils.data import DataLoader

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
    """创建模型 (Phase 1 或 Phase 2)"""
    unet_config = OmegaConf.to_container(cfg.dic_unet_params, resolve=True)

    # Phase 2: 带 mask generator
    mask_gen_config = None
    personalization = getattr(cfg, 'personalization', None)
    if personalization and personalization.get('enabled', False) and personalization.get('phase', 1) == 2:
        mask_gen_config = OmegaConf.to_container(personalization.mask_generator, resolve=True)

    model = EyeOnlyWrapper(unet_config, mask_generator_config=mask_gen_config)

    # Phase 2: 加载 Phase 1 权重并冻结 UNet
    if mask_gen_config is not None:
        phase1_ckpt = personalization.get('phase1_checkpoint', None)
        if phase1_ckpt and os.path.exists(phase1_ckpt):
            state = torch.load(phase1_ckpt, map_location='cpu')
            if 'unet_state_dict' in state:
                model.eye_unet.load_state_dict(state['unet_state_dict'])
            elif 'model_state_dict' in state:
                # 尝试从完整模型状态中提取 UNet 部分
                unet_state = {k.replace('eye_unet.', ''): v
                              for k, v in state['model_state_dict'].items()
                              if k.startswith('eye_unet.')}
                if unet_state:
                    model.eye_unet.load_state_dict(unet_state)
            logger.info(f"Loaded Phase 1 UNet from {phase1_ckpt}")

        # 冻结 UNet
        for p in model.eye_unet.parameters():
            p.requires_grad = False
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Phase 2: UNet frozen, trainable params (mask generator): {trainable/1e3:.1f}K")

    return model


def create_gaze_mlp(cfg):
    """创建 gaze embedding MLP"""
    gaze_params = cfg.model_params.gazenet_params
    gaze_dim = cfg.dic_unet_params.get('gaze_dim', 64)
    gaze_mlp = MLPNetwork(
        num_in=gaze_params.num_in,
        num_hidden=gaze_params.num_hidden,
        num_out=gaze_dim,
        num_layers=gaze_params.num_layers,
    )
    return gaze_mlp


def main(cfg, config_file_path=None):
    exp_name = cfg.exp_name
    save_dir = f"{cfg.output_dir}/{exp_name}"
    os.makedirs(save_dir, exist_ok=True)

    # Save config
    if config_file_path and os.path.exists(config_file_path):
        shutil.copy2(config_file_path, os.path.join(save_dir, "config.yaml"))
    else:
        OmegaConf.save(cfg, os.path.join(save_dir, "config.yaml"))

    # Accelerator
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
    personalization = getattr(cfg, 'personalization', None)
    is_phase2 = personalization and personalization.get('enabled', False) and personalization.get('phase', 1) == 2

    # === Create model ===
    model = create_model(cfg)
    gaze_mlp = create_gaze_mlp(cfg)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {total_params/1e6:.2f}M total, {trainable_params/1e6:.2f}M trainable")

    # === Optimizer ===
    if is_phase2:
        # Phase 2: 只优化 mask generator
        opt_params = [p for p in model.mask_generator.parameters() if p.requires_grad]
    else:
        # Phase 1: 优化所有参数
        opt_params = list(model.parameters()) + list(gaze_mlp.parameters())

    optimizer = torch.optim.AdamW(
        opt_params,
        lr=cfg.solver.learning_rate,
        betas=(cfg.solver.adam_beta1, cfg.solver.adam_beta2),
        weight_decay=cfg.solver.adam_weight_decay,
        eps=cfg.solver.adam_epsilon,
    )

    # === LR scheduler ===
    from diffusers.optimization import get_scheduler
    lr_scheduler = get_scheduler(
        cfg.solver.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.solver.lr_warmup_steps,
        num_training_steps=cfg.solver.max_train_steps,
    )

    # === Dataset ===
    train_dataset = HDFDataset(
        cfg.data.hdf_path,
        split="train",
        split_ratio=cfg.data.split_ratio,
        seed=cfg.seed,
        frame_offset_range=cfg.data.get('frame_offset_range', 2),
        eye_crop_size=cfg.data.get('eye_crop_size', [64, 64]),
        eye_expand_ratio=cfg.data.get('eye_expand_ratio', 1.5),
    )
    val_dataset = HDFDataset(
        cfg.data.hdf_path,
        split="val",
        split_ratio=cfg.data.split_ratio,
        seed=cfg.seed,
        frame_offset_range=cfg.data.get('frame_offset_range', 2),
        eye_crop_size=cfg.data.get('eye_crop_size', [64, 64]),
        eye_expand_ratio=cfg.data.get('eye_expand_ratio', 1.5),
    )

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
    model, gaze_mlp, optimizer, lr_scheduler, train_dataloader, val_dataloader = accelerator.prepare(
        model, gaze_mlp, optimizer, lr_scheduler, train_dataloader, val_dataloader
    )

    # === Training ===
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.solver.gradient_accumulation_steps)
    num_train_epochs = math.ceil(cfg.solver.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        run_time = datetime.now().strftime("%Y%m%d-%H%M")
        accelerator.init_trackers(cfg.exp_name, init_kwargs={"mlflow": {"run_name": run_time}})

    logger.info("***** FAGE Training *****")
    logger.info(f"Phase: {'2 (Personalization)' if is_phase2 else '1 (Shared)'}")
    logger.info(f"Epochs: {num_train_epochs}, Max steps: {cfg.solver.max_train_steps}")
    logger.info(f"Batch size: {cfg.data.train_bs}, Train samples: {len(train_dataset)}")

    global_step = 0
    progress_bar = tqdm(range(cfg.solver.max_train_steps),
                        disable=not accelerator.is_local_main_process)

    # Phase 2 温度退火参数
    tau_init = personalization.mask_generator.tau_init if is_phase2 else 1.0
    tau_min = personalization.mask_generator.tau_min if is_phase2 else 0.1
    lambda_sparse = personalization.lambda_sparse if is_phase2 else 0
    lambda_entropy = personalization.lambda_entropy if is_phase2 else 0

    for epoch in range(num_train_epochs):
        model.train()
        gaze_mlp.train()

        for step, batch in enumerate(train_dataloader):
            if global_step >= cfg.solver.max_train_steps:
                break

            # === 1. 提取配对帧数据 ===
            source_eye_crops = batch['source_eye_crops'].to(weight_dtype)
            target_eye_crops = batch['target_eye_crops'].to(weight_dtype)
            target_gaze = batch['target_gaze'].to(weight_dtype)
            target_head = batch['target_head'].to(weight_dtype)

            # === 2. Gaze embedding ===
            head_emb, gaze_emb = gaze_mlp(target_head, target_gaze)
            gaze_prompt = torch.cat([head_emb.unsqueeze(1), gaze_emb.unsqueeze(1)], dim=1)

            # Phase 2: 展平 gaze 给 mask generator
            gaze_cond_flat = None
            if is_phase2:
                gaze_cond_flat = gaze_emb  # [B, gaze_dim]
                # 温度退火
                progress = global_step / cfg.solver.max_train_steps
                tau = tau_init * (1 - progress) + tau_min * progress
                unwrapped = accelerator.unwrap_model(model)
                if unwrapped.mask_generator is not None:
                    unwrapped.mask_generator.set_tau(tau)

            # === 3. Forward: source眼 + target_gaze → 生成 target 眼 ===
            generated_eyes = model(source_eye_crops, gaze_prompt, gaze_cond_flat=gaze_cond_flat)

            # === 4. Loss ===
            # L1 on eye crops
            l1_loss = F.l1_loss(generated_eyes, target_eye_crops)
            loss = cfg.loss_params.l1_loss * l1_loss

            # VGG perceptual on eye crops (if VGG available)
            # TODO: 添加 VGG perceptual loss

            # Phase 2: Mask 正则化
            if is_phase2 and lambda_sparse > 0:
                unwrapped = accelerator.unwrap_model(model)
                if unwrapped.mask_generator is not None:
                    masks = unwrapped.mask_generator(source_eye_crops, gaze_cond_flat)

                    # 稀疏正则: 鼓励 mask≈1
                    sparsity_loss = sum((1 - m).abs().mean() for m in masks) * lambda_sparse
                    loss = loss + sparsity_loss

                    # 熵正则: 避免极端二值
                    if lambda_entropy > 0:
                        entropy_loss = sum(
                            -(m * torch.log(m + 1e-8) + (1-m) * torch.log(1-m + 1e-8)).mean()
                            for m in masks
                        ) * lambda_entropy
                        loss = loss + entropy_loss

            # === 5. Backward ===
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    cfg.solver.max_grad_norm
                )

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # === 6. Logging ===
            global_step += 1
            progress_bar.update(1)

            if global_step % 100 == 0 and accelerator.is_main_process:
                log_dict = {
                    "train/l1_loss": l1_loss.item(),
                    "train/total_loss": loss.item(),
                    "train/lr": lr_scheduler.get_last_lr()[0],
                }
                if is_phase2:
                    log_dict["train/tau"] = tau
                    if lambda_sparse > 0:
                        log_dict["train/sparsity_loss"] = sparsity_loss.item()
                accelerator.log(log_dict, step=global_step)

            if global_step % 500 == 0:
                logger.info(f"Step {global_step}: l1={l1_loss.item():.4f}, total={loss.item():.4f}")

            # === 7. Validation ===
            if global_step % cfg.val_freq == 0 and accelerator.is_main_process:
                model.eval()
                val_l1 = 0
                val_count = 0
                with torch.no_grad():
                    for vi, vbatch in enumerate(val_dataloader):
                        if vi >= cfg.val_batch_limit:
                            break
                        src = vbatch['source_eye_crops'].to(weight_dtype)
                        tgt = vbatch['target_eye_crops'].to(weight_dtype)
                        tgt_gaze = vbatch['target_gaze'].to(weight_dtype)
                        tgt_head = vbatch['target_head'].to(weight_dtype)

                        h_emb, g_emb = gaze_mlp(tgt_head, tgt_gaze)
                        gp = torch.cat([h_emb.unsqueeze(1), g_emb.unsqueeze(1)], dim=1)
                        gf = g_emb if is_phase2 else None
                        gen = model(src, gp, gaze_cond_flat=gf)

                        val_l1 += F.l1_loss(gen, tgt).item()
                        val_count += 1

                if val_count > 0:
                    accelerator.log({"val/l1_loss": val_l1 / val_count}, step=global_step)
                    logger.info(f"Val step {global_step}: l1={val_l1/val_count:.4f}")
                model.train()

            # === 8. Save checkpoint ===
            if global_step % cfg.checkpointing_steps == 0 and accelerator.is_main_process:
                ckpt_path = os.path.join(save_dir, f"checkpoint-{global_step}.pth")
                unwrapped = accelerator.unwrap_model(model)
                save_dict = {
                    'global_step': global_step,
                    'model_state_dict': unwrapped.state_dict(),
                    'gaze_mlp_state_dict': accelerator.unwrap_model(gaze_mlp).state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                if is_phase2:
                    # Phase 2: 也单独保存 mask generator (小文件, 方便部署)
                    save_dict['mask_generator_state_dict'] = unwrapped.mask_generator.state_dict()
                    save_dict['unet_state_dict'] = unwrapped.eye_unet.state_dict()

                torch.save(save_dict, ckpt_path)
                logger.info(f"Saved checkpoint: {ckpt_path}")

                # 清理旧 checkpoint
                ckpts = sorted(
                    [f for f in os.listdir(save_dir) if f.startswith("checkpoint-") and f.endswith(".pth")],
                    key=lambda x: int(x.split("-")[1].split(".")[0])
                )
                while len(ckpts) > cfg.total_limit:
                    old = os.path.join(save_dir, ckpts.pop(0))
                    os.remove(old)

    # 最终保存
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        final_path = os.path.join(save_dir, "final_model.pth")
        torch.save({
            'model_state_dict': unwrapped.state_dict(),
            'gaze_mlp_state_dict': accelerator.unwrap_model(gaze_mlp).state_dict(),
            'unet_state_dict': unwrapped.eye_unet.state_dict(),
        }, final_path)
        logger.info(f"Training complete. Final model: {final_path}")

    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    main(cfg, args.config)
