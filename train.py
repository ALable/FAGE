"""
FAGE — Training Script for Eye-Only Gaze Generation (Phase 1: Shared Pretraining)

Train GazeControlNet (EyeOnlyGazeDiC + GazeMLP) on all users.
Per-user SubjectAdapter fine-tuning → see finetune_adapter.py


accelerate launch --num_processes 4 train.py --config configs/training/dic_eye_only.yaml


"""
import os
import random
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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

from models.gaze_dic import EyeOnlyWrapper
from models.gazenet import MLPNetwork
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
    unet_config = OmegaConf.to_container(cfg.dic_unet_params, resolve=True)
    model = EyeOnlyWrapper(unet_config)
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
        cross_condition=gaze_params.get('cross_condition', False),
    )
    return gaze_mlp


def build_g_lr_scheduler(optimizer, cfg):
    """Two-phase G scheduler that keeps LR at peak until D joins.

    Phase 0  [0, warmup)           – linear warm-up
    Phase 1  [warmup, start_gan)   – flat at peak LR  ← D not yet active
    Phase 2  [start_gan, max]      – cosine decay to 0 ← D and G train together

    This prevents the common failure mode where a global cosine schedule
    decays G's LR to near-zero by the time D enters training.
    """
    warmup    = cfg.solver.lr_warmup_steps
    start_gan = cfg.discriminator_train_params.start_gan
    max_steps = cfg.solver.max_train_steps

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup:
            return float(current_step) / float(max(1, warmup))
        if current_step < start_gan:
            return 1.0  # flat: G is at peak when D enters
        progress = float(current_step - start_gan) / float(max(1, max_steps - start_gan))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _get_tb_writer(accelerator):
    """从 accelerator trackers 中取出 TensorBoard SummaryWriter"""
    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            return tracker.writer
    return None


def _denorm(t):
    """[-1,1] → [0,1]，clamp 防止越界"""
    return (t * 0.5 + 0.5).clamp(0, 1)


def _eye_to_grid(eye_tensor, n=4):
    """将 [B,3,H,W*2] 的眼部 tensor（width concat）直接做 make_grid

    返回: [3, H, n*(W*2+2)] 的可视化 grid
    """
    imgs = _denorm(eye_tensor[:n].detach().float().cpu())  # [n, 3, H, W*2]
    return make_grid(imgs, nrow=n, padding=2)


def log_vis(accelerator, global_step, source_eye_large, generated_tight,
            target_eye_crops, source_image, target_image, pasted_face, n=4,
            prefix="train"):
    """写入 TensorBoard 可视化图像

    Args:
        source_eye_large:  [B,6,in_h,in_w]  模型输入（大范围眼）
        generated_tight:   [B,6,h,w]         生成结果（紧范围）
        target_eye_crops:  [B,6,h,w]         GT（紧范围）
        source_image:      [B,3,H,W]         原始人脸
        target_image:      [B,3,H,W]         目标人脸
        pasted_face:       [B,3,H,W]         贴回后的人脸
        prefix:            "train" or "val"   TensorBoard tag 前缀
    """
    writer = _get_tb_writer(accelerator)
    if writer is None:
        return

    p = prefix.rstrip("/")

    # —— 眼部对比（垂直排布：source / generated / gt）——
    grid_src = _eye_to_grid(source_eye_large, n)  # [3, H1, W1]
    grid_gen = _eye_to_grid(generated_tight,  n)  # [3, H2, W2]
    grid_gt  = _eye_to_grid(target_eye_crops,  n)
    # source 大范围眼 grid 可能比 tight grid 更大，resize 统一尺寸
    if grid_src.shape != grid_gen.shape:
        grid_src = F.interpolate(
            grid_src.unsqueeze(0), size=grid_gen.shape[1:], mode='bilinear',
            align_corners=False,
        ).squeeze(0)
    eye_rows = torch.cat([grid_src, grid_gen, grid_gt], dim=1)
    writer.add_image(f"{p}/eyes_compare", eye_rows, global_step)

    # —— 全脸对比（垂直排布：source / pasted / target）——
    def face_grid(t):
        imgs = _denorm(t[:n].detach().float().cpu())
        return make_grid(imgs, nrow=n, padding=2)

    face_rows = torch.cat([
        face_grid(source_image),
        face_grid(pasted_face),
        face_grid(target_image),
    ], dim=1)  # 沿 H 方向拼接 → [3, 3H, W]
    writer.add_image(f"{p}/face_compare", face_rows, global_step)


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

    # === Create model ===
    model = create_model(cfg)
    gaze_mlp = create_gaze_mlp(cfg)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {total_params/1e6:.2f}M total, {trainable_params/1e6:.2f}M trainable")

    # === Optimizer ===
    opt_params = list(model.parameters()) + list(gaze_mlp.parameters())

    optimizer = torch.optim.AdamW(
        opt_params,
        lr=cfg.solver.learning_rate,
        betas=(cfg.solver.adam_beta1, cfg.solver.adam_beta2),
        weight_decay=cfg.solver.adam_weight_decay,
        eps=cfg.solver.adam_epsilon,
    )

    # === LR scheduler ===
    # 两阶段：warmup → flat（G-only 阶段 LR 保持峰值）→ cosine 衰减（D 加入后才开始衰减）
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

    # Select dataset based on config
    dataset_type = cfg.data.get('dataset_type', 'gaze_hdf')

    if dataset_type == 'eth_xgaze':
        logger.info("Using ETH-XGaze LMDB dataset")
        train_dataset = ETHXGazePairedDataset(
            cfg.data.lmdb_path,
            split='train',
            resolution=cfg.data.get('image_size', 256),
            aug=False,
            **_ds_kwargs
        )
        val_dataset = ETHXGazePairedDataset(
            cfg.data.lmdb_path,
            split='val',
            resolution=cfg.data.get('image_size', 256),
            aug=False,
            **_ds_kwargs
        )
    elif dataset_type == 'gaze_hdf':
        logger.info("Using GazeCapture HDF5 dataset")
        train_dataset = HDFDataset(cfg.data.hdf_path, split='train', **_ds_kwargs)
        val_dataset   = HDFDataset(cfg.data.hdf_path, split='val',   **_ds_kwargs)
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}. Must be 'gaze_hdf' or 'eth_xgaze'")

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

    # === Loss functions (discriminator + gaze perceptual) ===
    loss_dict = initialize_loss_functions(cfg, accelerator, cfg.solver.max_train_steps)

    # Prepare discriminator with accelerator for DDP + mixed precision
    if 'eye_discriminator' in loss_dict:
        loss_dict['eye_discriminator'], loss_dict['eye_optimizer_D'] = accelerator.prepare(
            loss_dict['eye_discriminator'], loss_dict['eye_optimizer_D']
        )

    # === Val metric tools (LPIPS + gaze estimator) ===
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

    # === Training ===
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.solver.gradient_accumulation_steps)
    num_train_epochs = math.ceil(cfg.solver.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        run_time = datetime.now().strftime("%Y%m%d-%H%M")
        accelerator.init_trackers(cfg.exp_name, init_kwargs={"mlflow": {"run_name": run_time}})

    logger.info("***** FAGE Training *****")
    logger.info(f"Epochs: {num_train_epochs}, Max steps: {cfg.solver.max_train_steps}")
    logger.info(f"Batch size: {cfg.data.train_bs}, Train samples: {len(train_dataset)}")

    global_step = 0
    start_epoch = 0
    steps_to_skip = 0

    # === 断点续训 ===
    resume_path = cfg.get('resume_from_checkpoint', False)
    if resume_path and isinstance(resume_path, str) and os.path.exists(resume_path):
        logger.info(f"从 checkpoint 恢复: {resume_path}")
        ckpt = torch.load(resume_path, map_location='cpu')
        global_step = ckpt.get('global_step', 0)
        resume_strict = cfg.get('resume_strict', True)
        incompatible = accelerator.unwrap_model(model).load_state_dict(
            ckpt['model_state_dict'], strict=resume_strict)
        if not resume_strict and (incompatible.missing_keys or incompatible.unexpected_keys):
            logger.warning(f"  model load_state_dict (strict=False): "
                           f"missing={incompatible.missing_keys}, "
                           f"unexpected={incompatible.unexpected_keys}")
        incompatible_mlp = accelerator.unwrap_model(gaze_mlp).load_state_dict(
            ckpt['gaze_mlp_state_dict'], strict=resume_strict)
        if not resume_strict and (incompatible_mlp.missing_keys or incompatible_mlp.unexpected_keys):
            logger.warning(f"  gaze_mlp load_state_dict (strict=False): "
                           f"missing={incompatible_mlp.missing_keys}, "
                           f"unexpected={incompatible_mlp.unexpected_keys}")
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
        start_epoch   = global_step // num_update_steps_per_epoch
        steps_to_skip = global_step % num_update_steps_per_epoch
        logger.info(f"已恢复到 step={global_step}（epoch={start_epoch}，本 epoch 跳过 {steps_to_skip} 步）")

    best_val_score = float('inf')  # composite score，越小越好
    progress_bar = tqdm(range(cfg.solver.max_train_steps),
                        initial=global_step,
                        disable=not accelerator.is_local_main_process)

    for epoch in range(start_epoch, num_train_epochs):
        model.train()
        gaze_mlp.train()

        for step, batch in enumerate(train_dataloader):
            # 续训时跳过本 epoch 已处理过的步骤
            if epoch == start_epoch and step < steps_to_skip:
                continue
            if global_step >= cfg.solver.max_train_steps:
                break

            with accelerator.accumulate(model, gaze_mlp):
                # === 1. 提取配对帧数据 ===
                source_eye_crops = batch['source_input_eye_crops'].to(weight_dtype)  # [B,3,in_h,in_w*2] 大范围，模型输入
                target_eye_crops = batch['target_eye_crops'].to(weight_dtype)         # [B,3,h,w*2] 紧范围，监督 GT
                target_gaze = batch['target_gaze'].to(weight_dtype)
                target_head = batch['target_head'].to(weight_dtype)
                source_image    = batch['source_image'].to(weight_dtype)
                target_image    = batch['target_image'].to(weight_dtype)
                source_eye_bbox = batch['source_eye_bbox'].to(weight_dtype)
                target_eye_bbox = batch["target_eye_bbox"].to(weight_dtype)

                # === 2. Gaze embedding ===
                head_emb, gaze_emb = gaze_mlp(target_head, target_gaze)
                gaze_prompt = torch.cat([head_emb.unsqueeze(1), gaze_emb.unsqueeze(1)], dim=1)

                # === 3. Forward: source眼 + target_gaze → 生成 target 眼 ===
                generated_eyes = model(source_eye_crops, gaze_prompt)

                # Center-crop 生成结果到监督尺寸（输入比监督区域大）
                in_h, in_w   = source_eye_crops.shape[-2:]
                tgt_h, tgt_w = target_eye_crops.shape[-2:]
                if in_h != tgt_h or in_w != tgt_w:
                    pad_h = (in_h - tgt_h) // 2
                    pad_w = (in_w - tgt_w) // 2
                    generated_tight = generated_eyes[:, :, pad_h:pad_h+tgt_h, pad_w:pad_w+tgt_w]
                else:
                    generated_tight = generated_eyes

                # === 4. Loss ===
                # L1 on eye crops
                l1_loss = F.l1_loss(generated_tight, target_eye_crops)
                loss = cfg.loss_params.l1_loss * l1_loss

                # Gaze perceptual loss（贴回原图后，用 GazePerceptualLoss from vgg_eye.py）
                start_perc = cfg.loss_params.get('start_gaze_perceptual', 500)
                start_id   = cfg.loss_params.get('start_id_loss', 1000)
                gaze_perc_loss = None
                id_loss_val    = None

                # 只要有任意一个 loss 需要 pasted face，就统一计算一次
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

                # ID loss: 贴回脸 与 源脸 保持身份一致
                if cfg.loss_params.get('id_loss', 0) > 0 and global_step >= start_id:
                    id_loss_val = loss_dict['id_loss'].loss(pasted, source_image)
                    loss = loss + cfg.loss_params.id_loss * id_loss_val

                # Eye GAN: Generator step（从 start_gan 步开始）
                start_gan = cfg.discriminator_train_params.start_gan
                g_adv = None
                if cfg.loss_params.eye_gan_loss > 0 and global_step >= start_gan:
                    # Freeze D params during G step
                    for p in loss_dict['eye_discriminator'].parameters():
                        p.requires_grad_(False)
                    fake_pred = loss_dict['eye_discriminator'](generated_tight)
                    g_adv = generator_loss(fake_pred, fake_pred.device)
                    loss = loss + cfg.loss_params.eye_gan_loss * g_adv
                    for p in loss_dict['eye_discriminator'].parameters():
                        p.requires_grad_(True)

                # === 5. Backward ===
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    all_trainable = (
                        [p for p in model.parameters() if p.requires_grad] +
                        [p for p in gaze_mlp.parameters() if p.requires_grad]
                    )
                    accelerator.clip_grad_norm_(all_trainable, cfg.solver.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # === 5.5 Eye Discriminator step ===
            if cfg.loss_params.eye_gan_loss > 0 and global_step >= start_gan:
                with accelerator.accumulate(loss_dict['eye_discriminator']):
                    loss_dict['eye_optimizer_D'].zero_grad()
                    # Single forward with cat([real, fake]) avoids BN/SN inplace version
                    # conflicts that would occur with two separate forward passes
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

            # === 6. Logging ===
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
                if g_adv is not None:
                    log_dict["train/g_adv_loss"] = g_adv.item()
                    log_dict["train/d_loss"] = d_loss.item()
                accelerator.log(log_dict, step=global_step)

            if global_step % 500 == 0:
                logger.info(f"Step {global_step}: l1={l1_loss.item():.4f}, total={loss.item():.4f}")

            # === 7. Validation ===
            if global_step % cfg.val_freq == 0 and accelerator.is_main_process:
                model.eval()
                gaze_mlp.eval()
                val_l1 = 0
                val_lpips = 0
                val_psnr = 0
                val_gaze_error = 0
                val_count = 0
                # 随机抽取一个 val batch 用于可视化（reservoir sampling）
                val_vis_data = None
                with torch.no_grad():
                    for vi, vbatch in enumerate(val_dataloader):
                        if vi >= cfg.val_batch_limit:
                            break
                        src = vbatch['source_input_eye_crops'].to(weight_dtype)  # 大范围输入
                        tgt = vbatch['target_eye_crops'].to(weight_dtype)         # 紧范围 GT
                        tgt_gaze = vbatch['target_gaze'].to(weight_dtype)
                        tgt_head = vbatch['target_head'].to(weight_dtype)

                        h_emb, g_emb = gaze_mlp(tgt_head, tgt_gaze)
                        gp = torch.cat([h_emb.unsqueeze(1), g_emb.unsqueeze(1)], dim=1)
                        gen = model(src, gp)

                        # center-crop 到监督尺寸
                        si_h, si_w = src.shape[-2:]
                        tg_h, tg_w = tgt.shape[-2:]
                        if si_h != tg_h or si_w != tg_w:
                            ph = (si_h - tg_h) // 2
                            pw = (si_w - tg_w) // 2
                            gen = gen[:, :, ph:ph+tg_h, pw:pw+tg_w]

                        val_l1 += F.l1_loss(gen, tgt).item()

                        # LPIPS: gen/tgt 已为 [B, 3, h, 2w]（宽度拼接），直接送入
                        val_lpips += lpips_fn(gen, tgt).mean().item()

                        # PSNR: 移至 [0, 2] 区间，clamp 防止模型输出略超 [-1,1]
                        val_psnr += piq_psnr(
                            (gen.float() + 1).clamp(0, 2),
                            (tgt.float() + 1).clamp(0, 2),
                            data_range=2.0,
                        ).item()

                        # Gaze error: 取左眼（宽度左半部分）[-1,1] → [0,1] → resize 224 → ImageNet norm → estimator
                        gen_left = (gen[:, :, :, :gen.shape[3]//2].float() + 1) / 2
                        gen_left = _gaze_eval_trans(gen_left)
                        gaze_pred, _ = gaze_estimator(gen_left)
                        val_gaze_error += gaze_angular_loss(tgt_gaze.float(), gaze_pred).item()

                        val_count += 1

                        # Reservoir sampling: 以 1/val_count 的概率替换当前选中的 batch
                        if random.randint(0, val_count - 1) == 0:
                            val_vis_data = {
                                'src': src,
                                'gen': gen,
                                'tgt': tgt,
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
                        f"Val step {global_step}: "
                        f"l1={avg_l1:.4f}  "
                        f"lpips={avg_lpips:.4f}  "
                        f"psnr={avg_psnr:.2f}dB  "
                        f"gaze={avg_gaze:.4f}°"
                    )

                    # === Best model: composite score (1:1:1:1) ===
                    # L1, LPIPS, gaze_error 越小越好；PSNR 越大越好 → 取倒数
                    # 归一化量级: L1~0.05, LPIPS~0.1, 1/PSNR~0.03, gaze~5°
                    # 用各自典型量级做 scale 使四项贡献均衡
                    score = avg_l1 / 0.05 + avg_lpips / 0.1 + 1.0 / max(avg_psnr, 1.0) / 0.03 + avg_gaze / 5.0

                    accelerator.log({"val/composite_score": score}, step=global_step)

                    if score < best_val_score:
                        best_val_score = score
                        best_path = os.path.join(save_dir, "best_model.pth")
                        unwrapped_best = accelerator.unwrap_model(model)
                        best_save = {
                            'global_step': global_step,
                            'best_score': best_val_score,
                            'metrics': {
                                'l1': avg_l1, 'lpips': avg_lpips,
                                'psnr': avg_psnr, 'gaze_error': avg_gaze,
                            },
                            'model_state_dict': unwrapped_best.state_dict(),
                            'gaze_mlp_state_dict': accelerator.unwrap_model(gaze_mlp).state_dict(),
                            'unet_state_dict': unwrapped_best.eye_unet.state_dict(),
                        }
                        torch.save(best_save, best_path)
                        logger.info(
                            f"New best model saved (score={score:.4f}): "
                            f"l1={avg_l1:.4f} lpips={avg_lpips:.4f} "
                            f"psnr={avg_psnr:.2f} gaze={avg_gaze:.4f}"
                        )

                # —— TensorBoard 图像可视化（train batch）——
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

                # —— TensorBoard 图像可视化（val batch）——
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
                gaze_mlp.train()

            # === 8. Save checkpoint ===
            if global_step % cfg.checkpointing_steps == 0 and accelerator.is_main_process:
                ckpt_path = os.path.join(save_dir, f"checkpoint-{global_step}.pth")
                unwrapped = accelerator.unwrap_model(model)
                save_dict = {
                    'global_step': global_step,
                    'model_state_dict': unwrapped.state_dict(),
                    'gaze_mlp_state_dict': accelerator.unwrap_model(gaze_mlp).state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                }
                # 判别器相关（仅在 D 已被初始化后才存入）
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
