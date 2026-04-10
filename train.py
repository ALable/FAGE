"""
FAGE — Training Script for Eye-Only Gaze Generation

Phase 1: Train EyeOnlyGazeDiC on all users (shared model)
Phase 2: Freeze UNet, train DynamicMaskGenerator per user
"""
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
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

    # Phase 2: 带 subject adapter
    subject_adapter_config = None
    personalization = getattr(cfg, 'personalization', None)
    if personalization and personalization.get('enabled', False) and personalization.get('phase', 1) == 2:
        subject_adapter_config = OmegaConf.to_container(personalization.subject_adapter, resolve=True)

    model = EyeOnlyWrapper(unet_config, subject_adapter_config=subject_adapter_config)

    # Phase 2: 加载 Phase 1 权重并冻结 GazeControlNet (eye_unet)
    if subject_adapter_config is not None:
        phase1_ckpt = personalization.get('phase1_checkpoint', None)
        if phase1_ckpt and os.path.exists(phase1_ckpt):
            state = torch.load(phase1_ckpt, map_location='cpu')
            if 'unet_state_dict' in state:
                model.eye_unet.load_state_dict(state['unet_state_dict'])
            elif 'model_state_dict' in state:
                unet_state = {k.replace('eye_unet.', ''): v
                              for k, v in state['model_state_dict'].items()
                              if k.startswith('eye_unet.')}
                if unet_state:
                    model.eye_unet.load_state_dict(unet_state)
            logger.info(f"Loaded Phase 1 GazeControlNet from {phase1_ckpt}")

        # 冻结 GazeControlNet
        for p in model.eye_unet.parameters():
            p.requires_grad = False
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Phase 2: GazeControlNet frozen, trainable (SubjectAdapter): {trainable/1e3:.1f}K")

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
    """将 [B,6,H,W] 的眼部 tensor 拆成左右眼，横向拼接后做 make_grid

    返回: [3, H, n*(W*2+2)] 的可视化 grid
    """
    imgs = _denorm(eye_tensor[:n].detach().float().cpu())
    left  = imgs[:, :3]           # [n, 3, H, W]
    right = imgs[:, 3:]           # [n, 3, H, W]
    # 左右横向拼 → [n, 3, H, W*2]
    lr = torch.cat([left, right], dim=3)
    return make_grid(lr, nrow=n, padding=2)


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
    personalization = getattr(cfg, 'personalization', None)
    is_phase2 = personalization and personalization.get('enabled', False) and personalization.get('phase', 1) == 2

    # === Create model ===
    model = create_model(cfg)
    gaze_mlp = create_gaze_mlp(cfg)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {total_params/1e6:.2f}M total, {trainable_params/1e6:.2f}M trainable")

    # Phase 2: gaze_mlp 不参与优化，冻结以避免梯度无端积累（optimizer 不含其参数）
    if is_phase2:
        for p in gaze_mlp.parameters():
            p.requires_grad = False
        logger.info("Phase 2: gaze_mlp frozen (requires_grad=False)")

    # === Optimizer ===
    if is_phase2:
        # Phase 2: 只优化 SubjectAdapter（主体外观保持模块）
        opt_params = [p for p in model.subject_adapter.parameters() if p.requires_grad]
    else:
        # Phase 1: 优化所有参数（GazeControlNet + SubjectAdapter + GazeMLP）
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
    # Phase 2 个性化: 使用 val subjects (Phase 1 从未见过的新用户) 做 SubjectAdapter fine-tune
    # Phase 1 共享训练: 使用 train subjects
    if is_phase2:
        _p2_split   = cfg.data.get('personalization_split', 'val')   # 默认 val subjects 作新用户
        _p2_prefixes = list(cfg.data.subject_keys) if cfg.data.get('subject_keys') else None
        _train_split = _p2_split
        _val_split   = _p2_split   # Phase 2: train/val 都来自同一批新用户，帧级别不重叠
        _prefixes    = _p2_prefixes
    else:
        _train_split = 'train'
        _val_split   = 'val'
        _prefixes    = None

    _ds_kwargs = dict(
        split_ratio=cfg.data.split_ratio,
        seed=cfg.seed,
        frame_offset_range=cfg.data.get('frame_offset_range', 2),
        eye_crop_size=cfg.data.get('eye_crop_size', [64, 64]),
        eye_expand_ratio=cfg.data.get('eye_expand_ratio', 1.5),
        input_eye_crop_size=cfg.data.get('input_eye_crop_size', None),
        input_eye_expand_ratio=cfg.data.get('input_eye_expand_ratio', None),
    )

    train_dataset = HDFDataset(cfg.data.hdf_path, prefixes=_prefixes, split=_train_split, **_ds_kwargs)
    val_dataset   = HDFDataset(cfg.data.hdf_path, prefixes=_prefixes, split=_val_split,   **_ds_kwargs)

    if is_phase2:
        logger.info(f"Phase 2 dataset: split='{_p2_split}', "
                    f"subjects={train_dataset.num_subjects}, "
                    f"train_frames={len(train_dataset)}, val_frames={len(val_dataset)}")

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
    logger.info(f"Phase: {'2 (Personalization - SubjectAdapter)' if is_phase2 else '1 (Shared)'}")
    logger.info(f"Epochs: {num_train_epochs}, Max steps: {cfg.solver.max_train_steps}")
    logger.info(f"Batch size: {cfg.data.train_bs}, Train samples: {len(train_dataset)}")

    global_step = 0
    best_val_score = float('inf')  # composite score，越小越好
    progress_bar = tqdm(range(cfg.solver.max_train_steps),
                        disable=not accelerator.is_local_main_process)

    for epoch in range(num_train_epochs):
        model.train()
        if not is_phase2:
            gaze_mlp.train()
        else:
            gaze_mlp.eval()

        for step, batch in enumerate(train_dataloader):
            if global_step >= cfg.solver.max_train_steps:
                break

            with accelerator.accumulate(model, gaze_mlp):
                # === 1. 提取配对帧数据 ===
                source_eye_crops = batch['source_input_eye_crops'].to(weight_dtype)  # [B,6,in_h,in_w] 大范围，模型输入
                target_eye_crops = batch['target_eye_crops'].to(weight_dtype)         # [B,6,h,w] 紧范围，监督 GT
                target_gaze = batch['target_gaze'].to(weight_dtype)
                target_head = batch['target_head'].to(weight_dtype)
                source_image    = batch['source_image'].to(weight_dtype)
                target_image    = batch['target_image'].to(weight_dtype)
                source_eye_bbox = batch['source_eye_bbox'].to(weight_dtype)

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
                gaze_perc_loss = None
                if cfg.loss_params.gaze_perceptual_loss > 0 and global_step >= start_perc:
                    pasted = accelerator.unwrap_model(model).paste_eyes(
                        generated_tight, source_image, source_eye_bbox)
                    gaze_perc_loss = loss_dict['angular_loss'](pasted, target_image)
                    loss = loss + cfg.loss_params.gaze_perceptual_loss * gaze_perc_loss

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
                # 保存第一个 val batch 用于可视化
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

                        # LPIPS: 左右眼横向拼接 → [B, 3, h, 2w]，输入已在 [-1, 1]
                        gen_lr = torch.cat([gen[:, :3], gen[:, 3:]], dim=3)
                        tgt_lr = torch.cat([tgt[:, :3], tgt[:, 3:]], dim=3)
                        val_lpips += lpips_fn(gen_lr, tgt_lr).mean().item()

                        # PSNR: 移至 [0, 2] 区间，clamp 防止模型输出略超 [-1,1]
                        val_psnr += piq_psnr(
                            (gen.float() + 1).clamp(0, 2),
                            (tgt.float() + 1).clamp(0, 2),
                            data_range=2.0,
                        ).item()

                        # Gaze error: 左眼 [-1,1] → [0,1] → resize 224 → ImageNet norm → estimator
                        gen_left = (gen[:, :3].float() + 1) / 2
                        gen_left = _gaze_eval_trans(gen_left)
                        gaze_pred, _ = gaze_estimator(gen_left)
                        val_gaze_error += gaze_angular_loss(tgt_gaze.float(), gaze_pred).item()

                        val_count += 1

                        if val_vis_data is None:
                            val_vis_data = {
                                'src': src,
                                'gen': gen,
                                'tgt': tgt,
                                'src_img': vbatch['source_image'].to(weight_dtype),
                                'tgt_img': vbatch['target_image'].to(weight_dtype),
                                'bbox':    vbatch['source_eye_bbox'].to(weight_dtype),
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
                        if is_phase2:
                            best_save['subject_adapter_state_dict'] = unwrapped_best.subject_adapter.state_dict()
                        torch.save(best_save, best_path)
                        logger.info(
                            f"New best model saved (score={score:.4f}): "
                            f"l1={avg_l1:.4f} lpips={avg_lpips:.4f} "
                            f"psnr={avg_psnr:.2f} gaze={avg_gaze:.4f}"
                        )

                # —— TensorBoard 图像可视化（train batch）——
                with torch.no_grad():
                    vis_pasted = accelerator.unwrap_model(model).paste_eyes(
                        generated_tight, source_image, source_eye_bbox)
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
                            val_vis_data['gen'], val_vis_data['src_img'], val_vis_data['bbox'])
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
                if not is_phase2:
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
                }
                if is_phase2:
                    # Phase 2: 单独保存 subject_adapter（小文件, 新用户只需更新此部分）
                    save_dict['subject_adapter_state_dict'] = unwrapped.subject_adapter.state_dict()
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
