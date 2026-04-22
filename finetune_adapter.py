"""
FAGE — Per-User SubjectAdapter Fine-Tuning

给定 Phase 1 预训练 checkpoint，为每个新用户独立 fine-tune 一个 SubjectAdapter，
保存轻量级 adapter 权重文件。

用法:
    # 单个用户
    python finetune_adapter.py --config configs/training/dic_eye_only.yaml \
        --checkpoint /path/to/phase1.pth \
        --subjects 00002 \
        --max_steps 500

    # 批量（所有 val 用户）
    python finetune_adapter.py --config configs/training/dic_eye_only.yaml \
        --checkpoint /path/to/phase1.pth \
        --split val \
        --max_steps 500

    # 自定义输出目录
    python finetune_adapter.py ... --output_dir /path/to/adapters

输出:
    output_dir/
      adapters/
        00002.pth       # subject_adapter state_dict
        00123.pth
        ...
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import sys
import argparse
import logging
import random
import numpy as np
import torch
import torch.nn.functional as F
import lpips
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from diffusers.optimization import get_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.gaze_dic import EyeOnlyWrapper
from models.gazenet import MLPNetwork
from models.subject_adapter import SubjectAdapter
from dataset.gaze_capture import HDFDataset
from dataset.eth_xgaze_paired import ETHXGazePairedDataset
from loss.basic_loss import IDLoss, discriminator_loss, generator_loss
from loss.discriminator import MultiScaleDiscriminator

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_phase1(cfg, checkpoint_path, device):
    """Load Phase 1 model (eye_unet + gaze_mlp), freeze all params."""
    unet_config = OmegaConf.to_container(cfg.dic_unet_params, resolve=True)
    model = EyeOnlyWrapper(unet_config)

    gaze_params = cfg.model_params.gazenet_params
    gaze_dim = cfg.dic_unet_params.get('gaze_dim', 64)
    gaze_mlp = MLPNetwork(
        num_in=gaze_params.num_in,
        num_hidden=gaze_params.num_hidden,
        num_out=gaze_dim,
        num_layers=gaze_params.num_layers,
        cross_condition=gaze_params.get('cross_condition', False),
    )
    state = torch.load(checkpoint_path, map_location='cpu')
    resume_strict = cfg.get('resume_strict', True)
    if 'unet_state_dict' in state:
        model.eye_unet.load_state_dict(state['unet_state_dict'], strict=resume_strict)
    elif 'model_state_dict' in state:
        model.load_state_dict(state['model_state_dict'], strict=False)
    else:
        raise KeyError(f"No model weights found. Keys: {list(state.keys())}")

    if 'gaze_mlp_state_dict' in state:
        gaze_mlp.load_state_dict(state['gaze_mlp_state_dict'], strict=resume_strict)


    # Freeze everything
    for p in model.parameters():
        p.requires_grad = False
    for p in gaze_mlp.parameters():
        p.requires_grad = False

    model.to(device).eval()
    gaze_mlp.to(device).eval()

    logger.info(f"Phase 1 loaded from {checkpoint_path}")
    return model, gaze_mlp


def create_adapter(model, cfg, device):
    """Create a fresh SubjectAdapter (zero-init) attached to model."""
    block_channels = model._get_block_channels()
    in_ch = cfg.dic_unet_params.get('in_channels', 3)
    subject_dim = cfg.get('subject_adapter', {}).get('subject_dim', 128)
    adapter = SubjectAdapter(
        in_channels=in_ch,
        subject_dim=subject_dim,
        block_channels=block_channels,
    ).to(device)
    model.subject_adapter = adapter
    return adapter


def get_subject_keys(cfg, split):
    """Get all subject keys for a given split from HDF5."""
    ds = HDFDataset(
        cfg.data.hdf_path,
        split=split,
        split_ratio=cfg.data.split_ratio,
        seed=cfg.seed,
        frame_offset_range=cfg.data.get('frame_offset_range', 2),
        eye_crop_size=cfg.data.get('eye_crop_size', [64, 64]),
        eye_expand_ratio=cfg.data.get('eye_expand_ratio', 1.5),
        input_eye_crop_size=cfg.data.get('input_eye_crop_size', None),
        input_eye_expand_ratio=cfg.data.get('input_eye_expand_ratio', None),
    )
    keys = sorted(ds.prefix_to_indices.keys())
    logger.info(f"Split '{split}': {len(keys)} subjects, {len(ds)} frames total")
    return keys


def _make_eye_grid(source_eye, generated_tight, target_eye, max_samples=4):
    """Create a comparison grid: [source_L | source_R | generated_L | generated_R | target_L | target_R].
    All images normalized to [0,1]. Returns a single grid tensor [3, H, W*6*max_samples]."""
    n = min(source_eye.size(0), max_samples)
    src_l = source_eye[:n, :3].cpu().float()   # left eye
    src_r = source_eye[:n, 3:].cpu().float()   # right eye
    gen_l = generated_tight[:n, :3].cpu().float()
    gen_r = generated_tight[:n, 3:].cpu().float()
    tgt_l = target_eye[:n, :3].cpu().float()
    tgt_r = target_eye[:n, 3:].cpu().float()

    def norm(t):
        lo, hi = t.flatten(1).min(1)[0], t.flatten(1).max(1)[0]
        lo = lo[:, None, None, None]; hi = hi[:, None, None, None]
        return (t - lo) / (hi - lo + 1e-8).clamp(0, 1)

    # Resize generated to match source size for display
    h, w = src_l.shape[-2:]
    gen_l = F.interpolate(gen_l, (h, w), mode='bilinear', align_corners=False)
    gen_r = F.interpolate(gen_r, (h, w), mode='bilinear', align_corners=False)
    tgt_l = F.interpolate(tgt_l, (h, w), mode='bilinear', align_corners=False)
    tgt_r = F.interpolate(tgt_r, (h, w), mode='bilinear', align_corners=False)

    cols = [norm(t) for t in [src_l, src_r, gen_l, gen_r, tgt_l, tgt_r]]
    # Interleave: sample0_col0, sample0_col1... then sample1_col0...
    tiles = []
    for i in range(n):
        for c in cols:
            tiles.append(c[i])
    grid = make_grid(tiles, nrow=len(cols), padding=2)
    return grid  # [3, H', W']


def pretrain_adapter_on_val(model, gaze_mlp, cfg, device, args, lpips_fn=None, writer: SummaryWriter = None):
    """Pre-train a shared SubjectAdapter on all val subjects.

    Loss: L1 + LPIPS + Multi-Scale GAN (adversarial on 6-ch eye crops).
    Trains adapter (encoder + block_modulators) + a fresh discriminator jointly.

    Returns:
        pretrained_state: adapter state_dict (on CPU)
    """
    adapter = create_adapter(model, cfg, device)
    adapter.train()

    # Multi-Scale Discriminator: 6-ch eye crops, two scales (1.0, 0.5)
    disc = MultiScaleDiscriminator(
        scales=([1]),
        num_channels=6,
        block_expansion=32,
        num_blocks=4,
        max_features=256,
        sn=True,
    ).to(device)
    disc.train()
    opt_D = torch.optim.AdamW(disc.parameters(), lr=args.pretrain_lr * 2,
                               betas=(0.0, 0.999), weight_decay=0.0)

    val_ds = HDFDataset(
        cfg.data.hdf_path,
        split='val',
        split_ratio=cfg.data.split_ratio,
        seed=cfg.seed,
        frame_offset_range=cfg.data.get('frame_offset_range', 2),
        eye_crop_size=cfg.data.get('eye_crop_size', [64, 64]),
        eye_expand_ratio=cfg.data.get('eye_expand_ratio', 1.5),
        input_eye_crop_size=cfg.data.get('input_eye_crop_size', None),
        input_eye_expand_ratio=cfg.data.get('input_eye_expand_ratio', None),
    )
    logger.info(f"Val dataset for adapter pre-training: {len(val_ds)} frames")

    dataloader = DataLoader(
        val_ds, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers,
        pin_memory=True, drop_last=True,
    )

    optimizer = torch.optim.AdamW(
        [
            {'params': adapter.encoder.parameters(),          'lr': args.pretrain_lr},
            {'params': adapter.block_modulators.parameters(), 'lr': args.pretrain_lr * 5},
        ],
        betas=(0.9, 0.999), weight_decay=args.weight_decay,
    )
    lr_sched = get_scheduler(
        "cosine", optimizer=optimizer,
        num_warmup_steps=min(100, args.pretrain_steps // 10),
        num_training_steps=args.pretrain_steps,
    )

    data_iter = iter(dataloader)
    for step in range(1, args.pretrain_steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        source_eye  = batch['source_input_eye_crops'].to(device)
        target_eye  = batch['target_eye_crops'].to(device)
        target_gaze = batch['target_gaze'].to(device)
        target_head = batch['target_head'].to(device)

        with torch.no_grad():
            head_emb, gaze_emb = gaze_mlp(target_head, target_gaze)
            gaze_prompt = torch.cat([head_emb.unsqueeze(1), gaze_emb.unsqueeze(1)], dim=1)

        generated = model(source_eye, gaze_prompt)

        in_h, in_w   = source_eye.shape[-2:]
        tgt_h, tgt_w = target_eye.shape[-2:]
        if in_h != tgt_h or in_w != tgt_w:
            ph = (in_h - tgt_h) // 2
            pw = (in_w - tgt_w) // 2
            generated_tight = generated[:, :, ph:ph+tgt_h, pw:pw+tgt_w]
        else:
            generated_tight = generated

        # ── D step ────────────────────────────────────────────────────────
        opt_D.zero_grad()
        real_pred = disc(target_eye.detach())
        fake_pred = disc(generated_tight.detach())
        loss_D = discriminator_loss(real_pred, fake_pred, device)
        loss_D.backward()
        opt_D.step()

        # ── G step ────────────────────────────────────────────────────────
        # L1
        l1 = F.l1_loss(generated_tight, target_eye)
        loss_G = cfg.loss_params.l1_loss * l1

        # LPIPS
        lpips_val = None
        if lpips_fn is not None and args.lpips_weight > 0:
            gen_l = F.interpolate(generated_tight[:, :3], args.lpips_size, mode='bilinear', align_corners=False)
            gen_r = F.interpolate(generated_tight[:, 3:], args.lpips_size, mode='bilinear', align_corners=False)
            tgt_l = F.interpolate(target_eye[:, :3],      args.lpips_size, mode='bilinear', align_corners=False)
            tgt_r = F.interpolate(target_eye[:, 3:],      args.lpips_size, mode='bilinear', align_corners=False)
            lpips_val = (lpips_fn(gen_l, tgt_l).mean() + lpips_fn(gen_r, tgt_r).mean()) * 0.5
            loss_G = loss_G + args.lpips_weight * lpips_val

        # GAN (generator)
        fake_pred_G = disc(generated_tight)
        loss_gan = generator_loss(fake_pred_G, device)
        loss_G = loss_G + args.gan_weight * loss_gan

        optimizer.zero_grad()
        loss_G.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        optimizer.step()
        lr_sched.step()

        if writer is not None:
            writer.add_scalar("pretrain/loss_total", loss_G.item(), step)
            writer.add_scalar("pretrain/loss_l1",    l1.item(),     step)
            writer.add_scalar("pretrain/loss_D",     loss_D.item(), step)
            writer.add_scalar("pretrain/loss_gan",   loss_gan.item(), step)
            if lpips_val is not None:
                writer.add_scalar("pretrain/loss_lpips", lpips_val.item(), step)
            writer.add_scalar("pretrain/lr", lr_sched.get_last_lr()[0], step)

        if writer is not None and step % args.vis_every == 0:
            with torch.no_grad():
                grid = _make_eye_grid(source_eye, generated_tight.detach(), target_eye)
            writer.add_image("pretrain/src_gen_tgt", grid, step)

        if step % args.log_every == 0:
            lpips_str = f"  lpips={lpips_val.item():.4f}" if lpips_val is not None else ""
            logger.info(
                f"  [pretrain] step {step}/{args.pretrain_steps}"
                f"  l1={l1.item():.4f}{lpips_str}"
                f"  gan={loss_gan.item():.4f}  D={loss_D.item():.4f}"
                f"  lr={lr_sched.get_last_lr()[0]:.2e}"
            )

        if args.save_every > 0 and step % args.save_every == 0:
            ckpt_path = os.path.join(args.output_dir, f"pretrain_step{step}.pth")
            torch.save({'step': step,
                        'subject_adapter_state_dict': {k: v.cpu().clone()
                                                       for k, v in adapter.state_dict().items()}},
                       ckpt_path)
            logger.info(f"  [pretrain] checkpoint saved → {ckpt_path}")

    pretrained_state = {k: v.cpu().clone() for k, v in adapter.state_dict().items()}
    model.subject_adapter = None
    logger.info(f"Adapter pre-training done ({args.pretrain_steps} steps on val set)")
    return pretrained_state


def finetune_one_subject(
    model, gaze_mlp, cfg, subject_key, device, args,
    lpips_fn=None, id_loss_fn=None, writer: SummaryWriter = None,
    init_state=None, save_dir=None,
):
    """Fine-tune a SubjectAdapter for one subject. Returns adapter state_dict.

    Args:
        init_state: optional state_dict from pretrain_adapter_on_val().
                    If provided, adapter starts from this instead of zero-init.
        save_dir:   directory to save latest/best checkpoints during training.
                    If None, checkpoints are only kept in memory.
    """
    adapter = create_adapter(model, cfg, device)
    if init_state is not None:
        adapter.load_state_dict(init_state)
        logger.info(f"  [{subject_key}] Loaded pre-trained adapter init")
    adapter.train()

    # Dataset: only this subject's frames
    ds_kwargs = dict(
        split_ratio=cfg.data.split_ratio,
        seed=cfg.seed,
        frame_offset_range=cfg.data.get('frame_offset_range', 2),
        eye_crop_size=cfg.data.get('eye_crop_size', [64, 64]),
        eye_expand_ratio=cfg.data.get('eye_expand_ratio', 1.5),
        input_eye_crop_size=cfg.data.get('input_eye_crop_size', None),
        input_eye_expand_ratio=cfg.data.get('input_eye_expand_ratio', None),
    )
    # Use 'train' split but filter to only this subject
    # (the split doesn't matter when prefixes is given — we just need the subject's frames)
    train_ds = HDFDataset(
        cfg.data.hdf_path, prefixes=[subject_key], split='train', **ds_kwargs
    )
    if len(train_ds) == 0:
        # Subject might only exist in val/test split — try without split filtering
        train_ds = HDFDataset(
            cfg.data.hdf_path, prefixes=[subject_key], split='val', **ds_kwargs
        )
    if len(train_ds) == 0:
        train_ds = HDFDataset(
            cfg.data.hdf_path, prefixes=[subject_key], split='test', **ds_kwargs
        )
    if len(train_ds) == 0:
        logger.warning(f"Subject {subject_key}: no frames found, skipping")
        return None

    dataloader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers,
        pin_memory=True, drop_last=len(train_ds) > args.batch_size,
    )

    # Optimizer: two param groups — block_modulators (zero-init) get higher LR
    optimizer = torch.optim.AdamW(
        [
            {'params': adapter.encoder.parameters(),          'lr': args.lr},
            {'params': adapter.block_modulators.parameters(), 'lr': args.lr * 5},
        ],
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )

    lr_sched = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=min(50, args.max_steps // 10),
        num_training_steps=args.max_steps,
    )

    # Training loop
    global_step = 0
    best_loss = float('inf')
    best_state = None
    data_iter = iter(dataloader)

    while global_step < args.max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        source_eye = batch['source_input_eye_crops'].to(device)
        source_image = batch['source_image'].to(device)
        source_bbox = batch['source_eye_bbox'].to(device)
        target_eye = batch['target_eye_crops'].to(device)
        target_image = batch['target_image'].to(device)
        target_gaze = batch['target_gaze'].to(device)
        target_head = batch['target_head'].to(device)

        # Gaze embedding (frozen)
        with torch.no_grad():
            head_emb, gaze_emb = gaze_mlp(target_head, target_gaze)
            gaze_prompt = torch.cat([head_emb.unsqueeze(1), gaze_emb.unsqueeze(1)], dim=1)

        # Forward (eye_unet frozen, adapter trainable)
        generated = model(source_eye, gaze_prompt)

        # Center-crop
        in_h, in_w = source_eye.shape[-2:]
        tgt_h, tgt_w = target_eye.shape[-2:]
        if in_h != tgt_h or in_w != tgt_w:
            ph = (in_h - tgt_h) // 2
            pw = (in_w - tgt_w) // 2
            generated_tight = generated[:, :, ph:ph+tgt_h, pw:pw+tgt_w]
        else:
            generated_tight = generated

        # Loss: L1 on tight crop (appearance + implicit gaze supervision)
        l1 = F.l1_loss(generated_tight, target_eye)
        loss = cfg.loss_params.l1_loss * l1

        # LPIPS: resize each eye separately, then average
        lpips_val = None
        if lpips_fn is not None and args.lpips_weight > 0:
            gen_l = F.interpolate(generated_tight[:, :3], args.lpips_size, mode='bilinear', align_corners=False)
            gen_r = F.interpolate(generated_tight[:, 3:], args.lpips_size, mode='bilinear', align_corners=False)
            tgt_l = F.interpolate(target_eye[:, :3],      args.lpips_size, mode='bilinear', align_corners=False)
            tgt_r = F.interpolate(target_eye[:, 3:],      args.lpips_size, mode='bilinear', align_corners=False)
            lpips_val = (lpips_fn(gen_l, tgt_l).mean() + lpips_fn(gen_r, tgt_r).mean()) * 0.5
            loss = loss + args.lpips_weight * lpips_val

        # ID loss: paste generated eyes back onto source face, compare with target face
        # (consistent with task: source_eye + target_gaze → should match target)
        id_val = None
        if id_loss_fn is not None and args.id_weight > 0:
            pasted = model.paste_eyes(generated_tight, source_image, source_bbox)  # [B,3,H,W]
            id_val = id_loss_fn.loss(pasted, target_image)
            loss = loss + args.id_weight * id_val

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        optimizer.step()
        lr_sched.step()

        global_step += 1

        # TensorBoard: scalars every step
        if writer is not None:
            writer.add_scalar("train/loss_total", loss.item(), global_step)
            writer.add_scalar("train/loss_l1", l1.item(), global_step)
            if lpips_val is not None:
                writer.add_scalar("train/loss_lpips", lpips_val.item(), global_step)
            if id_val is not None:
                writer.add_scalar("train/loss_id", id_val.item(), global_step)
            writer.add_scalar("train/lr", lr_sched.get_last_lr()[0], global_step)

        # TensorBoard: image grid
        if writer is not None and global_step % args.vis_every == 0:
            with torch.no_grad():
                grid = _make_eye_grid(source_eye, generated_tight.detach(), target_eye)
            writer.add_image("vis/src_gen_tgt", grid, global_step)

        # Track best & periodic save
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k: v.cpu().clone() for k, v in adapter.state_dict().items()}
            if save_dir is not None:
                torch.save({'subject_key': subject_key, 'step': global_step,
                            'subject_adapter_state_dict': best_state},
                           os.path.join(save_dir, f"{subject_key}_best.pth"))

        if save_dir is not None and global_step % args.save_every == 0:
            latest_state = {k: v.cpu().clone() for k, v in adapter.state_dict().items()}
            torch.save({'subject_key': subject_key, 'step': global_step,
                        'subject_adapter_state_dict': latest_state},
                       os.path.join(save_dir, f"{subject_key}_latest.pth"))

        if global_step % args.log_every == 0:
            lpips_str = f"  lpips={lpips_val.item():.4f}" if lpips_val is not None else ""
            id_str = f"  id={id_val.item():.4f}" if id_val is not None else ""
            logger.info(
                f"  [{subject_key}] step {global_step}/{args.max_steps}  "
                f"l1={l1.item():.4f}{lpips_str}{id_str}  total={loss.item():.4f}  "
                f"lr={lr_sched.get_last_lr()[0]:.2e}"
            )

    # Detach adapter from model (clean up for next subject)
    model.subject_adapter = None

    return best_state


def main():
    parser = argparse.ArgumentParser(description="FAGE Per-User SubjectAdapter Fine-Tuning")
    parser.add_argument("--config", type=str, required=True, help="Phase 1 config YAML")
    parser.add_argument("--checkpoint", type=str, required=True, help="Phase 1 checkpoint .pth")
    parser.add_argument("--output_dir", type=str, default="./adapter_output",
                        help="Output directory for adapter weights")

    # Subject selection (mutually exclusive)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--subjects", type=str, nargs="+",
                       help="Subject key(s) to fine-tune (e.g., 00002 00123)")
    group.add_argument("--split", type=str, choices=["train", "val", "test"],
                       help="Fine-tune all subjects in this split")

    # Training hyperparameters
    parser.add_argument("--max_steps", type=int, default=500, help="Steps per subject")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (block_modulators use 5x)")
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lpips_weight", type=float, default=0.1,
                        help="Weight for LPIPS perceptual loss (0 to disable)")
    parser.add_argument("--gan_weight", type=float, default=0.1,
                        help="Weight for GAN adversarial loss (used in pretrain, 0 to disable)")
    parser.add_argument("--id_weight", type=float, default=0.1,
                        help="Weight for ArcFace ID loss on pasted full face (0 to disable)")
    parser.add_argument("--lpips_size", type=int, nargs=2, default=[128, 128],
                        metavar=("H", "W"), help="Resize each eye to this size before LPIPS")
    parser.add_argument("--pretrain_steps", type=int, default=0,
                        help="Steps to pre-train adapter on full val set before per-user fine-tune (0=disabled)")
    parser.add_argument("--pretrain_lr", type=float, default=5e-5,
                        help="LR for val-set pre-training phase")
    parser.add_argument("--vis_every", type=int, default=100,
                        help="TensorBoard image visualization interval (steps)")
    parser.add_argument("--log_every", type=int, default=50,
                        help="Logging interval (steps)")
    parser.add_argument("--save_every", type=int, default=500,
                        help="Checkpoint save interval (steps). 0 to disable periodic saves")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    args.lpips_size = tuple(args.lpips_size)  # list → tuple for F.interpolate

    seed_everything(args.seed)
    device = torch.device(args.device)

    # Config
    cfg = OmegaConf.load(args.config)

    # Output directory
    adapter_dir = os.path.join(args.output_dir, "adapters")
    tb_dir = os.path.join(args.output_dir, "tensorboard")
    os.makedirs(adapter_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)

    # Load Phase 1 (frozen)
    model, gaze_mlp = load_phase1(cfg, args.checkpoint, device)

    # LPIPS perceptual loss (appearance supervision on eye crops)
    lpips_fn = None
    if args.lpips_weight > 0:
        lpips_fn = lpips.LPIPS(net='vgg').to(device)
        lpips_fn.eval()
        for p in lpips_fn.parameters():
            p.requires_grad = False
        logger.info(f"LPIPS (VGG) loaded, weight={args.lpips_weight}, resize={args.lpips_size}")

    # ArcFace ID loss (identity preservation on pasted full face)
    id_loss_fn = None
    if args.id_weight > 0:
        id_loss_fn = IDLoss().to(device)
        id_loss_fn.eval()
        logger.info(f"IDLoss (ArcFace IR-SE50) loaded, weight={args.id_weight}")

    # Determine subjects to fine-tune
    if args.subjects:
        subject_keys = args.subjects
    else:
        subject_keys = get_subject_keys(cfg, args.split)

    # Optional: pre-train adapter on full val set for a better initialization
    init_state = None
    if args.pretrain_steps > 0:
        logger.info(f"Pre-training adapter on val set for {args.pretrain_steps} steps...")
        pretrain_writer = SummaryWriter(log_dir=os.path.join(tb_dir, "_pretrain"))
        init_state = pretrain_adapter_on_val(
            model, gaze_mlp, cfg, device, args,
            lpips_fn=lpips_fn, writer=pretrain_writer,
        )
        pretrain_writer.close()

    logger.info(f"Fine-tuning {len(subject_keys)} subjects, {args.max_steps} steps each")

    logger.info(f"Adapter output: {adapter_dir}")
    logger.info(f"TensorBoard logs: {tb_dir}  (run: tensorboard --logdir {tb_dir})")

    # Fine-tune each subject
    for i, subj in enumerate(subject_keys):
        logger.info(f"[{i+1}/{len(subject_keys)}] Subject: {subj}")

        writer = SummaryWriter(log_dir=os.path.join(tb_dir, subj))
        state = finetune_one_subject(
            model, gaze_mlp, cfg, subj, device, args,
            lpips_fn=lpips_fn, id_loss_fn=id_loss_fn, writer=writer,
            init_state=init_state, save_dir=adapter_dir,
        )
        writer.close()

        if state is None:
            continue

        # Save adapter weights
        save_path = os.path.join(adapter_dir, f"{subj}.pth")
        torch.save({
            'subject_key': subj,
            'subject_adapter_state_dict': state,
            'config': {
                'in_channels': cfg.dic_unet_params.get('in_channels', 6),
                'subject_dim': 128,
                'max_steps': args.max_steps,
                'lr': args.lr,
            },
        }, save_path)

        size_kb = os.path.getsize(save_path) / 1024
        logger.info(f"  Saved: {save_path} ({size_kb:.0f} KB)")

    logger.info("Done.")


if __name__ == "__main__":
    main()
