"""
FAGE — Offline Validation Script

Reads config YAML, auto-detects the best checkpoint under output_dir/exp_name,
then evaluates on the full validation split and reports:
  L1 · LPIPS · PSNR · Gaze-error (angular °) · composite score

Usage:
    python validate.py --config configs/training/dic_eye_only.yaml
    python validate.py --config configs/training/dic_eye_only.yaml \\
        --checkpoint /path/to/checkpoint-50000.pth
    python validate.py --config configs/training/dic_eye_only.yaml \\
        --num_batches 20 --batch_size 8
"""
import os
import re
import json
import logging
import argparse

import torch
import torch.nn.functional as F
import lpips
from piq import psnr as piq_psnr
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm import tqdm

from models.gaze_dic import EyeOnlyWrapper
from models.gazenet import MLPNetwork
from dataset.gaze_capture import HDFDataset
from loss.basic_loss import gaze_angular_loss
from gaze_estimation.baseline_resnet import GazeHeadResNet

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Offline validation for FAGE eye-only generation")
    p.add_argument("--config", required=True, help="Path to training config YAML")
    p.add_argument(
        "--checkpoint", default=None,
        help="Override checkpoint path. Default: auto-detect best_model.pth "
             "or latest checkpoint-N.pth in output_dir/exp_name",
    )
    p.add_argument(
        "--num_batches", type=int, default=None,
        help="Limit validation to N batches (default: full val set)",
    )
    p.add_argument(
        "--batch_size", type=int, default=None,
        help="Override batch size from config",
    )
    p.add_argument(
        "--device", default=None,
        help="torch device (default: cuda if available, else cpu)",
    )
    p.add_argument(
        "--output", default=None,
        help="Path to write JSON results. Default: <save_dir>/val_metrics.json",
    )
    p.add_argument(
        "--num_workers", type=int, default=None,
        help="DataLoader workers (default: from config)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------

def find_best_checkpoint(save_dir: str) -> str:
    """Return best_model.pth if present, otherwise the highest-step checkpoint."""
    best = os.path.join(save_dir, "best_model.pth")
    if os.path.isfile(best):
        logger.info(f"Auto-detected best_model.pth: {best}")
        return best

    pattern = re.compile(r"checkpoint-(\d+)\.pth$")
    candidates = [
        (int(m.group(1)), os.path.join(save_dir, f))
        for f in os.listdir(save_dir)
        if (m := pattern.match(f))
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No best_model.pth or checkpoint-*.pth found in: {save_dir}"
        )
    candidates.sort(key=lambda x: x[0])
    step, path = candidates[-1]
    logger.info(f"Auto-detected latest checkpoint (step={step}): {path}")
    return path


# ---------------------------------------------------------------------------
# Model construction + weight loading
# ---------------------------------------------------------------------------

def build_models(cfg, ckpt_path: str, device: torch.device):
    unet_config = OmegaConf.to_container(cfg.dic_unet_params, resolve=True)
    subject_adapter_config = None
    if 'subject_adapter_params' in cfg and cfg.subject_adapter_params is not None:
        subject_adapter_config = OmegaConf.to_container(cfg.subject_adapter_params, resolve=True)
    model = EyeOnlyWrapper(unet_config, subject_adapter_config=subject_adapter_config).to(device)

    gp = cfg.model_params.gazenet_params
    gaze_dim = cfg.dic_unet_params.get("gaze_dim", 64)
    gaze_mlp = MLPNetwork(
        num_in=gp.num_in,
        num_hidden=gp.num_hidden,
        num_out=gaze_dim,
        num_layers=gp.num_layers,
        cross_condition=gp.get("cross_condition", False),
    ).to(device)

    logger.info(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    model.load_state_dict(ckpt["model_state_dict"])
    gaze_mlp.load_state_dict(ckpt["gaze_mlp_state_dict"])

    train_step = ckpt.get("global_step", "?")
    logger.info(f"Checkpoint step: {train_step}")
    if "metrics" in ckpt:
        m = ckpt["metrics"]
        logger.info(
            f"Metrics recorded at save time — "
            f"l1={m.get('l1', '?'):.4f}  "
            f"lpips={m.get('lpips', '?'):.4f}  "
            f"psnr={m.get('psnr', '?'):.2f}dB  "
            f"gaze={m.get('gaze_error', '?'):.4f}°"
        )

    model.eval()
    gaze_mlp.eval()
    return model, gaze_mlp, train_step


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def build_val_loader(cfg, batch_size: int, num_workers: int) -> DataLoader:
    d = cfg.data
    dataset = HDFDataset(
        hdf_file_path=d.hdf_path,
        split="val",
        split_ratio=list(d.split_ratio),
        frame_offset_range=d.get("frame_offset_range", 2),
        eye_crop_size=list(d.eye_crop_size),
        eye_expand_ratio=float(d.get("eye_expand_ratio", 1.5)),
        input_eye_crop_size=list(d.input_eye_crop_size),
        input_eye_expand_ratio=float(d.get("input_eye_expand_ratio", 2.0)),
        max_head_diff=d.get("max_head_diff", None),
        max_target_retries=int(d.get("max_target_retries", 10)),
    )
    logger.info(f"Val dataset: {len(dataset)} samples")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )


# ---------------------------------------------------------------------------
# Validation loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_validation(
    model, gaze_mlp, val_loader,
    lpips_fn, gaze_estimator, gaze_eval_trans,
    device, num_batches=None,
):
    total = {"l1": 0.0, "lpips": 0.0, "psnr": 0.0, "gaze": 0.0}
    count = 0

    pbar = tqdm(val_loader, desc="Validating", dynamic_ncols=True)
    for i, batch in enumerate(pbar):
        if num_batches is not None and i >= num_batches:
            break

        src      = batch["source_input_eye_crops"].to(device)   # [B,3,80,160]
        tgt      = batch["target_eye_crops"].to(device)          # [B,3,64,128]
        tgt_gaze = batch["target_gaze"].to(device)               # [B,2]
        tgt_head = batch["target_head"].to(device)               # [B,2]
        src_face = batch["source_image"].to(device)              # [B,3,256,256]

        # Forward
        h_emb, g_emb = gaze_mlp(tgt_head, tgt_gaze)
        gaze_prompt = torch.cat([h_emb.unsqueeze(1), g_emb.unsqueeze(1)], dim=1)  # [B,2,64]
        gen = model(src, gaze_prompt, src_face)                  # [B,3,80,160]

        # Center-crop generated output to supervision size (64×128)
        si_h, si_w = src.shape[-2:]
        tg_h, tg_w = tgt.shape[-2:]
        if si_h != tg_h or si_w != tg_w:
            ph = (si_h - tg_h) // 2
            pw = (si_w - tg_w) // 2
            gen = gen[:, :, ph:ph + tg_h, pw:pw + tg_w]

        # L1
        total["l1"] += F.l1_loss(gen, tgt).item()

        # LPIPS — width-concatenated eyes fed directly (matches training)
        total["lpips"] += lpips_fn(gen, tgt).mean().item()

        # PSNR — shift [-1,1] → [0,2], data_range=2
        total["psnr"] += piq_psnr(
            (gen.float() + 1).clamp(0, 2),
            (tgt.float() + 1).clamp(0, 2),
            data_range=2.0,
        ).item()

        # Gaze error — left-eye half: [-1,1]→[0,1] → resize 224 → ImageNet norm → estimator
        gen_left = (gen[:, :, :, : gen.shape[3] // 2].float() + 1) / 2
        gen_left = gaze_eval_trans(gen_left)
        gaze_pred, _ = gaze_estimator(gen_left)
        total["gaze"] += gaze_angular_loss(tgt_gaze.float(), gaze_pred).item()

        count += 1
        pbar.set_postfix({
            "l1":     f"{total['l1']  / count:.4f}",
            "lpips":  f"{total['lpips']/ count:.4f}",
            "psnr":   f"{total['psnr'] / count:.2f}",
            "gaze°":  f"{total['gaze'] / count:.4f}",
        })

    if count == 0:
        raise RuntimeError("No validation batches were processed.")

    return {
        "l1":          total["l1"]   / count,
        "lpips":       total["lpips"]/ count,
        "psnr":        total["psnr"] / count,
        "gaze_error":  total["gaze"] / count,
        "num_batches": count,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info(f"Device: {device}")

    cfg = OmegaConf.load(args.config)

    # Resolve save directory (mirrors train.py: output_dir/exp_name)
    save_dir = os.path.join(cfg.output_dir, cfg.exp_name)

    ckpt_path = args.checkpoint or find_best_checkpoint(save_dir)
    model, gaze_mlp, train_step = build_models(cfg, ckpt_path, device)

    batch_size  = args.batch_size  or int(cfg.data.train_bs)
    num_workers = args.num_workers or int(cfg.data.num_workers)
    val_loader  = build_val_loader(cfg, batch_size, num_workers)

    # Metric tools
    lpips_fn = lpips.LPIPS(net="alex").to(device).eval()
    for p in lpips_fn.parameters():
        p.requires_grad = False

    gaze_estimator = GazeHeadResNet(pretrained=False).to(device)
    ckpt_gaze = cfg.pretrained.gaze_eval_checkpoint_path
    if os.path.isfile(ckpt_gaze):
        _ckpt = torch.load(ckpt_gaze, map_location=device)
        gaze_estimator.load_state_dict(_ckpt["model_state_dict"])
        logger.info(f"Gaze estimator loaded: {ckpt_gaze}")
    else:
        logger.warning(f"Gaze estimator checkpoint not found: {ckpt_gaze}")
    gaze_estimator.eval()
    for p in gaze_estimator.parameters():
        p.requires_grad = False

    gaze_eval_trans = T.Compose([
        T.Resize((224, 224)),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    logger.info(
        f"Starting validation — "
        f"batches={args.num_batches or 'all'}, batch_size={batch_size}"
    )
    metrics = run_validation(
        model, gaze_mlp, val_loader,
        lpips_fn, gaze_estimator, gaze_eval_trans,
        device, num_batches=args.num_batches,
    )

    # Composite score (identical formula to train.py for direct comparability)
    score = (
        metrics["l1"]         / 0.05
        + metrics["lpips"]    / 0.1
        + 1.0 / max(metrics["psnr"], 1.0) / 0.03
        + metrics["gaze_error"] / 5.0
    )
    metrics["composite_score"] = score
    metrics["checkpoint"] = ckpt_path
    metrics["train_step"] = str(train_step)

    # Print table
    sep = "=" * 54
    print(f"\n{sep}")
    print(f"  Validation Results   (step {train_step})")
    print(sep)
    print(f"  L1            : {metrics['l1']:.6f}")
    print(f"  LPIPS         : {metrics['lpips']:.6f}")
    print(f"  PSNR          : {metrics['psnr']:.4f} dB")
    print(f"  Gaze Error    : {metrics['gaze_error']:.4f} °")
    print(f"  Composite     : {score:.4f}  (lower is better)")
    print(f"  Batches used  : {metrics['num_batches']}")
    print(f"  Checkpoint    : {os.path.basename(ckpt_path)}")
    print(sep)

    out_path = args.output or os.path.join(save_dir, "val_metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Results saved → {out_path}")


if __name__ == "__main__":
    main()
