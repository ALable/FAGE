"""
FAGE MPIIGaze Inference — Baseline vs Per-User Adapter Comparison

MPIIGaze 全部 subjects (p00-p14) 均用于评测 finetune 效果，不区分 train/val/test。

对每个 subject 分别运行：
  (1) Baseline  : 无 adapter 的 GazeControlNet
  (2) +Adapter  : fine-tune 后的 per-user adapter

每帧输出：
  - 对比图  : Source | Baseline | +Adapter | GT
  - Circle sweep 两行对比图 (Baseline / +Adapter)

定量指标（per-subject 平均后汇总表）：
  - L1      : eye crop 像素误差 (↓ better)
  - LPIPS   : 感知相似度 (↓ better)
  - Angular°: GazeHeadResNet 估算 gaze 角度误差 (↓ better)

用法:
    # 对所有 subjects 对比（adapter 目录模式）
    python inference_mpiigaze.py \\
        --config  configs/training/dic_eye_only_mpiigaze.yaml \\
        --checkpoint /path/to/checkpoint.pth \\
        --adapter ./adapter_output_mpiigaze/adapters

    # 只看部分 subjects
    python inference_mpiigaze.py ... --subjects p03 p06 p12

    # 固定目标 gaze
    python inference_mpiigaze.py ... --target_gaze_deg 10 -15
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import sys
import argparse
import random
import math
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import h5py
import lpips as lpips_lib
from PIL import Image
from omegaconf import OmegaConf
from torchvision import transforms

matplotlib_imported = False
try:
    import matplotlib
    matplotlib.use("Agg")
    matplotlib_imported = True
except ImportError:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.gaze_dic import EyeOnlyWrapper
from models.gazenet import MLPNetwork
from dataset.gaze_capture import HDFDataset
from gaze_estimation.baseline_resnet import GazeHeadResNet
from loss.basic_loss import gaze_angular_loss


# ──────────────────────────────────────────────────────────
#  Image helpers
# ──────────────────────────────────────────────────────────

def denorm(t: torch.Tensor) -> np.ndarray:
    """[-1,1] tensor [C,H,W] → uint8 HWC numpy"""
    return ((t * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


def deg2rad(d: float) -> float:
    return d * math.pi / 180.0


def draw_gaze_arrow(img: np.ndarray, pitch: float, yaw: float,
                    radius: int = 28, color=(0, 230, 0), thickness: int = 2) -> np.ndarray:
    """右下角圆盘+箭头标注 gaze 方向。"""
    img = img.copy()
    H, W = img.shape[:2]
    cx, cy = W - radius - 6, H - radius - 6
    overlay = img.copy()
    cv2.circle(overlay, (cx, cy), radius, (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    cv2.circle(img, (cx, cy), radius, (180, 180, 180), 1, cv2.LINE_AA)
    L = radius * 0.82
    dx = max(-radius, min(radius, int(-yaw   / 0.5 * L)))
    dy = max(-radius, min(radius, int(-pitch / 0.5 * L)))
    cv2.arrowedLine(img, (cx, cy), (cx + dx, cy + dy),
                    color, thickness, cv2.LINE_AA, tipLength=0.35)
    return img


def add_label(img: np.ndarray, text: str, h: int = 22) -> np.ndarray:
    bar = np.full((h, img.shape[1], 3), 240, dtype=np.uint8)
    cv2.putText(bar, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (30, 30, 30), 1, cv2.LINE_AA)
    return np.concatenate([img, bar], axis=0)


def hstack(imgs, gap: int = 4) -> np.ndarray:
    div = np.full((imgs[0].shape[0], gap, 3), 200, dtype=np.uint8)
    out = imgs[0]
    for p in imgs[1:]:
        out = np.concatenate([out, div, p], axis=1)
    return out


def vstack(imgs, gap: int = 4) -> np.ndarray:
    div = np.full((gap, imgs[0].shape[1], 3), 200, dtype=np.uint8)
    out = imgs[0]
    for p in imgs[1:]:
        out = np.concatenate([out, div, p], axis=0)
    return out


# ──────────────────────────────────────────────────────────
#  Model loading
# ──────────────────────────────────────────────────────────

def load_generation_model(cfg, checkpoint_path: str, device: torch.device,
                          with_adapter: bool = False):
    unet_config = OmegaConf.to_container(cfg.dic_unet_params, resolve=True)
    subject_adapter_config = None
    if with_adapter:
        if 'subject_adapter_params' in cfg and cfg.subject_adapter_params is not None:
            subject_adapter_config = OmegaConf.to_container(cfg.subject_adapter_params, resolve=True)
        else:
            subject_adapter_config = {'in_channels': unet_config.get('in_channels', 6), 'subject_dim': 128}
    model = EyeOnlyWrapper(unet_config, subject_adapter_config=subject_adapter_config)

    gp = cfg.model_params.gazenet_params
    gaze_mlp = MLPNetwork(num_in=gp.num_in, num_hidden=gp.num_hidden,
                          num_out=cfg.dic_unet_params.get("gaze_dim", 64),
                          num_layers=gp.num_layers,
                          cross_condition=gp.get("cross_condition", False))

    state = torch.load(checkpoint_path, map_location="cpu")
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"], strict=False)
    elif "unet_state_dict" in state:
        model.eye_unet.load_state_dict(state["unet_state_dict"])
    else:
        raise KeyError(f"No model weights. Keys: {list(state.keys())}")
    if "gaze_mlp_state_dict" in state:
        gaze_mlp.load_state_dict(state["gaze_mlp_state_dict"])

    model.to(device).eval()
    gaze_mlp.to(device).eval()
    n = sum(p.numel() for p in model.parameters())
    print(f"  Model loaded: {n:,} params  adapter={'on' if with_adapter else 'off'}")
    return model, gaze_mlp


def swap_adapter(model, adapter_path: str) -> bool:
    if not os.path.exists(adapter_path):
        return False
    d = torch.load(adapter_path, map_location="cpu")
    state = d.get('subject_adapter_state_dict', d)
    model.subject_adapter.load_state_dict(state)
    return True


def load_gaze_estimator(ckpt_path: str, device: torch.device):
    est = GazeHeadResNet(pretrained=False).to(device)
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        est.load_state_dict(ckpt['model_state_dict'])
        print(f"  GazeEstimator loaded: {ckpt_path}")
        est.eval()
        return est
    print(f"  [warn] GazeEstimator not found: {ckpt_path}  angular = N/A")
    return None


# ──────────────────────────────────────────────────────────
#  Inference
# ──────────────────────────────────────────────────────────

_eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


@torch.no_grad()
def run_inference(model, gaze_mlp, src_batch, tgt_gaze_np, tgt_head_np, cfg, device):
    """Returns (pasted_face_np [HWC uint8], eye_tight [1,3,H,W*2] tensor)"""
    eye_crop_size = list(cfg.data.get("eye_crop_size", [64, 64]))
    source_eye  = src_batch["source_input_eye_crops"].unsqueeze(0).to(device)
    source_img  = src_batch["source_image"].unsqueeze(0).to(device)
    source_bbox = src_batch["source_eye_bbox"].unsqueeze(0).to(device)

    tgt_gaze = torch.tensor(tgt_gaze_np, dtype=torch.float32).unsqueeze(0).to(device)
    tgt_head = torch.tensor(tgt_head_np, dtype=torch.float32).unsqueeze(0).to(device)
    head_emb, gaze_emb = gaze_mlp(tgt_head, tgt_gaze)
    gaze_prompt = torch.cat([head_emb.unsqueeze(1), gaze_emb.unsqueeze(1)], dim=1)

    generated = model(source_eye, gaze_prompt)
    in_h, in_w = source_eye.shape[-2:]
    # eye_crop_size specifies per-eye [H, W]; data is width-concatenated so actual W is *2
    tgt_h, tgt_w = eye_crop_size[0], eye_crop_size[1] * 2
    if in_h != tgt_h or in_w != tgt_w:
        ph, pw = (in_h - tgt_h) // 2, (in_w - tgt_w) // 2
        eye_tight = generated[:, :, ph:ph + tgt_h, pw:pw + tgt_w]
    else:
        eye_tight = generated

    pasted = model.paste_eyes(eye_tight, source_img, source_bbox)
    return denorm(pasted[0]), eye_tight


@torch.no_grad()
def compute_metrics(eye_tight, gt_eye, pasted_face, tgt_gaze_np,
                    gaze_estimator, lpips_fn, device):
    """Returns dict {l1, lpips, angular}. Values are floats or None."""
    gt_eye = gt_eye.to(device)
    l1 = F.l1_loss(eye_tight, gt_eye).item()

    lpips_val = None
    if lpips_fn is not None:
        # Both tensors are 3-ch width-concatenated [1,3,H,W*2]; split spatially
        w_half = eye_tight.shape[-1] // 2
        gl = F.interpolate(eye_tight[:, :, :, :w_half],  (64, 64), mode='bilinear', align_corners=False)
        gr = F.interpolate(eye_tight[:, :, :, w_half:],  (64, 64), mode='bilinear', align_corners=False)
        tl = F.interpolate(gt_eye[:, :, :, :w_half],     (64, 64), mode='bilinear', align_corners=False)
        tr = F.interpolate(gt_eye[:, :, :, w_half:],     (64, 64), mode='bilinear', align_corners=False)
        lpips_val = ((lpips_fn(gl, tl) + lpips_fn(gr, tr)) * 0.5).mean().item()

    angular_val = None
    if gaze_estimator is not None:
        face_01   = (pasted_face * 0.5 + 0.5).clamp(0, 1)
        face_norm = _eval_transform(face_01)
        gaze_pred, _ = gaze_estimator(face_norm)
        gt_t = torch.tensor(tgt_gaze_np, dtype=torch.float32).unsqueeze(0).to(device)
        angular_val = gaze_angular_loss(gt_t, gaze_pred).item()

    return {'l1': l1, 'lpips': lpips_val, 'angular': angular_val}


@torch.no_grad()
def circle_sweep(model, gaze_mlp, src_batch, head_np, cfg, device, n=8, r=0.3):  #0.8
    results = []
    for i in range(n):
        theta = 2 * math.pi * i / n
        g = np.array([r * math.sin(theta), r * math.cos(theta)], dtype=np.float32)
        face_np, _ = run_inference(model, gaze_mlp, src_batch, g, head_np, cfg, device)
        results.append(draw_gaze_arrow(face_np, g[0], g[1]))
    return results


# ──────────────────────────────────────────────────────────
#  Save comparison images
# ──────────────────────────────────────────────────────────

def save_compare(src, baseline, adapter, gt, save_path, m_base=None, m_adpt=None):
    """Source | Baseline | +Adapter | GT，底部标签含指标。"""
    def metric_str(m):
        if not m:
            return ""
        parts = []
        if m.get('l1')      is not None: parts.append(f"L1:{m['l1']:.3f}")
        if m.get('lpips')   is not None: parts.append(f"LP:{m['lpips']:.3f}")
        if m.get('angular') is not None: parts.append(f"Ag:{m['angular']:.1f}")
        return "  ".join(parts)

    cols = [
        add_label(src,      "Source"),
        add_label(baseline, f"Baseline  {metric_str(m_base)}"),
        add_label(adapter if adapter is not None else baseline,
                  f"+Adapter  {metric_str(m_adpt)}" if adapter is not None else "Baseline"),
    ]
    if gt is not None:
        cols.append(add_label(gt, "GT"))
    Image.fromarray(hstack(cols, gap=4)).save(save_path)


def save_circle_compare(src, base_sweeps, adpt_sweeps, save_path):
    H, lw, gap = src.shape[0], 76, 4
    div = np.full((H, gap, 3), 200, dtype=np.uint8)

    def row(label, sweeps):
        lbl = np.full((H, lw, 3), 240, dtype=np.uint8)
        cv2.putText(lbl, label, (4, H // 2 + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (30, 30, 30), 1, cv2.LINE_AA)
        r = np.concatenate([lbl, div, src], axis=1)
        for s in sweeps:
            r = np.concatenate([r, div, s], axis=1)
        return r

    rows = [row("Baseline", base_sweeps)]
    if adpt_sweeps is not None:
        rows.append(row("+Adapter", adpt_sweeps))
    Image.fromarray(vstack(rows, gap=4)).save(save_path)


# ──────────────────────────────────────────────────────────
#  Metrics table
# ──────────────────────────────────────────────────────────

def print_metrics_table(all_metrics: dict):
    W = 110
    print("\n" + "─" * W)
    print("  MPIIGAZE  Baseline vs +Adapter  (↓ = better)")
    print("─" * W)
    print(f"{'Subject':<10}  "
          f"{'L1-Base':>9}  {'L1-Adpt':>9}  {'ΔL1':>8}  "
          f"{'LP-Base':>9}  {'LP-Adpt':>9}  {'ΔLP':>8}  "
          f"{'Ag-Base':>8}  {'Ag-Adpt':>8}  {'ΔAg':>7}")
    print("─" * W)

    agg = {role: {k: [] for k in ('l1', 'lpips', 'angular')}
           for role in ('baseline', 'adapter')}

    def fv(v):
        return f"{v:.4f}" if v is not None else "    N/A "

    def fd(bv, av, fmt=".4f"):
        if bv is None or av is None:
            return "    N/A"
        d = av - bv
        return f"{'+' if d >= 0 else ''}{d:{fmt}}"

    for subj, m in sorted(all_metrics.items()):
        b, a = m['baseline'], m['adapter']
        print(f"{subj:<10}  "
              f"{fv(b['l1']):>9}  {fv(a['l1']):>9}  {fd(b['l1'],a['l1']):>8}  "
              f"{fv(b['lpips']):>9}  {fv(a['lpips']):>9}  {fd(b['lpips'],a['lpips']):>8}  "
              f"{fv(b['angular']):>8}  {fv(a['angular']):>8}  "
              f"{fd(b['angular'],a['angular'],'.2f'):>7}")
        for role, role_m in (('baseline', b), ('adapter', a)):
            for k in ('l1', 'lpips', 'angular'):
                if role_m[k] is not None:
                    agg[role][k].append(role_m[k])

    def mn(lst):
        return float(np.mean(lst)) if lst else None

    b_mn = {k: mn(v) for k, v in agg['baseline'].items()}
    a_mn = {k: mn(v) for k, v in agg['adapter'].items()}
    print("─" * W)
    print(f"{'MEAN':<10}  "
          f"{fv(b_mn['l1']):>9}  {fv(a_mn['l1']):>9}  {fd(b_mn['l1'],a_mn['l1']):>8}  "
          f"{fv(b_mn['lpips']):>9}  {fv(a_mn['lpips']):>9}  {fd(b_mn['lpips'],a_mn['lpips']):>8}  "
          f"{fv(b_mn['angular']):>8}  {fv(a_mn['angular']):>8}  "
          f"{fd(b_mn['angular'],a_mn['angular'],'.2f'):>7}")
    print("─" * W)
    print("  Δ = +Adapter − Baseline   (negative means improvement)")
    print("─" * W + "\n")

    return b_mn, a_mn


def save_metrics_csv(all_metrics: dict, csv_path: str):
    with open(csv_path, "w") as f:
        f.write("subject,baseline_l1,adapter_l1,baseline_lpips,adapter_lpips,"
                "baseline_angular,adapter_angular\n")
        for subj, m in sorted(all_metrics.items()):
            b, a = m['baseline'], m['adapter']
            fv = lambda v: f"{v:.6f}" if v is not None else ""
            f.write(f"{subj},{fv(b['l1'])},{fv(a['l1'])},"
                    f"{fv(b['lpips'])},{fv(a['lpips'])},"
                    f"{fv(b['angular'])},{fv(a['angular'])}\n")
    print(f"Metrics saved → {csv_path}")


# ──────────────────────────────────────────────────────────
#  Dataset: load all MPIIGaze subjects, no split filtering
# ──────────────────────────────────────────────────────────

def load_all_subjects(cfg) -> HDFDataset:
    """Load HDFDataset covering all subjects in the HDF5 file (no train/val/test split)."""
    hdf_path = cfg.data.hdf_path
    with h5py.File(hdf_path, 'r', libver='latest', swmr=True) as f:
        all_keys = sorted(f.keys())
    print(f"  HDF5 subjects: {all_keys}")

    # split_ratio=[1,0,0] + split='train' → all subjects go to 'train'
    ds = HDFDataset(
        hdf_path,
        prefixes=list(all_keys),
        split='train',
        split_ratio=[1.0, 0.0, 0.0],
        seed=cfg.seed,
        frame_offset_range=cfg.data.get('frame_offset_range', 2),
        eye_crop_size=cfg.data.get('eye_crop_size', [64, 64]),
        eye_expand_ratio=cfg.data.get('eye_expand_ratio', 1.5),
        input_eye_crop_size=cfg.data.get('input_eye_crop_size', None),
        input_eye_expand_ratio=cfg.data.get('input_eye_expand_ratio', None),
    )
    return ds


# ───────────────────────────────────────────────────��──────
#  Main
# ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="FAGE MPIIGaze Inference: Baseline vs Finetuned Adapter")
    parser.add_argument("--config",     type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--adapter",    type=str, default=None,
                        help="Directory of per-user .pth adapter files")
    parser.add_argument("--subjects",   type=str, nargs="+", default=None,
                        help="Subjects to evaluate (default: all in HDF5)")
    parser.add_argument("--num_frames", type=int, default=8,
                        help="Frames per subject (default: 8)")
    parser.add_argument("--output_dir", type=str, default="./inference_mpiigaze")
    parser.add_argument("--target_gaze", type=float, nargs=2, default=None,
                        metavar=("PITCH_RAD", "YAW_RAD"))
    parser.add_argument("--target_gaze_deg", type=float, nargs=2, default=None,
                        metavar=("PITCH_DEG", "YAW_DEG"))
    parser.add_argument("--target_head", type=float, nargs=2, default=None)
    parser.add_argument("--no_circle",  action="store_true", help="Skip circle sweep")
    parser.add_argument("--no_metrics", action="store_true", help="Skip metric computation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    cfg    = OmegaConf.load(args.config)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Device: {device}")

    # ── Models ──────────────────────────────────────────
    print("\n[1/4] Loading generation models...")
    use_adapter = args.adapter is not None
    model_adpt, gaze_mlp = load_generation_model(cfg, args.checkpoint, device, with_adapter=use_adapter)
    model_base, _        = load_generation_model(cfg, args.checkpoint, device, with_adapter=False)

    # ── LPIPS + Gaze estimator ───────────────────────────
    lpips_fn       = None
    gaze_estimator = None
    if not args.no_metrics:
        print("[2/4] Loading LPIPS (VGG)...")
        lpips_fn = lpips_lib.LPIPS(net='vgg').to(device)
        lpips_fn.eval()
        for p in lpips_fn.parameters():
            p.requires_grad = False

        print("[3/4] Loading GazeHeadResNet...")
        gaze_estimator = load_gaze_estimator(cfg.pretrained.gaze_eval_checkpoint_path, device)

    # ── Dataset ──────────────────────────────────────────
    print("[4/4] Loading dataset (all subjects, no split)...")
    ds = load_all_subjects(cfg)

    target_subjects = args.subjects if args.subjects else sorted(ds.prefix_to_indices.keys())
    missing = [s for s in target_subjects if s not in ds.prefix_to_indices]
    if missing:
        print(f"  [warn] subjects not found: {missing}")
        target_subjects = [s for s in target_subjects if s in ds.prefix_to_indices]
    print(f"  Evaluating {len(target_subjects)} subjects: {target_subjects}")

    # ── Target gaze ──────────────────────────────────────
    if args.target_gaze_deg is not None:
        fixed_gaze = np.array([deg2rad(d) for d in args.target_gaze_deg], dtype=np.float32)
        print(f"Target gaze: {args.target_gaze_deg}° → {fixed_gaze} rad")
    elif args.target_gaze is not None:
        fixed_gaze = np.array(args.target_gaze, dtype=np.float32)
    else:
        fixed_gaze = None
        print("Target gaze: random frame within same subject")

    fixed_head = np.array(args.target_head, dtype=np.float32) if args.target_head else None

    # ── Per-subject loop ─────────────────────────────────
    all_metrics = {}

    for si, subject in enumerate(target_subjects):
        print(f"\n{'─'*60}")
        print(f"[{si+1}/{len(target_subjects)}] Subject: {subject}")

        subj_indices = ds.prefix_to_indices[subject]
        subj_dir     = os.path.join(args.output_dir, subject)
        os.makedirs(subj_dir, exist_ok=True)

        # Load adapter
        has_adapter = False
        if use_adapter:
            adapter_file = os.path.join(args.adapter, f"{subject}.pth")
            has_adapter  = swap_adapter(model_adpt, adapter_file)
            status = f"loaded: {adapter_file}" if has_adapter else f"NOT FOUND: {adapter_file}"
            print(f"  Adapter {status}")

        # Per-frame metric accumulators
        acc = {role: {'l1': [], 'lpips': [], 'angular': []}
               for role in ('baseline', 'adapter')}

        chosen = random.sample(subj_indices, min(args.num_frames, len(subj_indices)))

        for fi, src_idx in enumerate(chosen):
            src_batch = ds[src_idx]

            # ── Target gaze / head ──────────────────────
            if fixed_gaze is not None:
                tgt_gaze_np = fixed_gaze.copy()
                tgt_head_np = fixed_head if fixed_head is not None \
                              else src_batch["source_head"].numpy()
                gt_eye = None
                gt_np  = None
            else:
                others  = [i for i in subj_indices if i != src_idx]
                ref_idx = random.choice(others if others else subj_indices)
                ref     = ds[ref_idx]
                tgt_gaze_np = ref["source_gaze"].numpy()
                tgt_head_np = fixed_head if fixed_head is not None \
                              else ref["source_head"].numpy()
                gt_eye = ref["target_eye_crops"].unsqueeze(0)   # [1,6,H,W]
                gt_np  = denorm(ref["source_image"])

            # ── Baseline inference ──────────────────────
            base_np, base_tight = run_inference(
                model_base, gaze_mlp, src_batch, tgt_gaze_np, tgt_head_np, cfg, device)

            # ── Adapter inference ───────────────────────
            adpt_np, adpt_tight = (None, None)
            if has_adapter:
                adpt_np, adpt_tight = run_inference(
                    model_adpt, gaze_mlp, src_batch, tgt_gaze_np, tgt_head_np, cfg, device)

            # ── Metrics ─────────────────────────────────
            m_base = m_adpt = None
            if not args.no_metrics and gt_eye is not None:
                src_img  = src_batch["source_image"].unsqueeze(0).to(device)
                src_bbox = src_batch["source_eye_bbox"].unsqueeze(0).to(device)

                base_face = model_base.paste_eyes(base_tight, src_img, src_bbox)
                m_base = compute_metrics(base_tight, gt_eye, base_face,
                                         tgt_gaze_np, gaze_estimator, lpips_fn, device)
                for k, v in m_base.items():
                    if v is not None: acc['baseline'][k].append(v)

                if has_adapter and adpt_tight is not None:
                    adpt_face = model_adpt.paste_eyes(adpt_tight, src_img, src_bbox)
                    m_adpt = compute_metrics(adpt_tight, gt_eye, adpt_face,
                                              tgt_gaze_np, gaze_estimator, lpips_fn, device)
                    for k, v in m_adpt.items():
                        if v is not None: acc['adapter'][k].append(v)

                parts = [f"frame {fi+1:02d}",
                         f"base L1={m_base['l1']:.4f}",
                         f"LP={m_base['lpips']:.4f}" if m_base['lpips'] else "LP=N/A",
                         f"Ag={m_base['angular']:.2f}°" if m_base['angular'] else "Ag=N/A"]
                if m_adpt:
                    parts += ["|",
                               f"adpt L1={m_adpt['l1']:.4f}",
                               f"LP={m_adpt['lpips']:.4f}" if m_adpt['lpips'] else "LP=N/A",
                               f"Ag={m_adpt['angular']:.2f}°" if m_adpt['angular'] else "Ag=N/A"]
                print("  " + "  ".join(parts))

            # ── Annotate images ──────────────────────────
            src_gaze = src_batch["source_gaze"].numpy()
            src_np   = draw_gaze_arrow(denorm(src_batch["source_image"]),
                                       src_gaze[0], src_gaze[1], color=(255, 220, 0))
            base_np  = draw_gaze_arrow(base_np, tgt_gaze_np[0], tgt_gaze_np[1])
            if adpt_np is not None:
                adpt_np = draw_gaze_arrow(adpt_np, tgt_gaze_np[0], tgt_gaze_np[1])
            if gt_np is not None:
                gt_np = draw_gaze_arrow(gt_np, tgt_gaze_np[0], tgt_gaze_np[1])

            # ── Save comparison strip ────────────────────
            save_compare(src_np, base_np, adpt_np, gt_np,
                         os.path.join(subj_dir, f"frame{fi+1:02d}_compare.png"),
                         m_base=m_base, m_adpt=m_adpt)

            # ── Circle sweep ─────────────────────────────
            if not args.no_circle:
                head_np     = tgt_head_np
                base_sweeps = circle_sweep(model_base, gaze_mlp, src_batch,
                                           head_np, cfg, device)
                adpt_sweeps = (circle_sweep(model_adpt, gaze_mlp, src_batch,
                                            head_np, cfg, device)
                               if has_adapter else None)
                save_circle_compare(src_np, base_sweeps, adpt_sweeps,
                                    os.path.join(subj_dir, f"frame{fi+1:02d}_circle.png"))

        # ── Per-subject summary ───────────────────────────
        def mn(lst): return float(np.mean(lst)) if lst else None
        all_metrics[subject] = {
            'baseline': {k: mn(v) for k, v in acc['baseline'].items()},
            'adapter':  {k: mn(v) for k, v in acc['adapter'].items()},
        }
        b, a = all_metrics[subject]['baseline'], all_metrics[subject]['adapter']
        nf = len(chosen)
        fv = lambda v: f"{v:.4f}" if v is not None else "N/A"
        print(f"\n  [{subject}] ({nf} frames)")
        print(f"    Baseline : L1={fv(b['l1'])}  LPIPS={fv(b['lpips'])}  Angular={fv(b['angular'])}°")
        if has_adapter:
            print(f"    +Adapter : L1={fv(a['l1'])}  LPIPS={fv(a['lpips'])}  Angular={fv(a['angular'])}°")

    # ── Summary table + CSV ───────────────────────────────
    if all_metrics and not args.no_metrics:
        print_metrics_table(all_metrics)
        save_metrics_csv(all_metrics, os.path.join(args.output_dir, "metrics.csv"))

    print(f"\nDone. Results → {args.output_dir}")


if __name__ == "__main__":
    main()
