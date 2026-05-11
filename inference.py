"""
FAGE Inference — Gaze Redirection on Val Subjects

从 val split 随机抽取人物，给定 target gaze 方向（命令行指定或自动从另一帧取），
输出原图 vs 重定向结果（以及 GT）的对比图。

用法:
    # 随机 target gaze（从 val 中另一帧取）
    python inference.py --config configs/training/dic_eye_only.yaml \
                        --checkpoint /path/to/checkpoint.pth

    # 指定 target gaze（弧度）
    python inference.py ... --target_gaze 0.1 -0.3

    # 指定 target gaze（角度）
    python inference.py ... --target_gaze_deg 10.0 -15.0
"""
"""
val split
00003  00034  00104  00130  00145  00192  00310  00331  00358  00491
00494  00501  00503  00505  00512  00518  00569  00599  00606  00616
00634  00644  00650  00682  00693  00711  00743  00749  00771  00819
00875  00961  00997  01000  01002  01060  01065  01084  01100  01107
01119  01127  01151  01177  01188  01207  01247  01256  01266  01276
01327  01349  01386  01421  01428  01438  01456  01457  01459  01470
01473  01486  01487  01496  01532  01546  01584  01627  01710  01713
01718  01821  01825  01828  01883  01907  01964  02024  02028  02035
02065  02113  02156  02220  02232  02236  02240  02272  02297  02367
02370  02394  02450  02474  02478  02534  02540  02673  02729  02761
02773  02774  02785  02797  02829  02851  02873  02879  02882  02911
02944  02945  02967  02987  03012  03051  03064  03116  03125  03139
03140  03177  03185  03190  03232  03248  03251  03253  03259  03266
03312  03351  03367  03377  03413  0346
"""

import os
import sys
import argparse
import random
import math
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
from omegaconf import OmegaConf
import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.gaze_dic import EyeOnlyWrapper
from models.gazenet import MLPNetwork
from dataset.gaze_capture import HDFDataset


# ──────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────

def denorm(t: torch.Tensor) -> np.ndarray:
    """[-1,1] tensor → [0,1] numpy HWC uint8"""
    arr = (t * 0.5 + 0.5).clamp(0, 1)
    return (arr.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


def deg2rad(d: float) -> float:
    return d * math.pi / 180.0


def draw_gaze_arrow(img_np: np.ndarray, pitch: float, yaw: float,
                    radius: int = 28, color=(0, 230, 0), thickness: int = 2) -> np.ndarray:
    """在图像右下角叠加一个圆盘+箭头，指示 gaze 方向。

    GazeCapture 使用面部坐标系 (相机正视时为镜像):
        pitch:  正值=朝上看，负值=朝下看 (rad)
        yaw:    正值=面部右方（图像左方），负值=图像右方 (rad)
    屏幕坐标 y 轴朝下，因此需对两个分量取反映射到屏幕方向。
    """
    img = img_np.copy()
    H, W = img.shape[:2]
    margin = radius + 6
    cx, cy = W - margin, H - margin  # 右下角圆心

    # 背景圆盘（半透明黑）
    overlay = img.copy()
    cv2.circle(overlay, (cx, cy), radius, (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

    # 外圆边框
    cv2.circle(img, (cx, cy), radius, (180, 180, 180), 1, cv2.LINE_AA)

    # 箭头: pitch 正=上→dy 负，yaw 正=图像左→dx 负，均取反
    arrow_len = radius * 0.82
    dx = int(round(-yaw   / 0.5 * arrow_len))
    dy = int(round(-pitch / 0.5 * arrow_len))
    dx = max(-radius, min(radius, dx))
    dy = max(-radius, min(radius, dy))
    ex, ey = cx + dx, cy + dy

    cv2.arrowedLine(img, (cx, cy), (ex, ey),
                    color, thickness, cv2.LINE_AA, tipLength=0.35)
    return img


# ──────────────────────────────────────────────
#  Model loading
# ──────────────────────────────────────────────

def load_models(cfg, checkpoint_path: str, device: torch.device, with_adapter: bool = False):
    """Load Phase 1 model. with_adapter=True creates SubjectAdapter structure (weights loaded later)."""
    unet_config = OmegaConf.to_container(cfg.dic_unet_params, resolve=True)

    subject_adapter_config = None
    if with_adapter:
        if 'subject_adapter_params' in cfg and cfg.subject_adapter_params is not None:
            subject_adapter_config = OmegaConf.to_container(cfg.subject_adapter_params, resolve=True)
        else:
            subject_adapter_config = {'in_channels': unet_config.get('in_channels', 6), 'subject_dim': 128}

    model = EyeOnlyWrapper(unet_config, subject_adapter_config=subject_adapter_config)

    gaze_params = cfg.model_params.gazenet_params
    gaze_dim = cfg.dic_unet_params.get("gaze_dim", 64)
    gaze_mlp = MLPNetwork(
        num_in=gaze_params.num_in,
        num_hidden=gaze_params.num_hidden,
        num_out=gaze_dim,
        num_layers=gaze_params.num_layers,
        cross_condition=gaze_params.get('cross_condition', False),
    )
    state = torch.load(checkpoint_path, map_location="cpu")

    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"], strict=False)
    elif "unet_state_dict" in state:
        model.eye_unet.load_state_dict(state["unet_state_dict"])
    else:
        raise KeyError(f"No model weights found. Keys: {list(state.keys())}")

    if "gaze_mlp_state_dict" in state:
        gaze_mlp.load_state_dict(state["gaze_mlp_state_dict"])
    else:
        print("  [warn] gaze_mlp weights not found in checkpoint, using random init")

    # Load per-user SubjectAdapter weights (single file mode only; dir mode uses swap_adapter())
    model.to(device).eval()
    gaze_mlp.to(device).eval()

    def count_params(m):
        total     = sum(p.numel() for p in m.parameters())
        trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        return total, trainable

    m_total, m_train = count_params(model)
    g_total, g_train = count_params(gaze_mlp)
    all_total = m_total + g_total

    print(f"  EyeOnlyWrapper : {m_total:>10,} params  ({m_train:,} trainable)")
    print(f"  GazeMLP        : {g_total:>10,} params  ({g_train:,} trainable)")
    print(f"  Total          : {all_total:>10,} params")

    return model, gaze_mlp


def swap_adapter(model, adapter_path: str):
    """Load adapter weights from file into model.subject_adapter (in-place).

    Returns True on success, False if file not found.
    """
    if not os.path.exists(adapter_path):
        return False
    adapter_state = torch.load(adapter_path, map_location="cpu")
    if 'subject_adapter_state_dict' in adapter_state:
        model.subject_adapter.load_state_dict(adapter_state['subject_adapter_state_dict'])
    else:
        model.subject_adapter.load_state_dict(adapter_state)
    return True


def resolve_adapter(adapter_arg: str, subject_key: str):
    """Given --adapter (file or dir) and subject_key, return the adapter file path or None."""
    if adapter_arg is None:
        return None
    if os.path.isdir(adapter_arg):
        return os.path.join(adapter_arg, f"{subject_key}.pth")
    # Single file: use for all subjects
    return adapter_arg


# ──────────────────────────────────────────────
#  Inference for a single sample
# ──────────────────────────────────────────────

@torch.no_grad()
def run_inference(model, gaze_mlp, src_batch, tgt_gaze_np, tgt_head_np, cfg, device):
    """
    Returns:
        pasted_np:   HWC uint8  — redirected full face
        gen_left_np: HWC uint8  — generated left eye crop
        gen_right_np:HWC uint8  — generated right eye crop
    """
    eye_crop_size = list(cfg.data.get("eye_crop_size", [64, 64]))

    source_eye = src_batch["source_input_eye_crops"].unsqueeze(0).to(device)   # [1,6,in_h,in_w]
    source_img = src_batch["source_image"].unsqueeze(0).to(device)             # [1,3,H,W]
    source_bbox = src_batch["source_eye_bbox"].unsqueeze(0).to(device)         # [1,8]

    tgt_gaze = torch.tensor(tgt_gaze_np, dtype=torch.float32).unsqueeze(0).to(device)
    tgt_head = torch.tensor(tgt_head_np, dtype=torch.float32).unsqueeze(0).to(device)

    head_emb, gaze_emb = gaze_mlp(tgt_head, tgt_gaze)
    gaze_prompt = torch.cat([head_emb.unsqueeze(1), gaze_emb.unsqueeze(1)], dim=1)  # [1,2,gaze_dim]

    generated = model(source_eye, gaze_prompt)  # [1,6,in_h,in_w]

    # Center-crop to supervised (tight) size
    in_h, in_w = source_eye.shape[-2:]
    tgt_h, tgt_w = eye_crop_size
    if in_h != tgt_h or in_w != tgt_w:
        ph = (in_h - tgt_h) // 2
        pw = (in_w - tgt_w) // 2
        generated_tight = generated[:, :, ph:ph + tgt_h, pw:pw + tgt_w]
    else:
        generated_tight = generated

    pasted = model.paste_eyes(generated_tight, source_img, source_bbox)  # [1,3,H,W]

    pasted_np    = denorm(pasted[0])
    gen_left_np  = denorm(generated_tight[0, :3])
    gen_right_np = denorm(generated_tight[0, 3:])

    return pasted_np, gen_left_np, gen_right_np


@torch.no_grad()
def run_circle_sweep(model, gaze_mlp, src_batch, head_np, cfg, device,
                     n=8, radius=0.8):
    """在半径 radius (rad) 的圆上均匀采 n 个 gaze 方向，返回 n 张 pasted 图。

    面部坐标系中均匀采样: pitch=sin(θ), yaw=cos(θ), θ=2π*i/n
    箭头经 draw_gaze_arrow 取反后在图像中顺时针旋转: ←↖↑↗→↘↓↙
    """
    results = []
    for i in range(n):
        theta = 2 * math.pi * i / n
        gaze_np = np.array([radius * math.sin(theta),
                            radius * math.cos(theta)], dtype=np.float32)
        pasted_np, _, _ = run_inference(model, gaze_mlp, src_batch, gaze_np, head_np, cfg, device)
        pasted_np = draw_gaze_arrow(pasted_np, gaze_np[0], gaze_np[1])
        results.append(pasted_np)
    return results


def save_compare_strip(source_np, baseline_np, adapter_np, gt_np, save_path):
    """保存对比图: source | baseline | +adapter | GT(可选)

    每列下方附带文字标签。
    """
    gap = 4
    _, W = source_np.shape[:2]

    panels = [source_np, baseline_np, adapter_np]
    labels = ["Source", "Baseline", "+Adapter"]
    if gt_np is not None:
        panels.append(gt_np)
        labels.append("GT")

    import cv2 as _cv2
    label_h = 24
    labeled = []
    for img, lbl in zip(panels, labels):
        bar = np.full((label_h, W, 3), 240, dtype=np.uint8)
        _cv2.putText(bar, lbl, (4, 17), _cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1, _cv2.LINE_AA)
        labeled.append(np.concatenate([img, bar], axis=0))

    # divider 高度需与拼了标签后的实际列高一致
    col_h = labeled[0].shape[0]
    divider = np.full((col_h, gap, 3), 200, dtype=np.uint8)
    strip = labeled[0]
    for p in labeled[1:]:
        strip = np.concatenate([strip, divider, p], axis=1)

    Image.fromarray(strip).save(save_path)


# ──────────────────────────────────────────────
#  Visualization
# ──────────────────────────────────────────────

def save_grid(rows, output_path: str):
    """每行拼接 source | redirect | target，所有行纵向堆叠输出为一张图。"""
    has_gt = any("gt_np" in r for r in rows)
    gap = 4  # 列间白色分隔宽度 (px)

    strip_rows = []
    for row in rows:
        panels = [row["source_np"], row["pasted_np"]]
        if has_gt:
            panels.append(row.get("gt_np", np.zeros_like(row["source_np"])))

        H = panels[0].shape[0]
        divider = np.full((H, gap, 3), 255, dtype=np.uint8)
        strip = panels[0]
        for p in panels[1:]:
            strip = np.concatenate([strip, divider, p], axis=1)
        strip_rows.append(strip)

    # 行间白色分隔
    W = strip_rows[0].shape[1]
    row_gap = np.full((gap, W, 3), 255, dtype=np.uint8)
    canvas = strip_rows[0]
    for sr in strip_rows[1:]:
        canvas = np.concatenate([canvas, row_gap, sr], axis=0)

    Image.fromarray(canvas).save(output_path)
    print(f"  Grid saved: {output_path}")


# ──────────────────────────────────────────────
#  FPS Benchmark
# ──────────────────────────────────────────────

@torch.no_grad()
def benchmark_fps(model, gaze_mlp, cfg, device,
                  warmup: int = 50, iterations: int = 500):
    """测试 batch_size=1 时模型的推理 FPS。

    使用随机 dummy 输入，仅计时模型前向传播（gaze_mlp + UNet），
    不含数据加载和后处理。GPU 使用 cuda.synchronize() 保证计时准确。

    Args:
        model:      EyeOnlyWrapper，已 .eval()
        gaze_mlp:   MLPNetwork，已 .eval()
        cfg:        OmegaConf config
        device:     torch.device
        warmup:     预热迭代次数（不计入统计）
        iterations: 正式计时迭代次数

    Returns:
        dict: fps, mean_ms, std_ms, min_ms, max_ms
    """
    import time

    # ── 从配置读取输入尺寸 ──────────────────────
    in_channels   = cfg.dic_unet_params.get("in_channels", 6)
    gaze_dim      = cfg.dic_unet_params.get("gaze_dim", 64)
    inp_size      = list(cfg.data.get("input_eye_crop_size", [80, 80]))  # [H, W]

    # Dummy 输入 (batch_size=1)
    eye_input  = torch.randn(1, in_channels, inp_size[0], inp_size[1], device=device)
    tgt_gaze   = torch.randn(1, 2, device=device)   # [1, 2]
    tgt_head   = torch.randn(1, 2, device=device)   # [1, 2]

    is_cuda = device.type == "cuda"

    def sync():
        if is_cuda:
            torch.cuda.synchronize()

    def one_step():
        head_emb, gaze_emb = gaze_mlp(tgt_head, tgt_gaze)
        gaze_prompt = torch.cat(
            [head_emb.unsqueeze(1), gaze_emb.unsqueeze(1)], dim=1
        )                                            # [1, 2, gaze_dim]
        _ = model(eye_input, gaze_prompt)

    # ── 预热 ───────────────────────────────────
    print(f"[benchmark] Warming up ({warmup} iters)…", flush=True)
    for _ in range(warmup):
        one_step()
    sync()

    # ── 正式计时 ───────────────────────────────
    print(f"[benchmark] Timing {iterations} iters…", flush=True)
    latencies = []
    for _ in range(iterations):
        sync()
        t0 = time.perf_counter()
        one_step()
        sync()
        latencies.append((time.perf_counter() - t0) * 1e3)  # ms

    latencies = np.array(latencies)
    mean_ms = latencies.mean()
    std_ms  = latencies.std()
    min_ms  = latencies.min()
    max_ms  = latencies.max()
    fps     = 1000.0 / mean_ms

    print("\n" + "=" * 50)
    print("  Inference Benchmark  (batch_size=1)")
    print("=" * 50)
    print(f"  Device     : {device}")
    print(f"  Input shape: [{1}, {in_channels}, {inp_size[0]}, {inp_size[1]}]")
    print(f"  Iterations : {iterations}  (warmup={warmup})")
    print(f"  FPS        : {fps:.1f}")
    print(f"  Latency    : {mean_ms:.2f} ± {std_ms:.2f} ms  "
          f"[min={min_ms:.2f}  max={max_ms:.2f}]")
    print("=" * 50 + "\n")

    return dict(fps=fps, mean_ms=mean_ms, std_ms=std_ms,
                min_ms=min_ms, max_ms=max_ms)


# ──────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="FAGE Gaze Redirection Inference")
    parser.add_argument("--config",     type=str, required=True,  help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, required=True,  help="Path to .pth checkpoint")
    parser.add_argument("--adapter",   type=str, default=None,
                        help="Path to per-user SubjectAdapter weights (.pth) or directory of adapters")
    parser.add_argument("--subject",   type=str, default=None,
                        help="Run inference only for this subject key (auto-derived from --adapter filename if omitted)")
    parser.add_argument("--compare", action="store_true",
                        help="Side-by-side comparison: baseline vs +adapter (requires --adapter)")
    parser.add_argument("--output_dir", type=str, default="./inference_results")
    parser.add_argument("--num_samples", type=int, default=6,
                        help="Number of val subjects to randomly sample")
    parser.add_argument("--target_gaze", type=float, nargs=2, default=None,
                        metavar=("PITCH_RAD", "YAW_RAD"),
                        help="Target gaze in radians. If omitted, use a random val frame's gaze.")
    parser.add_argument("--target_gaze_deg", type=float, nargs=2, default=None,
                        metavar=("PITCH_DEG", "YAW_DEG"),
                        help="Target gaze in degrees (overrides --target_gaze).")
    parser.add_argument("--target_head", type=float, nargs=2, default=None,
                        metavar=("PITCH_RAD", "YAW_RAD"),
                        help="Target head pose in radians. Default: keep source head pose.")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run FPS benchmark (batch_size=1) and exit.")
    parser.add_argument("--benchmark_warmup",    type=int, default=50,
                        help="Warmup iterations for benchmark (default: 50).")
    parser.add_argument("--benchmark_iterations", type=int, default=500,
                        help="Timed iterations for benchmark (default: 500).")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Config & models ──────────────────────
    cfg = OmegaConf.load(args.config)
    device = torch.device(args.device)
    print(f"Device: {device}")

    adapter_is_dir = args.adapter is not None and os.path.isdir(args.adapter)
    use_adapter    = args.adapter is not None

    # 带 SubjectAdapter 结构的主模型（adapter 权重在 subject 循环中换入）
    model, gaze_mlp = load_models(cfg, args.checkpoint, device, with_adapter=use_adapter)

    # --compare: 同时加载 baseline（无 adapter，用于对比）
    model_baseline = None
    if args.compare:
        if not use_adapter:
            print("[warn] --compare requires --adapter; running single-model mode")
        else:
            model_baseline, _ = load_models(cfg, args.checkpoint, device, with_adapter=False)
            print("  Comparison mode: baseline vs +adapter per-subject")

    # ── FPS benchmark (optional early exit) ──
    if args.benchmark:
        benchmark_fps(model, gaze_mlp, cfg, device,
                      warmup=args.benchmark_warmup,
                      iterations=args.benchmark_iterations)
        return

    # ── Val dataset ──────────────────────────
    val_dataset = HDFDataset(
        cfg.data.hdf_path,
        split="val",
        split_ratio=cfg.data.split_ratio,
        seed=cfg.seed,
        frame_offset_range=cfg.data.get("frame_offset_range", 2),
        eye_crop_size=cfg.data.get("eye_crop_size", [64, 64]),
        eye_expand_ratio=cfg.data.get("eye_expand_ratio", 1.5),
        input_eye_crop_size=cfg.data.get("input_eye_crop_size", None),
        input_eye_expand_ratio=cfg.data.get("input_eye_expand_ratio", None),
    )

    subjects = list(val_dataset.prefix_to_indices.keys())
    print(f"Val: {len(val_dataset)} frames, {len(subjects)} subjects")

    # ── Target gaze ──────────────────────────
    if args.target_gaze_deg is not None:
        fixed_gaze = np.array([deg2rad(args.target_gaze_deg[0]),
                               deg2rad(args.target_gaze_deg[1])], dtype=np.float32)
        print(f"Target gaze (deg→rad): pitch={fixed_gaze[0]:.3f}, yaw={fixed_gaze[1]:.3f}")
    elif args.target_gaze is not None:
        fixed_gaze = np.array(args.target_gaze, dtype=np.float32)
        print(f"Target gaze (rad): pitch={fixed_gaze[0]:.3f}, yaw={fixed_gaze[1]:.3f}")
    else:
        fixed_gaze = None
        print("Target gaze: random from val dataset")

    fixed_head = None
    if args.target_head is not None:
        fixed_head = np.array(args.target_head, dtype=np.float32)

    # ── Sample subjects ───────────────────────
    # 确定目标 subject pool
    if args.subject:
        # 显式指定 subject
        target_subjects = [args.subject]
    elif not adapter_is_dir and use_adapter:
        # 单文件模式：从文件名自动推导 subject key（e.g. 00002.pth → 00002）
        derived = os.path.splitext(os.path.basename(args.adapter))[0]
        target_subjects = [derived]
        print(f"  Single adapter file → subject: {derived}")
    elif adapter_is_dir:
        # 目录模式：只选有对应权重文件的 subject
        target_subjects = [s for s in subjects
                           if os.path.exists(os.path.join(args.adapter, f"{s}.pth"))]
        print(f"  Adapter dir: {len(target_subjects)}/{len(subjects)} subjects have adapters")
    else:
        target_subjects = subjects

    # 检查指定 subject 是否存在于数据集
    missing = [s for s in target_subjects if s not in val_dataset.prefix_to_indices]
    if missing:
        print(f"  [warn] subjects not found in val split: {missing}")
        target_subjects = [s for s in target_subjects if s in val_dataset.prefix_to_indices]

    chosen = random.sample(target_subjects, min(args.num_samples, len(target_subjects)))
    rows = []

    print(f"\nRunning inference on {len(chosen)} subjects...")
    for si, subject in enumerate(chosen):
        subj_indices = val_dataset.prefix_to_indices[subject]

        # 换入该 subject 的 adapter 权重
        if use_adapter:
            adapter_file = resolve_adapter(args.adapter, subject)
            ok = swap_adapter(model, adapter_file)
            if not ok:
                print(f"  [{si+1}] {subject}: adapter not found at {adapter_file}, skipping")
                continue
            print(f"  [{si+1}/{len(chosen)}] {subject}  adapter={os.path.basename(adapter_file)}")

        # Random source frame
        src_idx = random.choice(subj_indices)
        src_batch = val_dataset[src_idx]

        # Target gaze / head — target 帧必须来自同一个人
        gt_np = None
        if fixed_gaze is not None:
            tgt_gaze_np = fixed_gaze.copy()
            tgt_head_np = fixed_head if fixed_head is not None else src_batch["source_head"].numpy()
        else:
            # 从同一 subject 中选一个与 source 不同的帧作为 target
            other_subj_indices = [i for i in subj_indices if i != src_idx]
            ref_idx = random.choice(other_subj_indices if other_subj_indices else subj_indices)
            ref_batch = val_dataset[ref_idx]
            tgt_gaze_np = ref_batch["source_gaze"].numpy()   # 该帧自身的 gaze 作为目标
            tgt_head_np = fixed_head if fixed_head is not None else ref_batch["source_head"].numpy()
            gt_np = denorm(ref_batch["source_image"])         # 该帧原图作为 GT

        pasted_np, gen_left_np, gen_right_np = run_inference(
            model, gaze_mlp, src_batch, tgt_gaze_np, tgt_head_np, cfg, device
        )

        source_np = denorm(src_batch["source_image"])
        src_gaze_np = src_batch["source_gaze"].numpy()

        # 箭头标注：source 用黄色显示当前 gaze，generated/GT 用绿色显示目标 gaze
        source_np = draw_gaze_arrow(source_np, src_gaze_np[0], src_gaze_np[1], color=(255, 220, 0))
        pasted_np = draw_gaze_arrow(pasted_np, tgt_gaze_np[0], tgt_gaze_np[1])
        if gt_np is not None:
            gt_np = draw_gaze_arrow(gt_np, tgt_gaze_np[0], tgt_gaze_np[1])

        print(f"  [{si+1}/{len(chosen)}] {subject[:20]:20s}  "
              f"src pitch={src_gaze_np[0]:+.3f} yaw={src_gaze_np[1]:+.3f}  "
              f"tgt pitch={tgt_gaze_np[0]:+.3f} yaw={tgt_gaze_np[1]:+.3f}")

        row = dict(
            subject=subject,
            source_np=source_np,
            pasted_np=pasted_np,
            gen_left_np=gen_left_np,
            gen_right_np=gen_right_np,
            target_gaze=tgt_gaze_np,
        )
        if gt_np is not None:
            row["gt_np"] = gt_np
        rows.append(row)

        # 每个 subject 保存一张拼接图 source | redirect | target
        subj_safe = subject.replace("/", "_")
        gap = 4
        panels = [source_np, pasted_np]
        if gt_np is not None:
            panels.append(gt_np)
        H = panels[0].shape[0]
        divider = np.full((H, gap, 3), 255, dtype=np.uint8)
        strip = panels[0]
        for p in panels[1:]:
            strip = np.concatenate([strip, divider, p], axis=1)
        Image.fromarray(strip).save(
            os.path.join(args.output_dir, f"{si:02d}_{subj_safe}.png"))

        # --compare: baseline vs +adapter side-by-side
        if model_baseline is not None:
            baseline_np, _, _ = run_inference(
                model_baseline, gaze_mlp, src_batch, tgt_gaze_np, tgt_head_np, cfg, device
            )
            baseline_np = draw_gaze_arrow(baseline_np, tgt_gaze_np[0], tgt_gaze_np[1])
            compare_path = os.path.join(args.output_dir, f"{si:02d}_{subj_safe}_compare.png")
            save_compare_strip(source_np, baseline_np, pasted_np, gt_np, compare_path)
            print(f"    compare → {compare_path}")

        # Circle sweep: source | gen(θ0) | gen(θ1) | ... | gen(θ7)
        sweep_head = fixed_head if fixed_head is not None else src_batch["source_head"].numpy()
        sweep_imgs = run_circle_sweep(model, gaze_mlp, src_batch, sweep_head, cfg, device)
        gap = 4
        H = source_np.shape[0]
        divider = np.full((H, gap, 3), 255, dtype=np.uint8)
        circle_strip = source_np  # source_np 已有黄色箭头（source gaze）
        for s in sweep_imgs:
            circle_strip = np.concatenate([circle_strip, divider, s], axis=1)
        Image.fromarray(circle_strip).save(
            os.path.join(args.output_dir, f"{si:02d}_{subj_safe}_circle.png"))

        # --compare circle: baseline vs +adapter two-row circle sweep comparison
        if model_baseline is not None:
            baseline_sweep_imgs = run_circle_sweep(
                model_baseline, gaze_mlp, src_batch, sweep_head, cfg, device)

            label_w = 72
            H_row = source_np.shape[0]
            div_v = np.full((H_row, gap, 3), 200, dtype=np.uint8)

            def make_circle_row(label, src_img, imgs):
                col = np.full((H_row, label_w, 3), 240, dtype=np.uint8)
                cv2.putText(col, label, (4, H_row // 2 + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1, cv2.LINE_AA)
                row = np.concatenate([col, div_v, src_img], axis=1)
                for img in imgs:
                    row = np.concatenate([row, div_v, img], axis=1)
                return row

            row_base = make_circle_row("Baseline", source_np, baseline_sweep_imgs)
            row_adpt = make_circle_row("+Adapter",  source_np, sweep_imgs)
            row_div  = np.full((gap, row_base.shape[1], 3), 200, dtype=np.uint8)
            circle_cmp = np.concatenate([row_base, row_div, row_adpt], axis=0)

            circle_cmp_path = os.path.join(
                args.output_dir, f"{si:02d}_{subj_safe}_circle_compare.png")
            Image.fromarray(circle_cmp).save(circle_cmp_path)
            print(f"    circle_compare → {circle_cmp_path}")

    # ── Grid visualization ────────────────────
    grid_path = os.path.join(args.output_dir, "inference_grid.png")
    save_grid(rows, grid_path)
    print(f"\nDone. Results in: {args.output_dir}")


if __name__ == "__main__":
    main()
