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

def load_models(cfg, checkpoint_path: str, device: torch.device):
    unet_config = OmegaConf.to_container(cfg.dic_unet_params, resolve=True)
    model = EyeOnlyWrapper(unet_config)

    gaze_params = cfg.model_params.gazenet_params
    gaze_dim = cfg.dic_unet_params.get("gaze_dim", 64)
    gaze_mlp = MLPNetwork(
        num_in=gaze_params.num_in,
        num_hidden=gaze_params.num_hidden,
        num_out=gaze_dim,
        num_layers=gaze_params.num_layers,
    )

    print(f"  Loading checkpoint: {checkpoint_path}")
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

    model, gaze_mlp = load_models(cfg, args.checkpoint, device)

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
    chosen = random.sample(subjects, min(args.num_samples, len(subjects)))
    rows = []

    print(f"\nRunning inference on {len(chosen)} subjects...")
    for si, subject in enumerate(chosen):
        subj_indices = val_dataset.prefix_to_indices[subject]

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

    # ── Grid visualization ────────────────────
    grid_path = os.path.join(args.output_dir, "inference_grid.png")
    save_grid(rows, grid_path)
    print(f"\nDone. Results in: {args.output_dir}")


if __name__ == "__main__":
    main()
