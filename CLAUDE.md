# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Eye-only gaze redirection with per-user personalization. Based on DiC (CVPR 2025) pure convolution UNet.

- **Input**: Eye crop [6, 80, 80] (left + right concat, large-range) + target gaze direction
- **Output**: Generated eye crop [6, 80, 80]，center-crop 到 [6, 64, 64] 与 GT 对比
- **Personalization**: SubjectAdapter (~346K) FiLM 调制；冻结 GazeControlNet，只更新 SubjectAdapter

## Environment

```bash
conda activate deepface
export PYTHONPATH="${PYTHONPATH}:/home/xuhy/PycharmProjects/FAGE"
```

Key dependencies: `omegaconf`, `accelerate`, `diffusers` (LR scheduler), `h5py`, `lpips`, `piq`, `cv2`, `torchvision`

## Training

```bash
# Phase 1: 共享预训练（GazeControlNet + SubjectAdapter 全量训练）
python train.py --config configs/training/dic_eye_only.yaml

# Phase 2: 个性化（冻结 GazeControlNet，只训练 SubjectAdapter）
# 先在 dic_eye_personalized_v2.yaml 中设置 personalization.phase1_checkpoint
python train.py --config configs/training/dic_eye_personalized_v2.yaml
```

**Phase 2 数据协议**：使用 val subjects（从未出现在 Phase 1 训练中），验证对新用户的泛化。

## Architecture

### 双模块设计（明确分离）

```text
GazeControlNet (eye_unet, ~2.41M)  ← Phase 2 冻结
  ├─ GazeMLP (MLPNetwork, ~17K): head/gaze → embedding [B, 64]
  ├─ OverlapPatchEmbed: [B,6,H,W] → [B,32,H,W]
  ├─ ConditionEmbedder × 3: per-stage gaze embedding
  ├─ UNetBlock × 12: AdaLN gaze 注入 + Subject FiLM 注入
  │    ① gaze: gate/scale/shift from gaze_emb (AdaLN)
  │    ② subject: scale/shift from SubjectAdapter (FiLM，零初始化)
  └─ DiCFinalLayer: AdaLN-Zero 输出层

SubjectAdapter (~346K)  ← Phase 2 只训练这部分
  ├─ SubjectEncoder: [B,6,H,W] → z_s [B,128]
  │    Conv(6→16)→Conv(16→32,s2)→Conv(32→64,s2)→Conv(64→128,s2)→GAP→LN
  └─ BlockModulators × 12: z_s → (scale_i, shift_i) per block
       Linear(128, 2C)，零初始化
```

### UNet 结构（3层）

| Stage  | 分辨率 | 通道 | Blocks |
|--------|--------|------|--------|
| enc0   | 80×80  | 32   | 2      |
| enc1   | 40×40  | 64   | 2      |
| latent | 20×20  | 128  | 4      |
| dec0   | 40×40  | 64   | 2      |
| dec1   | 80×80  | 32   | 2      |

### 参数量

| 模块           | 参数量  | Phase 1 | Phase 2    |
|----------------|---------|---------|------------|
| GazeControlNet | 2.41M   | 训练    | 冻结       |
| SubjectAdapter | 346K    | 训练    | **只训练** |
| 合计           | ~2.76M  | 全量    | 346K       |

## Key Files

- `models/gaze_dic.py` — EyeOnlyGazeDiC (UNet), UNetBlock (AdaLN+FiLM), EyeOnlyWrapper
- `models/subject_adapter.py` — SubjectEncoder, SubjectAdapter (FiLM per-block)
- `models/gazenet.py` — MLPNetwork: [B,2] → [B, gaze_dim=64]
- `train.py` — Phase 1 & 2 训练逻辑（数据集、optimizer、checkpoint）
- `dataset/gaze_capture.py` — HDFDataset: 配对帧 + 双尺寸眼部 crop
- `loss/basic_loss.py` — gaze_angular_loss, GAN losses
- `loss/discriminator.py` — MultiScaleDiscriminator
- `utils/training_utils.py` — initialize_loss_functions()
- `configs/training/dic_eye_only.yaml` — Phase 1 config
- `configs/training/dic_eye_personalized_v2.yaml` — Phase 2 config

## Key Design

- **Complementary gating**: `sigmoid(gate) * x + (1 - sigmoid(gate)) * skip`（凸组合��输出有界）
- **AdaLN modulation**: `modulate(x, shift, scale)` 注入 gaze 条件（注视控制）
- **Subject FiLM**: `x = x * (1 + scale_i) + shift_i`，零初始化，注入主体外观（Phase 1 也训练）
- **Paired frames**: 源帧提供眼部外观，目标帧提供 gaze 方向 + GT
- **Two-scale crops**: input [6,80,80] 大范围 → 模型输入；tight [6,64,64] → loss 监督

## Data

- GazeCapture HDF5: `/mnt/data/xhy/eye-concat/GazeCapture_256.h5`
  - Subject-level split: 80% train (1092 subjects) / 10% val (136 subjects) / 10% test
  - 配对帧偏移 ±N（Phase 1: ±4，Phase 2: ±2）
  - **Phase 2 使用 val subjects（新用户，与 Phase 1 无重叠）**
- Phase 1 checkpoint: `/mnt/data/xhy/Logs/FAGE/logs/EyeOnly/4-7/FAGE-EyeOnly-v1/`

## Fast Inference (新用户)

```python
# 对新用户只运行一次
z_s = model.subject_adapter.encode(ref_frames).mean(0, keepdim=True)
cached_mods = model.subject_adapter.get_modulations(z_s)

# 每帧推理（SubjectEncoder 不再运行）
out = model.eye_unet(eye_crop, gaze_cond, subject_mods=cached_mods)
```
