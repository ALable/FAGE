# CLAUDE.md — FAGE (Few-shot Adaptive Gaze Generation)

## Project Overview

Eye-only gaze redirection with few-shot user adaptation. Based on DiC (CVPR 2025) pure convolution UNet.

- **Input**: Eye crop [3, 64, 128] (left + right concatenated) + target gaze direction
- **Output**: Generated eye crop with redirected gaze
- **Personalization**: DynamicMaskGenerator (~300K params) with Gumbel-Sigmoid, freeze UNet + fine-tune masks

## Environment

```bash
conda activate deepface
export PYTHONPATH="${PYTHONPATH}:/home/xuhy/PycharmProjects/FAGE"
```

## Training

```bash
# Phase 1: Eye-only generation (shared model)
python train_fage.py --config configs/training/dic_eye_only.yaml

# Phase 2: User personalization (freeze UNet, train mask generator)
python train_fage.py --config configs/training/dic_eye_personalized.yaml
```

## Architecture

- `models/gaze_dic.py` — EyeOnlyGazeDiC (UNet, ~3M), EyeOnlyWrapper, UNetBlock (complementary gating)
- `models/mask_generator.py` — DynamicMaskGenerator (Gumbel-Sigmoid, ~300K)
- `models/gazenet.py` — MLPNetwork (gaze MLP)
- `dataset/gaze_capture.py` — HDFDataset (paired frames + eye crops)

## Key Design

- **Complementary gating**: `sigmoid(gate) * x + (1 - sigmoid(gate)) * skip` (bounded output)
- **Channel masking**: Per-block channel masks from DynamicMaskGenerator, applied after gated residual
- **Gumbel-Sigmoid**: Training=soft+noise, Eval=hard binary. Temperature annealing tau: 1.0→0.1
- **Paired frames**: Source eye appearance + target gaze direction → generated target eye

## Data

- GazeCapture HDF5: `/mnt/data/xhy/eye-concat/GazeCapture_256.h5`
- ArcFace: `checkpoints/arcface/model_ir_se50.pth` (symlink from Unet-Gaze)
