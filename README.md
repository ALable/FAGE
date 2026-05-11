# Personalized Gaze Redirection Research Notes

本仓库当前主线是 FAGE / EyeOnlyGazeDiC：输入 source eye crop 与目标 head/gaze embedding，输出目标视线下的眼部图像。本文档记录新增的 PGRA（Personalized Gaze Redirection Adapter）实验方向。

## Motivation

现有 `SubjectAdapter` 只从 source eye 提取外观 embedding，并为每个 UNet block 生成 FiLM 参数。这有助于保留身份纹理，但它没有显式表达一个用户的眼形、眼睑运动范围、虹膜可见区域如何影响不同目标 gaze 的重定向强度。

PGRA 的核心假设是：个性化 gaze redirection 不应只是 appearance preservation，而应是 subject appearance 与 target gaze 的交互调制。

## Architecture

新增模块位于 `models/subject_adapter.py`：

- `SubjectEncoder`：复用现有多尺度 eye appearance encoder。
- `GazeAwareSubjectAdapter`：输入 source eye 与目标 head/gaze embedding。
- `fuse_context()`：融合 subject、head、gaze、gaze-head delta，以及 subject-gaze interaction。
- `block_modulators`：为 EyeOnlyGazeDiC 的每个 UNet block 输出 zero-init FiLM `(scale, shift)`。

数据流如下：

```text
source_eye -> SubjectEncoder -> subject_emb
target_head, target_gaze -> GazeMLP -> head_emb, gaze_emb
subject_emb + head_emb + gaze_emb + interactions -> context vector
context vector -> per-block FiLM -> EyeOnlyGazeDiC blocks
```

由于 FiLM heads 仍然 zero-init，PGRA 在初始化时等价于原始 backbone 的 identity adapter，适合加载已有模型后继续训练或按用户微调。

## Training

新增配置：

```bash
accelerate launch --num_processes 4 train.py --config configs/training/gaze_aware_subject_adapter.yaml
```

该配置启用：

```yaml
subject_adapter_params:
  method: "gaze_aware"
  in_channels: 3
  subject_dim: 128
  gaze_dim: 64
  hidden_dim: 128
  dropout: 0.05
```

## Personalization

MPIIGaze adapter fine-tuning 会根据配置自动创建 `GazeAwareSubjectAdapter`：

```bash
python finetune_adapter_mpiigaze.py \
  --config configs/training/gaze_aware_subject_adapter.yaml \
  --checkpoint /path/to/phase1.pth \
  --subjects p12 \
  --max_steps 500
```

推理时，如果使用 adapter 权重，`inference.py` 和 `inference_mpiigaze.py` 会读取相同的 `subject_adapter_params`，从而创建匹配结构。

## Evaluation Focus

建议优先比较以下指标：

1. Gaze angular error：确认个性化调制没有削弱 gaze controllability。
2. LPIPS / ID loss：验证 subject-specific appearance 是否更稳定。
3. Extreme gaze cases：观察大角度 yaw/pitch 下眼睑、眼角、虹膜边缘是否更自然。
4. Few-shot adaptation：比较 50、100、300、500 step 下每个 subject 的收敛速度。
