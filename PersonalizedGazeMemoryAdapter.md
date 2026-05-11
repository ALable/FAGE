# Personalized Gaze Memory Adapter

## Goal

PGMA is a small architecture extension for personalized gaze redirection. It keeps the
existing `EyeOnlyGazeDiC` generator and adds a gaze-aware subject memory path that can
be trained together with the shared model or fine-tuned for a new user.

## Motivation

The current SubjectAdapter maps a source eye crop to one global subject vector and then
generates block-wise FiLM parameters. This is efficient, but a single vector can lose
local identity cues such as eyelid shape, iris boundary, eyelashes, and asymmetric eye
appearance. PGMA preserves these details as multiple memory tokens and lets the target
head/gaze embedding query the memory.

## Architecture

```text
source_eye
  -> SubjectMemoryEncoder
  -> memory tokens [B, T, D]

target head/gaze embedding
  -> query MLP
  -> query token [B, 1, D]

query token + memory tokens
  -> MultiHeadAttention
  -> context vector [B, D]
  -> per-block FiLM heads
  -> [(scale_i, shift_i)] for EyeOnlyGazeDiC blocks
```

The FiLM heads are zero-initialized. At initialization PGMA is an identity adapter, so
enabling it does not immediately disturb a stable base generator.

## Training config

Use the new config:

```bash
accelerate launch --num_processes 4 train.py --config configs/training/pgma_eye_only.yaml
```

The key switches are:

```yaml
dic_unet_params:
  personalized_gaze_adapter: true
  pgma_token_dim: 128
  pgma_base_channels: 32
  pgma_token_grid: [2, 4]
  pgma_num_heads: 4
```

## Inference and caching

For repeated inference on the same subject, the memory tokens can be cached:

```python
tokens = model.personalized_adapter.encode(reference_eye_crops).mean(dim=0, keepdim=True)
mods = model.personalized_adapter.get_modulations(tokens, gaze_prompt)
```

The wrapper computes this path automatically during normal `model(source_eye, gaze)`
calls. Manual caching is useful for deployment code that reuses the same reference
appearance across many target gazes.

## Expected benefits

- Preserves local subject cues better than a single global embedding.
- Makes personalization explicitly target-gaze-aware.
- Keeps the main generator unchanged and checkpoint-compatible when the config flag is
  disabled.
- Adds a small trainable path that can be isolated for fast per-user fine-tuning.

## Validation checklist

```bash
python3 -m py_compile models/personalized_gaze_adapter.py models/gaze_dic.py
python3 - <<'PY'
import torch
from models.gaze_dic import EyeOnlyWrapper

cfg = {
    "in_channels": 3,
    "hidden_size": 16,
    "mult_channels": [1, 2, 4, 2, 1],
    "depth": [1, 1, 1, 1, 1],
    "gaze_dim": 64,
    "personalized_gaze_adapter": True,
    "pgma_token_dim": 64,
    "pgma_num_heads": 4,
}
model = EyeOnlyWrapper(cfg)
eye = torch.randn(2, 3, 80, 160)
gaze = torch.randn(2, 2, 64)
out = model(eye, gaze)
print(out.shape)
PY
```
