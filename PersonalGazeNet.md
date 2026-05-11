# PersonalGazeNet: Personalized Gaze Redirection via Spatial Identity Tokens and Gaze-Guided Warping

## 1. Motivation

Personalized gaze redirection requires two orthogonal capabilities:

| Capability | Description |
|---|---|
| **Appearance preservation** | Retain iris colour, eyelid shape, skin texture |
| **Geometric redirection** | Move iris/pupil to target viewing direction |

The existing `EyeOnlyWrapper + SubjectAdapter` in this codebase conflates both via a **global FiLM modulation vector** (single scale+shift per channel, same for all spatial locations). This limits:

- **Spatial specificity**: Iris-region features cannot selectively attend to iris-region identity keys — the same global modulation is applied everywhere.
- **Robustness**: A single reference frame is sensitive to occlusion and illumination variation.
- **Disentanglement**: Geometry and appearance share the same conditioning pathway, making it hard to independently optimize them.

PersonalGazeNet addresses all three limitations.

---

## 2. Novel Architecture Overview

```
PersonalGazeNet
│
├── IdentityEncoder (gaze-unconditional CNN)
│     Shared weights process each of N_ref reference frames.
│     Produces feature pyramids:  feat_0 / feat_1 / feat_lat
│     at three spatial scales.
│
├── IdentityAggregator
│     Mean-pools N_ref feature pyramids → stable identity representation.
│     Global identity embedding id_emb [B, D_id] for decoder FiLM.
│     Spatial identity tokens id_feat_lat [B, C2, H/4, W/4] for cross-attention.
│
├── SourceEncoder (gaze-conditioned UNetBlocks)
│     Same 3-level structure as EyeOnlyGazeDiC encoder.
│     Gaze conditioning via ConditionEmbedder → AdaLN in each UNetBlock.
│     Outputs: src_feat_0, src_feat_1, src_lat
│
├── GazeDeltaWarpModule ← KEY INNOVATION
│     Δgaze = target_gaze − source_gaze  [B, 2]
│     MLP: [B, 2] → [B, warp_hidden × seed_h × seed_w]
│     CNN decoder: upsample to [B, 2, H/4, W/4] flow field
│     F.grid_sample(src_lat, base_grid + flow) → warped_lat
│     Explicit geometric correction BEFORE appearance synthesis.
│
├── CrossAttentionBottleneck ← KEY INNOVATION
│     Q: warped_lat (flattened)  [B, H/4×W/4, C2]
│     K, V: id_feat_lat (flattened)  [B, H/4×W/4, C2]
│     Spatially-aware identity injection: iris queries attend iris keys.
│     Zero-init output projection → stable initialisation.
│
└── GazeCondDecoder
      Two upsampling stages with skip connections from SourceEncoder.
      Gaze conditioning: ConditionEmbedder → AdaLN (existing UNetBlock API)
      Identity FiLM: id_emb → IdentityFiLMHead → (scale, shift) per stage
                     (uses existing subject_scale/subject_shift API)
      Final: DiCFinalLayer → ΔI → source + ΔI  (residual prediction)
```

---

## 3. Key Innovations vs Existing Code

| Feature | EyeOnlyWrapper | PersonalGazeNet |
|---|---|---|
| Identity encoding | Single frame, FiLM (global) | N frames, mean-pool, cross-attention (spatial) |
| Geometric correction | Implicit (learn by pixel reconstruction) | Explicit: gaze-delta flow field + grid_sample |
| Identity injection | Global: same scale/shift everywhere | Spatial: cross-attn at bottleneck + FiLM in decoder |
| Gaze / identity coupling | Shared UNet (entangled) | Separate encoder paths (decoupled) |
| Identity robustness | Single frame | N_ref frames, average → handles occlusion |

---

## 4. Two-Phase Training

### Phase 1: Shared Pre-training (Self-supervised)

All parameters trained jointly. Reference = source frame (N_ref = 1), so the cross-attention is trivially self-attention. This gives a stable initialisation for all modules.

```bash
accelerate launch train_personal_gaze_net.py \
    --config configs/training/personal_gaze_net.yaml
```

Config: `configs/training/personal_gaze_net.yaml`
- `data.num_ref_frames: 1`
- All parameters trainable

### Phase 2: Per-Subject Identity Fine-tuning

Freeze the backbone (SourceEncoder + Decoder + GazeDeltaWarpModule + CrossAttentionBottleneck).
Only update:
- `identity_encoder` — learns gaze-invariant features for this subject
- `identity_aggregator` — adapts pooling/projection to new appearance
- `id_film_dec0`, `id_film_dec1` — personalises decoder FiLM heads

This only requires 300–1000 steps per new subject (approximately the same cost as SubjectAdapter).

```python
# Freeze backbone, only train identity modules
for name, p in model.named_parameters():
    if any(k in name for k in ['identity_encoder', 'identity_aggregator',
                                'id_film_dec']):
        p.requires_grad = True
    else:
        p.requires_grad = False
```

---

## 5. Loss Functions

### Standard losses (same as EyeOnlyWrapper)

| Loss | Weight | Start step |
|---|---|---|
| L1 pixel | 2.0 | 0 |
| GAN (multi-scale discriminator) | 0.3 | 3000 |
| Gaze perceptual (VGG gaze net) | 0.01 | 500 |
| Identity (ArcFace) | 0.1 | 1000 |

### New: Flow Smoothness Loss

Penalises spatially abrupt warping fields, encouraging physically-plausible iris motion:

```
L_smooth = Σ |∂flow/∂x| + |∂flow/∂y|
```

This is a total-variation regulariser on the predicted displacement field. Weight: `0.01` (light, prevents sharp flow artefacts without over-constraining the geometry).

---

## 6. Architecture Parameters

Default configuration (`configs/training/personal_gaze_net.yaml`):

```yaml
pgn_params:
  hidden_size:    32        # base channel count (matches EyeOnlyGazeDiC)
  mult_channels:  [1,2,4,2,1]
  depth:          [2,2,4,2,2]
  gaze_dim:       64
  id_emb_dim:     128       # global identity embedding dimension
  num_attn_heads: 4         # cross-attention heads
  warp_hidden:    128       # flow predictor MLP hidden dim
  warp_max_disp:  0.3       # max displacement in normalised coords
```

**Parameter count (approximate)**:

| Module | Params |
|---|---|
| IdentityEncoder | ~500K |
| IdentityAggregator | ~100K |
| SourceEncoder | ~400K (matches EyeOnlyGazeDiC encoder) |
| GazeDeltaWarpModule | ~200K |
| CrossAttentionBottleneck | ~200K |
| Decoder + FiLM heads | ~400K |
| **Total** | **~1.8M** |

Phase 2 fine-tuning updates only ~600K parameters (IdentityEncoder + Aggregator + FiLM heads).

---

## 7. Design Choices and Justifications

### Why gaze-unconditional IdentityEncoder?

The identity encoder is intentionally **not** conditioned on gaze. This:
- Forces the representation to be gaze-invariant (appearance, not pose)
- During Phase 1 (ref = source), gaze is identical → no information leak
- During Phase 2, reference frames with similar gaze are preferred

If gaze-invariance becomes a bottleneck, a future option is to add a gaze-normalisation step (rotate features to canonical gaze before encoding).

### Why cross-attention only at bottleneck?

The bottleneck spatial size (H/4 × W/4 = 20 × 40 = 800 tokens) keeps attention cost manageable. At full resolution (12,800 tokens), standard dot-product attention would be prohibitively expensive. The global FiLM at decoder stages compensates for the coarser injection.

### Why gaze delta (not absolute gaze) in WarpModule?

Gaze-delta encoding provides:
- Translation invariance in gaze space (only the change matters for motion)
- Zero initial flow when source = target (natural for residual learning)
- Smaller input range → more stable MLP training

### Why flow at latent resolution?

Latent flow avoids modelling pixel-level appearance, focusing only on structural displacement. Refinement of appearance details is left to the cross-attention and decoder.

---

## 8. Inference Flow

```
# Step 1: Encode subject identity (run once per subject)
id_feat_lat, id_emb = model.net.encode_identity(ref_eyes)  # cached

# Step 2: Per-frame generation
generated, _ = model(
    source_eye_crops=source_eye,
    encoder_hidden_states=gaze_prompt,
    ref_eyes=ref_eyes,             # or use cached id_feat_lat/id_emb directly
    source_gaze_raw=source_gaze,
    target_gaze_raw=target_gaze,
)

# Step 3: Paste back into source face
result = model.paste_eyes(generated, source_face, eye_bbox)
```

---

## 9. File Map

| File | Role |
|---|---|
| `models/personal_gaze_net.py` | Full architecture (all 7 components) |
| `configs/training/personal_gaze_net.yaml` | Training config (Phase 1) |
| `train_personal_gaze_net.py` | Training script with flow smoothness loss |

Reused from existing codebase:
- `models/gaze_dic.py` — UNetBlock, Downsample, Upsample, DiCFinalLayer, ConditionEmbedder
- `models/subject_adapter.py` — EncoderBlock (used in IdentityEncoder)
- `models/gazenet.py` — MLPNetwork (gaze embedding)
- `dataset/gaze_capture.py` — HDFDataset (already provides `source_gaze`)
- `loss/` — all existing losses
- `utils/training_utils.py` — initialize_loss_functions
