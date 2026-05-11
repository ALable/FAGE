# PersonaGazeNet — 面向个性化的视线重定向新架构

> 研究草案 / 设计文档 (v0.1)
>
> 目标: 在保持当前 FAGE 项目"眼部生成质量"基线的前提下, 解决"新用户个性化"这一核心痛点
> ——以"K 参考帧 + 一次前向"取代"逐用户 fine-tune", 同时显著提升对身份局部细节
> (虹膜纹路 / 眼角 / 睫毛 / 眼睑形状) 的保持能力.

---

## 0. TL;DR

我们提出 **PersonaGazeNet (PGN)**, 一个把"个性化"由"梯度过程"重新表述为"条件前向过程"的
视线重定向网络. 它有三个核心新点:

1. **参考帧 Token Bank + 空间交叉注意力 (RFA, Reference-Frame Attention)**.
   不再把用户压成一个 128 维向量, 而是保留 *K 帧 × N tokens* 的空间结构, 让解码器在每个
   空间位置上"按需检索"它需要的局部外观证据.

2. **解耦双流 + 显式注入点 (Disentangled Gaze / Identity Streams)**.
   Gaze (几何) 与 Identity (外观) 走两条不同尺度的注入路径:
   gaze 走 AdaLN-Zero 在所有层做 *几何变形*; identity 只走 *解码器中后段* 做
   *外观贴图*. 这种"几何前注入, 外观后注入"的归纳偏置, 使得 gaze 控制和身份控制不再互相
   抢通道.

3. **HyperLoRA 个性化头 (Hypernetwork-predicted LoRA)**.
   一个独立的小网络 `HyperLoRA(\bar{z}_{user}) → {A_l, B_l}` 在一次前向中预测每一层 conv
   的低秩残差; 完全消除了 per-user 反向传播. 当数据更多时, 我们仍允许 *只对 LoRA 矩阵* 做
   少量 (<100 步) 的可选 online 微调, 形成 *zero-shot → few-shot → fine-tune*
   三个性能档位的统一接口.

辅助贡献 (训练侧):

- **Diverse-Gaze K-shot 参考采样**: 训练时强制参考集在 gaze 方向上离散覆盖, 让模型学会
  *跨视线的身份插值*.
- **Identity-Anchored Cycle Loss**: 把生成结果再交还给"源 gaze"应当还原源眼, 提供
  强身份监督信号.
- **Gaze-Code Whitening**: 在 gaze MLP 输出端加一个轻量白化层, 防止 head/gaze code 与
  identity code 在表征空间共线.

---

## 1. 问题再定义

输入:

- **目标 gaze 条件**: `target_head ∈ R^2`, `target_gaze ∈ R^2` (pitch, yaw, 弧度).
- **参考集** (用户特有): `R = {ref_eye_1, ..., ref_eye_K}`, 每张 `[3, 64, 128]`
  (左右眼宽拼). 参考集 *不要求* 包含目标 gaze.
- **源眼图** (用于残差 / 贴回): `source_eye ∈ R^{3 \times 64 \times 128}`.
- (可选) **源全脸** `source_face ∈ R^{3 \times 256 \times 256}`, 用于推理阶段贴回.

输出:

- `eye_pred ∈ R^{3 \times 64 \times 128}`: 目标 gaze 下、保持参考用户身份的眼部图像.

性能档位 (统一接口):

| 档位        | 步骤                                                       | 适用场景         |
| ----------- | ---------------------------------------------------------- | ---------------- |
| **K0**      | K=1, 仅一次前向                                            | 直播 / 实时      |
| **K-shot**  | K=4~8 参考帧, 一次前向                                     | 用户已注册       |
| **+Online** | 在 K-shot 基础上, 仅对 HyperLoRA 输出的 `A_l, B_l` 做 SGD | 离线高保真渲染   |

---

## 2. 与现有方案的对标

| 模块                | 当前 FAGE                                  | PersonaGazeNet (本文)                                                    |
| ------------------- | ------------------------------------------ | ------------------------------------------------------------------------ |
| 身份编码            | `SubjectEncoder` → 单一 `z_s ∈ R^128`     | `RefTokenizer` → token bank `T ∈ R^{K × N × C}`                          |
| 身份注入            | 通道 FiLM (scale/shift per channel)        | 空间 Cross-Attention + 通道 FiLM (双轨)                                  |
| 几何/外观解耦       | 隐式 (都进 UNetBlock 的 affine 参数)       | **显式**: gaze 在 enc + latent, identity 主要在 dec 后半                  |
| 个性化方式          | per-user SGD (500 步 / 用户)              | **HyperLoRA**: 一次前向预测 LoRA delta (可选 online 微调 ≤100 步)        |
| 参考帧数            | 1 (源眼)                                   | 任意 K ∈ {1, 4, 8} (训练时随机)                                          |
| 头部外观信号        | `HeadImageEncoder` 全局池化                | `HeadTokenizer` 保留空间 → 同样进 Cross-Attention                        |
| Phase 2 finetune    | 必需 (核心模型 frozen, adapter 训练)       | 可选 (默认 zero-shot)                                                    |

直观地说: 我们把 SubjectAdapter 的"一次性梯度调参"换成了"一次性前向预测",
同时把"全局向量调制"扩展为"空间 token 交叉注意力 + LoRA 双轨注入".

---

## 3. 架构总览

```text
                ┌──────────────────────────────────────────┐
                │   Gaze MLP (head, gaze) → e_gaze ∈ R^G   │
                └──────────────────────────────────────────┘
                                   │
                          (AdaLN-Zero, 全程注入)
                                   │
   ref_eyes[K] ─► RefTokenizer ──► T_ref ∈ R^{K·N × C_t}
                                   │         │
                                   │         └──► HyperLoRA Head ─► {A_l, B_l} per decoder conv
                                   │
                                   ▼ (Cross-Attention, 仅 decoder 后段)
   source_eye ─► EyeUNet (enc → latent) ─► [B, C, h, w] ──► decoder ──► δI
                                              ▲                            │
                                              │                            ▼
                                              └──── (skip + RFA + LoRA) ──► output = source_eye + δI
```

### 3.1 模块清单

- `RefTokenizer`: 共享权重的 CNN, 把每张 `[3, 64, 128]` 参考眼变成 `[N, C_t]` token 序列 (空间展平后做 1×1 投影 + 位置编码).
- `HeadTokenizer` (可选): 同样的设计, 输入是 `[3, 256, 256]` source face.
- `EyeUNet`: 复用 `EyeOnlyGazeDiC` 的 enc / latent / dec 结构, 但去除原 `SubjectAdapter` FiLM 注入点, 改在 decoder 加入两个新模块:
  - `RefCrossAttn` (空间 cross-attention)
  - `LoRAConv2d` (低秩 conv 残差)
- `GazeEmbedder`: 同 `ConditionEmbedder` (沿用 head+gaze 双分支融合).
- `HyperLoRAHead`: MLP, 输入 `\bar{T}_ref ∈ R^{C_t}` (token bank 的 attention-pool), 输出每层 LoRA 的 `A_l, B_l`.

### 3.2 RefTokenizer (参考帧分词器)

```text
input  : [B*K, 3, H_e=64, W_e=128]
stem   : Conv 3×3, stride 1, ch=C_t/4
down×2 : PixelUnshuffle to 16×32, ch=C_t
norm   : GroupNorm + GELU
project: 1×1 conv → flatten HW → [B*K, N=H_e/4 * W_e/4 = 16×32 = 512, C_t]
add pos: 学习式 2D 位置编码
```

注意我们刻意 **保留 (16, 32) 的空间分辨率**, 因为虹膜 / 瞳孔的位置纹路必须在 token 级别可寻址.

### 3.3 RefCrossAttn (参考交叉注意力)

仅插在 **decoder 的上采样块之间** (即 latent → 32×64 → 64×128 这两步之间), 不在 encoder
插入, 因为 encoder 的语义还是 *gaze 几何*, 不需要看 identity.

```python
# q ∈ [B, h*w, C], 来自 decoder 当前特征 (h, w)
# k = v ∈ [B, K*N, C], 来自 token bank, 经 LayerNorm + Linear 映射到 C
# 多头注意力, num_heads=4, dropout=0.1
# 输出 [B, h*w, C], reshape 回 [B, C, h, w], gate-residual 加回 decoder 特征
out = gate * MHA(q, k, v) + (1 - gate) * decoder_feat       # gate 初始化为 0
```

`gate` 用 `sigmoid(g)` 形式, `g` 初始为 `−5`, 训练前期 RFA 几乎是 identity, 慢慢
"通车"——这与 AdaLN-Zero 同样的稳定化策略.

### 3.4 HyperLoRA 个性化头

对于 decoder 中的每个 `Conv2d(C_in, C_out, 3×3)`, 我们引入低秩残差:

```text
ΔW_l = α / r · (B_l @ A_l),   A_l ∈ R^{r × C_in}, B_l ∈ R^{C_out × r}
W̃_l = W_l + ΔW_l
```

`A_l, B_l` 由 `HyperLoRAHead(\bar{T}_ref)` 预测:

```text
\bar{T}_ref = AttentionPool_over_(K, N)(T_ref) ∈ R^{C_t}
shared = MLP(\bar{T}_ref) ∈ R^{C_s}
for each decoder conv l:
    A_l = head_A_l(shared).reshape(r, C_in)
    B_l = head_B_l(shared).reshape(C_out, r)   # 零初始化 → 零 LoRA → 不破坏 backbone
```

参数量分析 (decoder 假设 6 个 conv, 平均 C_in=C_out=128, r=8):

- 经典 per-user LoRA 总参数: ~12K × 6 = 72K. 现在所有用户共享一个 hypernet 输出这些 72K
  数字, hypernet 自身约 ~200K, **跨用户被摊薄**.
- 推理: 一次前向得到 ΔW, 直接合并到 W; 后续帧推理 0 额外开销 (LoRA 推理友好).

### 3.5 解耦注入策略 (核心 inductive bias)

> 这是论文卖点之一: gaze 与 identity 不应在同一通道竞争.

| 层级           | 走什么条件                                              | 通过什么机制        |
| -------------- | ------------------------------------------------------- | ------------------- |
| enc0 / enc1    | **gaze only** (AdaLN-Zero)                              | UNetBlock 的 affine |
| latent         | **gaze only**                                           | UNetBlock 的 affine |
| dec0 (8×16)    | gaze + identity (RFA, low-res)                          | AdaLN + RFA         |
| dec1 (16×32)   | gaze + identity (RFA + LoRA)                            | AdaLN + RFA + LoRA  |
| dec2 (32×64)   | identity-dominated (RFA + LoRA, gaze 仅 final layer 用) | AdaLN gate 衰减     |
| final layer    | gaze (微调最终颜色分布)                                 | DiCFinalLayer       |

直观上: 几何 (转眼球的位置) 必须在低分辨率定下来; 外观 (虹膜的褐色 / 灰色, 巩膜的红血丝)
必须在高分辨率"贴上"——这正好是 SDXL / IP-Adapter 系列发现的经验.

---

## 4. 训练策略

### 4.1 三阶段课程

| 阶段                     | 训练的模块                                           | 数据                               | 步数            | 目标                          |
| ------------------------ | ---------------------------------------------------- | ---------------------------------- | --------------- | ----------------------------- |
| **Phase 1: Gaze-Only**   | EyeUNet + GazeMLP                                    | GazeCapture, 不喂参考集            | 80K             | 学会 gaze 几何控制            |
| **Phase 2: RFA Warm-up** | + RefTokenizer + RefCrossAttn (gate 解锁)            | GazeCapture, K=1 (source eye)      | 60K             | 让 RFA 通车, 接管 identity    |
| **Phase 3: HyperLoRA**   | + HyperLoRAHead, K ∈ {1,4,8} 随机                   | GazeCapture, MPIIGaze 混合         | 60K             | 学会零样本 LoRA 预测          |

### 4.2 损失函数 (总览)

```text
L = λ_rec  · L1(eye_pred, eye_gt)
  + λ_lpips · LPIPS(eye_pred, eye_gt)
  + λ_gaze  · GazeAngular(GazeEstimator(eye_pred), target_gaze)
  + λ_id    · ArcFace(eye_pred, ref_eyes.mean)         # 仅 Phase 2/3
  + λ_cycle · L1(cycle_eye_pred, source_eye)           # 见 4.3
  + λ_adv   · PatchGAN(eye_pred)                       # 后期开
  + λ_hwhite · WhiteningLoss(e_head, e_gaze)           # 见 4.4
```

### 4.3 Identity-Anchored Cycle Consistency

```text
# 正向: source_eye + (head_t, gaze_t) → eye_pred
# 逆向: eye_pred 作为参考, source_eye + (head_s, gaze_s) → cycle_eye_pred
# 监督: cycle_eye_pred 应当近似 source_eye

# 这相当于把生成结果当作新的"参考集", 强制模型保持身份在 gaze 变换下的可逆性.
```

### 4.4 Gaze-Code Whitening

我们经验地发现 head/gaze code 与 identity code 在初期会通过通道偏向耦合 (因为它们都
inject 到同一个 UNetBlock). PGN 通过 *物理隔离* (3.5 节) 已减轻该问题, 但仍加一个软约束:

```python
# e_head, e_gaze : [B, G]
# 鼓励它们的协方差矩阵接近单位阵
L_white = ||cov([e_head, e_gaze]) - I||_F
```

只是一项辅助 loss, 权重 1e-3.

### 4.5 Diverse-Gaze K-shot 参考采样

修改 `HDFDataset` 的 paired-frame 采样:

```text
给定一个 (subject, source_frame), 不再只采一个 target_frame, 而是:
  - 1 个 target_frame (用于监督)
  - K 个 ref_frames, 它们彼此 gaze 距离 ≥ τ (e.g. 5°)
  - K 帧的 head pose 与 source 接近 (max_head_diff)
  - K 帧 *不包含* target_frame 本身 (防泄漏)
```

K 在 batch 内随机选 1/4/8, 让模型适应不同的"用户注册量".

---

## 5. 评估方案

复用现有 `validate.py` 指标 (L1 / LPIPS / PSNR / GazeAngular). 新增:

1. **K-curve**: 固定 subject, 让 K=1,2,4,8,16, 报告 LPIPS / GazeError 随 K 的变化.
2. **Cross-subject leak**: 把 subject A 的参考帧喂 subject B 的 source eye, 应当看到
   生成结果向 A 的外观偏移 (这是积极信号, 说明 identity 真在起作用).
3. **Per-user adaptation curve**: 对比"PGN 零样本" vs "原 SubjectAdapter 500 步 SGD" vs
   "PGN + 100 步 LoRA SGD". 期望 PGN-zero ≥ Adapter-500, PGN-100 全面胜出.

---

## 6. 计算/参数预算

| 模块                  | 参数量 (估算)  | 推理开销 (per frame)              |
| --------------------- | -------------- | --------------------------------- |
| EyeUNet (backbone)    | ~3.0M (沿用)   | 同 baseline                       |
| RefTokenizer (K=4)    | ~250K          | 仅注册期一次, 之后 token 缓存     |
| RefCrossAttn × 2 头   | ~180K          | +~3% latency                      |
| HyperLoRAHead         | ~200K          | 仅注册期一次, 之后 LoRA 合并到 W  |
| GazeMLP               | 同基线         | 同基线                            |
| **总参数**            | **~3.6M**      | 推理近似零额外开销 (LoRA 合并后)  |

---

## 7. 与文献的关系 (简要)

- *Reference Cross-Attention* 思想类: IP-Adapter, Paint-by-Example, StyleAdapter——它们都
  证明了 spatial token-level 条件远优于全局向量条件. PGN 把这个思想第一次系统地用到 gaze
  redirection.
- *Hypernetwork → LoRA* 思想类: HyperDreamBooth (2023), Cones (2023). 我们在 *eye crop*
  分辨率下使用更轻量的 hypernet (200K vs 1B).
- *Disentangled 注入点*: 受 SDXL / Cross-Attn-Map 分析启发——粗几何在 latent, 细外观在 high-res
  decoder. 在 gaze redirection 里这种解耦在我们仓库里之前没人显式做.

---

## 8. 落地路线 (本 PR 范围)

本 PR 仅交付 **第一步 (架构定义 + 原型 + smoke test)**, 不跑大规模训练:

- [x] `models/persona_gaze.py`: PGN 的最小可前向版本 (含 RefTokenizer / RefCrossAttn / HyperLoRA / GazeEmbedder).
- [x] `configs/training/persona_gaze.yaml`: 训练超参草案.
- [x] `dataset/gaze_capture.py` 的 K-shot 参考采样 (新方法 `__getitem__multi_ref`, 不破坏旧接口).
- [x] `train_persona_gaze.py`: 训练脚本骨架 (Phase 1/2/3 切换, 沿用 Accelerate / TB).
- [x] `tests/test_persona_gaze_forward.py`: 单元自检, 用随机张量跑通 forward + backward.

后续 PR 再补:

- HyperLoRA 的 *合并到 W* 推理路径 (现在每步重新计算 ΔW).
- Cycle loss / Whitening loss 的具体实现.
- MPIIGaze K-shot eval 脚本.
