# DualStreamGazeNet: 双流个性化视线重定向网络

## 1. 动机与问题分析

### 当前 FAGE (EyeOnlyGazeDiC) 架构的局限性

| 局限 | 说明 | 影响 |
|------|------|------|
| **FiLM 调制瓶颈** | SubjectAdapter 仅通过通道级 scale/shift (FiLM) 注入身份信息 | 无法精确控制*空间哪些位置*需要保留细节 |
| **单向量身份表征** | SubjectEncoder 将整个眼部压缩为 128 维向量 | 丢失虹膜纹理、睫毛分布等空间细节 |
| **绝对角度条件** | GazeMLP 直接编码目标注视角度 | 无法利用源-目标角度差异的结构信息 |
| **身份-注视耦合** | 单流 UNet 中身份和注视特征共享相同的卷积通道 | 调节注视方向时容易破坏身份特征 |

### 新架构的核心思想

**将身份保持与注视控制解耦为两个显式的信息流, 通过交叉注意力融合:**

```text
源眼部裁剪 ─┬─── AppearanceEncoder (Stream 1) ──── 多尺度身份特征 (无注视信号)
             │                                              ↓
             │                                      Cross-Attention
             │                                              ↓
             └─── GazeTransformDecoder (Stream 2) ──── 注视重定向生成 ← Δgaze 条件
```

## 2. 架构总览

### 2.1 组件清单

| 组件 | 功能 | 参数量 (hidden_size=32) |
|------|------|------------------------|
| **AppearanceEncoder** | 提取多尺度身份特征, 不接收注视信号 | ~150K |
| **DeltaGazeEmbedder** | 正弦位置编码 + MLP, 生成每阶段注视嵌入 | ~180K |
| **GazeTransformDecoder** | 注视条件 UNet + 交叉注意力, 生成重定向眼部 | ~2.2M |
| **PersonalizationHead** | 轻量级个性化: 注意力偏置 + 可选 FiLM | ~200K |
| **合计** | | ~2.7M |

### 2.2 数据流

```text
输入:
  source_eye_crops: [B, 3, H, W×2]     # 左右眼宽度拼接 (例如 80×160)
  source_gaze:      [B, 2]              # 源注视角度 (pitch, yaw)
  target_gaze:      [B, 2]              # 目标注视角度
  head_pose:        [B, 2]              # 头部姿态角度

处理流程:
  1. AppearanceEncoder(source_eye_crops)
     → [feat_L0, feat_L1, feat_L2]      # 三个尺度的身份特征图

  2. DeltaGazeEmbedder(source_gaze, target_gaze, head_pose)
     → [emb_0, emb_1, emb_2]            # 每个 UNet 阶段的注视嵌入

  3. PersonalizationHead(source_eye_crops)
     → {attn_biases, unet_films}         # 每用户的个性化调制

  4. GazeTransformDecoder(source_eye_crops, gaze_embs, app_features, person_mods)
     → generated_eyes [B, 3, H, W×2]     # 残差预测: clamp(input + Δ)

输出:
  generated_eyes: [B, 3, H, W×2]
```

## 3. 核心创新点

### 3.1 Delta-Gaze 条件表示

**问题:** 绝对角度条件 `f(θ_target)` 无法表达 "保持不变" (Δ=0) 这一自然先验.

**解决:** 使用相对角度差 `Δgaze = target_gaze - source_gaze` 作为主要条件输入:

$$\text{input} = \text{SinEncode}(\Delta\text{gaze}) \oplus \text{SinEncode}(\text{source\_gaze}) \oplus \text{SinEncode}(\text{head\_pose})$$

**正弦位置编码:** 每个标量 → $2K$ 维 (K=freq_bands), 使相近角度有相似表示, 支持连续插值.

**优势:**
- $\Delta\text{gaze} = 0$ 自然映射到身份变换 (恒等映射)
- 小角度变化 → 小特征差异, 更稳定的训练
- 泛化到训练集未见的角度组合

### 3.2 交叉注意力融合

**问题:** FiLM (scale/shift) 仅在通道维度调制, 无法表达"关注源眼的左上角虹膜纹理"这种空间选择性.

**解决:** 在解码器的每个分辨率层, 注视流 (Query) 与外观流 (Key/Value) 进行交叉注意力:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{(Q + \beta_{\text{gaze}}) K^T}{\sqrt{d}}\right) V$$

其中 $\beta_{\text{gaze}}$ 是注视条件的查询偏置, 控制"在哪个方向寻找外观参考".

```text
GazeTransformDecoder (32×64)          AppearanceEncoder (32×64)
        ↓ Norm → Q                          ↓ Norm → K, V
        └──────── Cross-Attention ──────────┘
                       ↓
              Gaze AdaLN FFN
                       ↓
                 Updated Features
```

**关键设计:**
- **可学习门控 (learnable gate):** 初始化为 0, 训练初期交叉注意力是 skip-through, 避免干扰
- **注视条件查询偏置:** $Q + \beta_\text{gaze}$ 使得不同注视方向关注外观流的不同空间位置
- **Per-user 注意力偏置:** PersonalizationHead 生成 per-head 偏置, 调整个体关注模式

### 3.3 多尺度身份特征

**问题:** 当前 SubjectAdapter 将整个眼部压缩为 128 维向量, 丢失空间细节.

**解决:** AppearanceEncoder 在三个尺度输出完整特征图:

| 层级 | 分辨率 (H×W) | 通道 | 捕获信息 |
|------|-------------|------|---------|
| Level 0 | 64×128 | 32 | 虹膜纹理, 睫毛, 皮肤细节 |
| Level 1 | 32×64 | 64 | 眼睛形状, 眼睑轮廓 |
| Level 2 | 16×32 | 128 | 整体眼部结构, 面部上下文 |

解码器在对应分辨率通过交叉注意力访问这些特征, 实现分辨率匹配的身份信息传递.

### 3.4 个性化注意力偏置

**问题:** 当前 SubjectAdapter 的 FiLM 仅调制通道增益, 表达力有限.

**解决:** PersonalizationHead 生成 per-head 注意力偏置 $b \in \mathbb{R}^{H_{\text{heads}} \times 1 \times 1}$:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}} + b_{\text{person}}\right) V$$

每个注意力头对应不同的关注模式 (如: 头1关注虹膜, 头2关注眼白, 头3关注眼睑), 个性化偏置调节每个人不同头的激活强度.

**Phase 2 微调:** 仅训练 PersonalizationHead (~200K 参数), 冻结其余所有组件.

## 4. 训练策略

### 4.1 损失函数

| 损失 | 权重 | 启用步数 | 说明 |
|------|------|---------|------|
| L1 重建 | 2.0 | 0 | 眼部裁剪像素级重建 |
| **Delta-zero 一致性** | 0.5 | 500 | **新增:** Δgaze=0 时输出应等于输入 |
| 注视感知损失 | 0.08 | 100 | 冻结 VGG 注视估计器的方向一致性 |
| ID 损失 | 0.1 | 1000 | ArcFace 身份保持 (贴回脸后比较) |
| GAN | 0.3 | 2000 | Multi-Scale Discriminator |

### 4.2 Delta-zero 一致性损失 (新增)

当 `target_gaze == source_gaze` 时, 网络应输出与输入相同的眼部图像:

$$\mathcal{L}_{\Delta 0} = \| G(x, \Delta g = 0) - x \|_1$$

这提供了强正则化:
- 确保网络在无需改变时不引入伪影
- 约束残差预测在 Δ=0 附近的行为
- 为 Delta-gaze 表示提供锚点

### 4.3 两阶段训练

**Phase 1 — 联合预训练 (多用户):**
- 训练所有组件: AppearanceEncoder + GazeTransformDecoder + DeltaGazeEmbedder + PersonalizationHead
- 400K 步, 学习率 1e-4, AdamW
- LR 调度: warmup → flat → cosine (GAN 加入后衰减)

**Phase 2 — 个性化微调 (单用户):**
- 冻结 AppearanceEncoder + GazeTransformDecoder + DeltaGazeEmbedder
- 仅训练 PersonalizationHead (~200K 参数)
- 300-1000 步, 学习率 5e-5

### 4.4 快速推理路径

```python
# 初始化 (每用户一次):
app_feats = model.encode_appearance(ref_eye)     # 缓存外观特征
person = model.personal_head(ref_eye)             # 缓存个性化参数

# 每帧推理:
output = model.gaze_decoder(
    source_eye, gaze_embs,
    app_features=app_feats,                        # 复用缓存
    attn_biases=person['attn_biases'],             # 复用缓存
)
```

## 5. 与现有架构对比

| 特性 | EyeOnlyGazeDiC (v3) | **DualStreamGazeNet** |
|------|---------------------|----------------------|
| 身份表征 | 128-d 全局向量 (SubjectAdapter) | **多尺度空间特征图** |
| 身份注入 | FiLM (通道 scale/shift) | **交叉注意力 (空间选择性)** |
| 注视条件 | 绝对角度 MLP | **Delta-gaze + 正弦编码** |
| 身份-注视解耦 | 隐式 (共享 UNet) | **显式 (双流)** |
| 正则化 | 无 | **Delta-zero 一致性** |
| 个性化参数 | ~430K (SubjectAdapter) | **~200K (PersonalizationHead)** |
| 个性化机制 | FiLM scale/shift | **注意力偏置** |
| 模型总参数 | ~3M | **~2.7M** |

## 6. 预期优势与风险

### 预期优势

1. **更好的身份保持:** 多尺度空间特征 + 交叉注意力, 虹膜纹理等细节不再被压缩
2. **更精确的注视控制:** Delta-gaze 表示使小角度调整更稳定
3. **更高效的个性化:** ~200K 参数 (vs ~430K), 更快收敛
4. **更鲁棒的泛化:** Delta-zero 正则化约束网络在已知锚点的行为

### 风险与缓解

| 风险 | 缓解措施 |
|------|---------|
| 交叉注意力增加计算量 (~15%) | 仅在解码器 2 个分辨率层使用, 用可学习门控渐进启用 |
| 外观编码器可能过拟合训练用户 | Phase 1 使用大规模多用户数据; 外观编码器无注视信号防止记忆 |
| Delta-gaze 在大角度差异时可能不稳定 | 同时输入源注视和头部姿态提供锚点; 正弦编码平滑化 |
| 训练初期交叉注意力可能不稳定 | 门控零初始化: 训练初期 skip-through, 逐步学习融合 |

## 7. 文件结构

```text
models/
├── dual_stream_gaze_net.py     # 新架构: DualStreamGazeNet
├── gaze_dic.py                 # 原架构: EyeOnlyGazeDiC (复用基础组件)
├── gazenet.py                  # GazeMLP (DualStream 不使用, 保留兼容)
├── subject_adapter.py          # SubjectAdapter (DualStream 用 PersonalizationHead 替代)
└── facetoeyenet.py             # FaceToEyeNet (独立架构, 不受影响)

configs/training/
└── dual_stream_gaze_net.yaml   # DualStreamGazeNet 训练配置

train_dual_stream.py            # DualStreamGazeNet 训练脚本
```
