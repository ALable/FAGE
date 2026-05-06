# **LocalGaze-Net: 个性化实时视线重定向网络说明文件**

## 1. 任务定义 (Task Definition)输入: 256x256 RGB 全脸图像 + 3D 头部姿态 (Head Pose) + 目标视线 (Target Gaze)。输出: 64x128 目标视线下的眼部区域图像 (含双眼拼接或单眼)。核心目标: 保持用户身份特征 (Identity) 的同时，实现高精度视线偏转，且推理延迟需满足直播级实时要

## 2. 输入/输出张量规格 (I/O Specifications)维度名称形状 (Shape)数据类型描述Input_Face[B, 3, 256, 256]Float32标准化后的全脸图像 (归一化至 [-1, 1])Head_Pose[B, 3]Float32相机坐标系下的 [Pitch, Yaw, Roll] (弧度)Target_Gaze[B, 2]Float32相机坐标系下的绝对 [Pitch, Yaw] (弧度)Output_Eye[B, 3, 64, 128]Float32重定向后的局部眼部图像

## 3. 网络架构详细配置 (Architecture Config)

### 3.1 全局特征编码器 (Global Encoder)Backbone: RepVGG-A0 (或 A1 视算力而定)。特性: 训练时为多分支，推理时重参数化为单路 3x3 卷积，极大提升吞吐量。下采样率: 总下采样 16x，最终全脸特征图大小为 [B, C, 16, 16]

### 3.2 局部特征裁剪 (ROI Align)机制: 根据输入的人脸关键点 (Landmarks)，在 Encoder 的中间特征图上进行 ROI Align。输出大小: 裁剪出代表眼部区域的特征块，建议尺寸为 [B, C, 4, 8]

### 3.3 视线调制器 (FiLM Modulator)输入向量: $\Delta Gaze = Gaze_{target} - Head_{pose}[:2]$。MLP 结构: * Input: [sin, cos] 编码后的 $\Delta Gaze$ 与 $Head_{pose}$。Hidden: 256-wide FC layers (2层)。Output: 为每个通道输出一组 $(\gamma, \beta)$。注入方式: $F_{out} = \gamma \cdot F_{roi} + \beta$。3.4 局部眼部解码器 (Local Decoder)结构: 3层轻量级上采样模块。上采样方式: PixelShuffle (比传统反卷积更快且能减少棋盘格伪影)。输出层: Tanh 激活函数

### 4. 个性化离线微调 (LoRA Configuration)为了实现个性化，在以下层注入 LoRA 旁路：注入位置: Decoder 的所有 Conv2d 层。LoRA Rank ($r$): 建议 r=8。Alpha: 16

策略: 1.  加载 Base Model 权重并冻结。2.  仅对特定用户数据训练 LoRA 矩阵。3.  推理部署前进行权重合并: $W = W_{base} + \frac{\alpha}{r}(A \times B)$。5. 训练损失函数 (Loss Functions)Reconstruction Loss ($L_{rec}$): $L_1(I_{gen}, I_{gt})$。Perceptual Loss ($L_{vgg}$): 基于预训练 VGG-16 的特征一致性。Gaze Consistency Loss ($L_{gaze}$): * 使用冻结的 L2CS-Net 作为评估器。$L_{gaze} = ||Estimator(I_{gen}) - Gaze_{target}||^2$。Adversarial Loss ($L_{adv}$): * 使用 PatchGAN 判别器，仅针对 64x128 区域进行判别。6. 数据标准化流水线 (Normalization Pipeline)在送入网络前，必须对 GazeCapture 或其他数据集执行以下操作：虚拟相机旋转: 消除头部 Roll 角，使双眼中心连线水平。距离缩放: 将人脸缩放至虚拟相机平面 600mm 处。插值控制: 强制使用 Bicubic 或 Lanczos 插值，严禁使用 Bilinear，以防 Base Model 学习到模糊的纹理。7. 部署建议 (Deployment Optimization)框架: 建议使用 TensorRT (NVIDIA) 或 CoreML (iOS)。量化: 模型在合并 LoRA 权重后，支持 INT8 量化 以进一步压低延迟。并行化: 头部姿态计算 (SolvePnP) 与特征编码前处理建议放在 CPU 多线程或 GPU Shader 中并行执行。
