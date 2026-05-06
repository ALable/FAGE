import torch
import torch.nn as nn
import torch.nn.functional as F

# 假设复用原有的 GroupNorm, UNetBlock, ConditionEmbedder, Downsample, Upsample
from models.gaze_dic import OverlapPatchEmbed, UNetBlock, Downsample, Upsample, DiCFinalLayer, ConditionEmbedder

class FaceToEyeNet(nn.Module):
    """
    非对称 面部->眼部 生成网络
    输入: [B, 3, 256, 256] (全脸)
    输出: [B, 3, 64, 128] (双眼拼接)
    结构: 重型编码器 (4级下采样) -> 长宽比转换瓶颈层 -> 轻量级解码器 (3级上采样)
    """
    def __init__(
        self,
        in_channels=3,
        hidden_size=64, # 提升基础通道数以增强特征提取
        gaze_dim=64,
        actfunc='gelu',
        norm_type='gnorm'
    ):
        super().__init__()
        
        # ==========================================
        # 1. Heavy Encoder (重编码器：更深，通道更多)
        # 输入: 256x256 -> 128 -> 64 -> 32 -> 16x16
        # ==========================================
        self.enc_channels = [hidden_size, hidden_size*2, hidden_size*4, hidden_size*4, hidden_size*8]
        self.x_embedder = OverlapPatchEmbed(3, 1, in_channels, self.enc_channels[0])
        
        self.gaze_embedder_enc = ConditionEmbedder(gaze_dim, self.enc_channels[4])
        
        self.enc_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        
        # 4 个 Stage 的重编码器 (depth = [2, 3, 4, 4])
        depths_enc = [2, 3, 4, 4]
        for i in range(4):
            stage = nn.ModuleList()
            for j in range(depths_enc[i]):
                in_ch = self.enc_channels[i] if j == 0 and i == 0 else self.enc_channels[i]
                stage.append(UNetBlock(
                    in_channels=in_ch, out_channels=self.enc_channels[i],
                    emb_channels=self.enc_channels[4], # 统一使用最高维条件
                    actfunc=actfunc, norm_type=norm_type
                ))
            self.enc_blocks.append(stage)
            self.downs.append(Downsample(self.enc_channels[i], self.enc_channels[i+1]))

        # ==========================================
        # 2. Bottleneck & Aspect Ratio Adapter (长宽比适配)
        # 脸部特征 [B, C, 16, 16] -> 眼部特征 [B, C, 8, 16]
        # ==========================================
        self.bottleneck_blocks = nn.ModuleList([
            UNetBlock(self.enc_channels[4], self.enc_channels[4], self.enc_channels[4], actfunc=actfunc)
            for _ in range(4) # 重型 Latent 层
        ])
        
        # 将 1:1 的面部全局特征压缩为 1:2 的眼部特征布局
        self.aspect_adapter = nn.Sequential(
            nn.Conv2d(self.enc_channels[4], self.enc_channels[4], kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.SiLU()
        )

        # ==========================================
        # 3. Light Decoder (轻解码器：较浅，通道递减快)
        # 输入: 8x16 -> 16x32 -> 32x64 -> 64x128
        # 注意：因为输入输出空间不对齐，这里*没有*横向 Skip Connection
        # ==========================================
        self.dec_channels = [self.enc_channels[4], hidden_size*4, hidden_size*2, hidden_size]
        self.ups = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        
        # 3 个 Stage 的轻解码器 (depth = [1, 1, 1])
        depths_dec = [1, 1, 1]
        for i in range(3):
            self.ups.append(Upsample(self.dec_channels[i], self.dec_channels[i+1]))
            stage = nn.ModuleList()
            for j in range(depths_dec[i]):
                stage.append(UNetBlock(
                    in_channels=self.dec_channels[i+1], out_channels=self.dec_channels[i+1],
                    emb_channels=self.enc_channels[4], 
                    actfunc=actfunc, norm_type=norm_type
                ))
            self.dec_blocks.append(stage)

        self.final_layer = DiCFinalLayer(self.dec_channels[-1], in_channels, self.enc_channels[4])

    def forward(self, full_face_input, gaze_cond):
        """
        Args:
            full_face_input: [B, 3, 256, 256] 
            gaze_cond: [B, 2, gaze_dim]
        """
        # [B, C0, 256, 256]
        x = self.x_embedder(full_face_input) 
        
        # 提取全局 Gaze Embedding
        gaze_emb = self.gaze_embedder_enc(gaze_cond)
        
        # --- Heavy Encoder ---
        for i in range(4):
            for block in self.enc_blocks[i]:
                # 此处无须依赖 SubjectAdapter，可移除 mask 和 subject_mods
                x = block(x, gaze_emb)
            x = self.downs[i](x) 
        # 此时 x 的形状通常为 [B, 512, 16, 16]

        # --- Bottleneck ---
        for block in self.bottleneck_blocks:
            x = block(x, gaze_emb)
            
        # 空间维度转换: [B, C, 16, 16] -> [B, C, 8, 16] 匹配眼部的 1:2 长宽比
        x = self.aspect_adapter(x)

        # --- Light Decoder ---
        for i in range(3):
            x = self.ups[i](x)
            for block in self.dec_blocks[i]:
                x = block(x, gaze_emb)
        # 最终 x 的形状达到 [B, C, 64, 128]

        # --- Output ---
        output_eyes = self.final_layer(x, gaze_emb) # [B, 3, 64, 128]
        return output_eyes