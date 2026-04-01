import torch
import torch.nn as nn
from omegaconf import OmegaConf
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
# from torch.optim.lr_scheduler import CosineAnnealingLR
# from loss.discriminator import MultiScaleDiscriminator,DiscriminatorFullModel
# import loss.vgg_face as vgg_face

# import torchvision.transforms.functional as TFF
from src.Face_models.encoders.model_irse import Backbone

# gan loss for patch gan
def discriminator_loss(real, fake, device):
        GANLoss = nn.BCEWithLogitsLoss(reduction='mean')
        real_size = list(real.size())
        fake_size = list(fake.size())
        real_label = torch.zeros(real_size, dtype=torch.float32).to(device)
        fake_label = torch.ones(fake_size, dtype=torch.float32).to(device)

        discriminator_loss = (GANLoss(fake, fake_label) + GANLoss(real, real_label)) / 2

        return discriminator_loss

def generator_loss(fake,device):
        GANLoss = nn.BCEWithLogitsLoss(reduction='mean')
        fake_size = list(fake.size())
        fake_label = torch.zeros(fake_size, dtype=torch.float32).to(device)
        return GANLoss(fake, fake_label)

class IDLoss(nn.Module):
    def __init__(self, multiscale=True):
        super(IDLoss, self).__init__()
        print('Loading ResNet ArcFace')
        self.multiscale = multiscale
        self.face_pool_1 = torch.nn.AdaptiveAvgPool2d((256, 256))
        
        # 🔥 加载 ArcFace ResNet50 模型
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        self.facenet.load_state_dict(torch.load("/home/xuhy/PycharmProjects/Unet-Gaze/checkpoints/arcface/model_ir_se50.pth"))
        
        self.face_pool_2 = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()
        self.set_requires_grad(False)  # 冻结参数

    def set_requires_grad(self, flag=True):
        for p in self.parameters():
            p.requires_grad = flag
    
    
    def extract_feats(self, x):
        # 图像预处理       
        x = self.face_pool_1(x) if x.shape[2] != 256 else x  # 调整到 256x256
        # x = x[:, :, 35:223, 32:220]  # 裁剪人脸区域
        x = self.face_pool_2(x)  # 调整到 112x112
        
        # 🔥 提取多尺度特征
        x_feats = self.facenet(x, multi_scale=self.multiscale)
        return x_feats
    
    def forward(self, x):
        x_feats_ms = self.extract_feats(x)
        return x_feats_ms[-1]  # 返回最后一层特征 (512维)



class Interpolate(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=None):
        super(Interpolate, self).__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, input):
        return F.interpolate(input, self.size, self.scale_factor, self.mode, self.align_corners)

def set_requires_grad(net, requires_grad=False):
    if net is not None:
        for param in net.parameters():
            param.requires_grad = requires_grad

def pitchyaw_to_vector(pitchyaws):
    sin = torch.sin(pitchyaws)
    cos = torch.cos(pitchyaws)
    return torch.stack([cos[:, 0] * sin[:, 1], sin[:, 0], cos[:, 0] * cos[:, 1]], 1)


def nn_angular_distance(a, b):
    sim = F.cosine_similarity(a, b, eps=1e-6)
    sim = F.hardtanh(sim, -1.0, 1.0)
    return torch.acos(sim) * (180 / np.pi)

def gaze_angular_loss(y, y_hat):
    y = pitchyaw_to_vector(y)
    y_hat = pitchyaw_to_vector(y_hat)
    loss = nn_angular_distance(y, y_hat)
    return torch.mean(loss)

