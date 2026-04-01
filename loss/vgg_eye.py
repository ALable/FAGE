import torch
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from gaze_estimation.baseline_vgg import GazeHeadNet as gaze_network

trans_eval = transforms.Compose([transforms.Resize(size=(224, 224))])

# GazeRirection Loss
class GazePerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True, device=None,path=".logs/checkpoints/gazecapture_256_eye_concat"):
        """
        Init function for the gaze perceptual loss using VGG16 model.
        :resize: Boolean value that indicates if to resize the images
        """

        super(GazePerceptualLoss, self).__init__()
        self.device = device
        self.model = gaze_network().to(device)
        self.path = path 
        state_dict = torch.load(self.path, map_location=torch.device("cpu"))
        self.model.load_state_dict(state_dict=state_dict["model_state_dict"])
        self.model.eval()
        self.img_dim = 224
        for p in self.model.parameters():
            p.requires_grad = False
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        # Ensure mean and std are on the correct device
        mean_tensor = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std_tensor = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        if device is not None:
            mean_tensor = mean_tensor.to(device)
            std_tensor = std_tensor.to(device)
        self.register_buffer("mean", mean_tensor)
        self.register_buffer("std", std_tensor)


    def nn_angular_distance(self, a, b):
        sim = F.cosine_similarity(a, b, eps=1e-6)
        sim = F.hardtanh(sim, -1.0, 1.0)
        return torch.acos(sim) * (180 / np.pi)

    def pitchyaw_to_vector(self, pitchyaws):
        sin = torch.sin(pitchyaws)
        cos = torch.cos(pitchyaws)
        return torch.stack([cos[:, 0] * sin[:, 1], sin[:, 0], cos[:, 0] * cos[:, 1]], 1)

    def gaze_angular_loss(self, y, y_hat):
        y = self.pitchyaw_to_vector(y)
        y_hat = self.pitchyaw_to_vector(y_hat)
        loss = self.nn_angular_distance(y, y_hat)
        return torch.mean(loss)

    def forward(self, input, target):
        """
        Forward function that calculate a perceptual loss using the VGG16 model.
        :input: Generated image
        :target: Groundtruth image
        :feature_layers: Which layers to use from the VGG16 model
        :style_layers: Style layers to use
        :return: Returns a perceptual loss between the groundtruth and generated images
        """
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        if self.resize:
            input = trans_eval(input)
            target = trans_eval(target)

        x = input
        y = target

        gaze_x, head_x = self.model(x)

        gaze_y, head_y = self.model(y)

        angular_loss = self.gaze_angular_loss(gaze_y.detach(), gaze_x)


        return angular_loss