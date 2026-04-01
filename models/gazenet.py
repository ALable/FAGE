import torch.nn as nn
import torch.nn.functional as F


class MLPBlock(nn.Module):
    def __init__(self, num_layers, num_in, num_hidden, num_out=384, non_linear=nn.LeakyReLU, non_linear_last=None):
        super(MLPBlock, self).__init__()

        layers = []
        current_num_in = num_in
        # hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(current_num_in, num_hidden))
            layers.append(non_linear())
            current_num_in = num_hidden

        # last layer
        layers.append(nn.Linear(current_num_in, num_out))
        if non_linear_last is not None:
            layers.append(non_linear_last())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class MLPNetwork(nn.Module):
    def __init__(self, num_layers, num_in, num_hidden, num_out=384, non_linear=nn.LeakyReLU, non_linear_last=None):
        super(MLPNetwork, self).__init__()

        # Create two separate MLP networks for head pose and gaze
        self.head_mlp = MLPBlock(num_layers, num_in, num_hidden, num_out, non_linear, non_linear_last)
        self.gaze_mlp = MLPBlock(num_layers, num_in, num_hidden, num_out, non_linear, non_linear_last)

    def forward(self, head_angles, gaze_angles):
        """
        Args:
            head_angles (torch.Tensor): Head pose angles of shape (B, num_in)
            gaze_angles (torch.Tensor): Gaze direction angles of shape (B, num_in)
        Returns:
            tuple: (head_embedding, gaze_embedding) each of shape (B, num_out)
        """
        head_embedding = self.head_mlp(head_angles)  # (B, num_out)
        gaze_embedding = self.gaze_mlp(gaze_angles)  # (B, num_out)

        return head_embedding, gaze_embedding

