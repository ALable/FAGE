import torch
import torch.nn as nn


import torch
import torch.nn as nn

class MLPBlock(nn.Module):
    def __init__(self, num_layers, num_in, num_hidden, num_out=384, 
                 non_linear=nn.GELU,
                 use_ln=True):        
        super(MLPBlock, self).__init__()

        layers = []
        current_num_in = num_in
        
        for i in range(num_layers):
            is_last_layer = (i == num_layers - 1)
            out_dim = num_out if is_last_layer else num_hidden
            
            layers.append(nn.Linear(current_num_in, out_dim))
            if not is_last_layer:
                if use_ln:
                    layers.append(nn.LayerNorm(out_dim))
                layers.append(non_linear())
            
            current_num_in = out_dim

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.net(x)


class MLPNetwork(nn.Module):
    def __init__(self, num_layers, num_in, num_hidden, num_out=384,
                 non_linear=nn.LeakyReLU, non_linear_last=None,
                 cross_condition=False):
        """
        Args:
            cross_condition: if True, gaze_mlp receives cat([gaze, head.detach()])
                             as input, improving head-gaze joint modeling.
                             Set False (default) for checkpoint compatibility.
        """
        super(MLPNetwork, self).__init__()
        self.cross_condition = cross_condition

        self.head_mlp = MLPBlock(num_layers, num_in, num_hidden, num_out, non_linear, non_linear_last)
        # When cross_condition=True, gaze sees head context (head compensates gaze).
        # .detach() prevents gradient flowing back through head_mlp via gaze path.
        gaze_in = num_in + num_out if cross_condition else num_in
        self.gaze_mlp = MLPBlock(num_layers, gaze_in, num_hidden, num_out, non_linear, non_linear_last)

    def forward(self, head_angles, gaze_angles):
        """
        Args:
            head_angles: (B, num_in)
            gaze_angles: (B, num_in)
        Returns:
            (head_embedding, gaze_embedding) each (B, num_out)
        """
        head_embedding = self.head_mlp(head_angles)
        if self.cross_condition:
            gaze_input = torch.cat([gaze_angles, head_embedding.detach()], dim=-1)
        else:
            gaze_input = gaze_angles
        gaze_embedding = self.gaze_mlp(gaze_input)
        return head_embedding, gaze_embedding

