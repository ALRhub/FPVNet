from torch import nn
import torch


class PointMLPEncoder(nn.Module):
    def __init__(
        self,
        use_pc_color: bool,
        out_channels: int,
        use_layer_norm: bool,
        use_final_layer_norm: bool,
        use_max_pool: bool,
    ):
        super().__init__()
        self.use_pc_color = use_pc_color
        self.in_channels = 6 if use_pc_color else 3

        block_channel = [64, 128, 256]

        self.mlp = nn.Sequential(
            nn.Linear(self.in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layer_norm else nn.Identity(),
            nn.ReLU(),
        )

        if use_final_layer_norm:
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], out_channels),
                nn.LayerNorm(out_channels)
            )
        else:
            self.final_projection = nn.Linear(block_channel[-1], out_channels)

        self.use_max_pool = use_max_pool

    def forward(self, obs_dict):
        x = obs_dict["pc_neighborhood"]

        assert (
            x.shape[-1] == self.in_channels
        ), f"Use pc color: {self.use_pc_color}, but input channels: {x.shape[-1]}"

        x = self.mlp(x)
        if self.use_max_pool:
            x = torch.max(x, -2)[0]
        x = self.final_projection(x)
        return x
