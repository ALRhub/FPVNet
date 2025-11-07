import hydra
from omegaconf import DictConfig
from torch import nn
import torch


class PointMLPResNetEncoder(nn.Module):
    def __init__(
        self,
        point_mlp_encoder: DictConfig,
        resnet_encoder: DictConfig,
        pc_emb_as_cond: bool,
        img_emb_as_cond: bool,
    ):
        super().__init__()

        self.point_mlp_encoder = hydra.utils.instantiate(point_mlp_encoder)
        self.resnet_encoder = hydra.utils.instantiate(resnet_encoder)
        self.pc_emb_as_cond = pc_emb_as_cond
        self.img_emb_as_cond = img_emb_as_cond

    def forward(self, obs_dict, lang_cond=None):
        pc_emb = self.point_mlp_encoder(obs_dict)
        resnet_emb = self.resnet_encoder(obs_dict, lang_cond)

        if self.pc_emb_as_cond and self.img_emb_as_cond:
            pc_emb = torch.max(pc_emb, dim=-2, keepdim=True)[0]
            resnet_emb = torch.max(resnet_emb, dim=-2, keepdim=True)[0]
            return torch.cat([resnet_emb, pc_emb], dim=-2)

        if self.pc_emb_as_cond:
            pc_emb = torch.max(pc_emb, dim=-2, keepdim=True)[0]
            return torch.cat([resnet_emb, pc_emb], dim=-2)

        if self.img_emb_as_cond:
            resnet_emb = torch.max(resnet_emb, dim=-2, keepdim=True)[0]
            return torch.cat([pc_emb, resnet_emb], dim=-2)

        return torch.cat([pc_emb, resnet_emb], dim=-2)
