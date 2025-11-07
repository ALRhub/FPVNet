import einops
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
        attn_model: DictConfig,
        latent_dim: int,
    ):
        super().__init__()

        self.point_mlp_encoder = hydra.utils.instantiate(point_mlp_encoder)
        self.resnet_encoder = hydra.utils.instantiate(resnet_encoder)
        self.pc_emb_as_cond = pc_emb_as_cond
        self.img_emb_as_cond = img_emb_as_cond

        if self.pc_emb_as_cond and self.img_emb_as_cond:
            self.pc_attn_model = hydra.utils.instantiate(attn_model)
            self.pc_class_token = nn.Parameter(torch.zeros(1, 1, latent_dim))

            self.img_attn_model = hydra.utils.instantiate(attn_model)
            self.img_class_token = nn.Parameter(torch.zeros(1, 1, latent_dim))

        self.attn_model = hydra.utils.instantiate(attn_model)
        self.class_token = nn.Parameter(torch.zeros(1, 1, latent_dim))

        num_input_tokens = (8 * 8 + 1) * 3  # TODO: Fix this, hardcoded for now
        self.pos_emb = nn.Parameter(torch.zeros(1, num_input_tokens, latent_dim))

    def forward(self, obs_dict, lang_cond=None):
        pc_emb = self.point_mlp_encoder(obs_dict)
        resnet_emb = self.resnet_encoder(obs_dict, lang_cond)

        resnet_emb += self.pos_emb[:, : resnet_emb.shape[1], :]

        if self.pc_emb_as_cond and self.img_emb_as_cond:
            pc_emb = torch.cat(
                (pc_emb, self.pc_class_token.expand(pc_emb.shape[0], -1, -1)), dim=1
            )
            resnet_emb = torch.cat(
                (resnet_emb, self.img_class_token.expand(resnet_emb.shape[0], -1, -1)),
                dim=1,
            )

            pc_emb = self.pc_attn_model(pc_emb)[:, -1:, :]
            resnet_emb = self.img_attn_model(resnet_emb)[:, -1:, :]

            return torch.cat([resnet_emb, pc_emb], dim=-2)

        if self.pc_emb_as_cond:
            pc_emb = torch.cat(
                (pc_emb, self.class_token.expand(pc_emb.shape[0], -1, -1)), dim=1
            )
            pc_emb = self.attn_model(pc_emb)[:, -1:, :]
            return torch.cat([resnet_emb, pc_emb], dim=-2)

        if self.img_emb_as_cond:
            resnet_emb = torch.cat(
                (resnet_emb, self.class_token.expand(resnet_emb.shape[0], -1, -1)),
                dim=1,
            )
            resnet_emb = self.attn_model(resnet_emb)[:, -1:, :]
            return torch.cat([pc_emb, resnet_emb], dim=-2)

        return torch.cat([pc_emb, resnet_emb], dim=-2)
