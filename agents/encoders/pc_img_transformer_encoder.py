import hydra
from omegaconf import DictConfig
from torch import nn
import torch

class PcImgTransformerEncoder(nn.Module):
    def __init__(
        self,
        pc_encoder: DictConfig,
        img_encoder: DictConfig,
        pc_emb_dim: int,
        img_emb_dim: int,
        img_num_tokens: int,
        use_global_pc_emb: bool,
        use_global_img_emb: bool,
        latent_dim: int,
        attn_model: DictConfig,
    ):
        super().__init__()

        self.pc_encoder = hydra.utils.instantiate(pc_encoder)
        self.img_encoder = hydra.utils.instantiate(img_encoder)

        self.use_global_pc_emb = use_global_pc_emb
        self.use_global_img_emb = use_global_img_emb

        self.img_pos_emb = nn.Parameter(torch.zeros(1, img_num_tokens, latent_dim))

        self.pc_emb_mapping = nn.Linear(pc_emb_dim, latent_dim)
        self.img_emb_mapping = nn.Linear(img_emb_dim, latent_dim)
        
        if use_global_pc_emb:
            self.pc_global_feature_model = hydra.utils.instantiate(attn_model)
            self.pc_class_token = nn.Parameter(torch.zeros(1, 1, latent_dim))

        if use_global_img_emb:
            self.img_global_feature_model = hydra.utils.instantiate(attn_model)
            self.img_class_token = nn.Parameter(torch.zeros(1, 1, latent_dim))

    def forward(self, obs_dict, lang_cond=None):
        pc_emb = self.pc_encoder(obs_dict)
        pc_emb = self.pc_emb_mapping(pc_emb)
        
        img_emb = self.img_encoder(obs_dict, lang_cond)
        img_emb = self.img_emb_mapping(img_emb)
        img_emb += self.img_pos_emb

        if self.use_global_pc_emb:
            pc_emb = torch.cat(
                (pc_emb, self.pc_class_token.expand(pc_emb.shape[0], -1, -1)), dim=1
            )
            pc_emb = self.pc_global_feature_model(pc_emb)[:, -1:, :]

        if self.use_global_img_emb:
            img_emb = torch.cat(
                (img_emb, self.img_class_token.expand(img_emb.shape[0], -1, -1)), dim=1
            )
            img_emb = self.img_global_feature_model(img_emb)[:, -1:, :]

        # If there is a global embedding, the last token should be a global embedding
        # If none is global embedding or both are global embeddings, order doesn't matter
        if self.use_global_pc_emb:
            return torch.cat([img_emb, pc_emb], dim=-2)
        else:
            return torch.cat([pc_emb, img_emb], dim=-2)