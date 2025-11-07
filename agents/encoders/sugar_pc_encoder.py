import os
from torch import nn
import torch

from agents.encoders.sugar.pc_transformer import MaskedPCTransformer

from pathlib import Path


class SUGARPCEncoder(nn.Module):
    def __init__(
        self,
        ckpt_path: os.PathLike,
        freeze_backbone: bool,
        device: str,
        unfreeze_last_n_layers: int = 0,
    ):
        super().__init__()
        self.pc_transformer = MaskedPCTransformer(
            input_size=6,
            hidden_size=384,
            num_heads=6,
            depth=12,
            cross_attn_input="post",
            cross_attn_layers=None,
            mask_ratio=0.0,
            mask_type="rand",
            drop_path_rate=0.2,
            detach_enc_dec=False,
            csc_skip_dec_sa=False,
        ).to(device)

        pct_pretrain_model_state_dict = torch.load(
            ckpt_path, map_location=device, weights_only=True
        )
        pc_transformer_state_dict = {}
        for param in pct_pretrain_model_state_dict:
            if param.startswith("mae_encoder"):
                pc_transformer_state_dict[param.replace("mae_encoder.", "")] = (
                    pct_pretrain_model_state_dict[param]
                )

        self.pc_transformer.load_state_dict(pc_transformer_state_dict)

        if freeze_backbone:
            for param in self.pc_transformer.parameters():
                param.requires_grad = False

        if unfreeze_last_n_layers:
            for i in range(unfreeze_last_n_layers):
                for param in self.pc_transformer.blocks[self.pc_transformer.depth - 1 - i].parameters():
                    param.requires_grad = True

    def forward(self, obs_dict):
        return self.pc_transformer(obs_dict["pc_neighborhood"], obs_dict["pc_center"])["pc_vis"]


if __name__ == "__main__":
    model_weight_path = "/home/i53/student/donat/fast_mail/agents/encoders/sugar/pretrain/objaverse4-nolvis/multiobjrandcam1-pc4096.g256.s32-mae.color.0.05-csc.l1.txt.img-openclip-scene.mae.csc.obj.ref-multi.grasp-nodetach-init.objaverse_nolvis.single/ckpts/model_step_200000.pt"
    device = "cuda"
    encoder = SUGARPCEncoder(num_groups=128, group_size=32, ckpt_path=model_weight_path, freeze_backbone=False, device=device)

    x = torch.randn(2, 1024, 6).to(device)
    out = encoder(x)
    print(out.shape)
