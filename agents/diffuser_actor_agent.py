import logging

import dill
import einops
import hydra
from omegaconf import DictConfig
import torch

from agents.base_agent import BaseAgent
from utils.transform_utils import axisangle2quat_torch, quat2axisangle_torch

from agents.models.diffuser_actor.utils.converter import convert_multihead_flash_attention, convert_efficient_fpn

log = logging.getLogger(__name__)


class DiffuserActorAgent(BaseAgent):
    def __init__(
        self,
        model: DictConfig,
        obs_encoders: DictConfig,
        language_encoders: DictConfig,
        optimization: DictConfig,
        device: str,
        state_dim: int,
        latent_dim: int,
        obs_seq_len: int,
        act_seq_len: int,
        cam_names: list[str],
    ):
        super().__init__(
            model,
            obs_encoders,
            language_encoders,
            device,
            state_dim,
            latent_dim,
            obs_seq_len,
            act_seq_len,
            cam_names,
        )

        self.model = convert_multihead_flash_attention(self.model).to(self.device)

        self.optimizer_config = optimization
        self.use_lr_scheduler = False

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(
            self.optimizer_config, params=self.parameters()
        )
        return optimizer

    def forward(self, obs_dict, actions=None):
        assert obs_dict["point_cloud"].shape[-1] == 6, "Point cloud must have color"

        pcd_obs = obs_dict["point_cloud"][..., :3]
        pcd_obs = einops.rearrange(
            pcd_obs, "B T (num_cam H W) C -> (B T) num_cam C H W", num_cam=3, H=128, W=128
        )

        rgb_obs = obs_dict["point_cloud"][..., 3:]
        rgb_obs = einops.rearrange(
            rgb_obs, "B T (num_cam H W) C -> (B T) num_cam C H W", num_cam=3, H=128, W=128
        )

        lang_emb = self.language_encoder(obs_dict["lang"]).float()

        eef_state = torch.cat([obs_dict["eef_pos"], obs_dict["eef_quat"]], dim=-1)

        batch_size = pcd_obs.shape[0]
        trajectory_mask = torch.ones((batch_size, self.act_seq_len)).to(self.device)

        if actions is None:
            pred_actions = self.model(
                gt_trajectory=None,
                trajectory_mask=trajectory_mask,
                rgb_obs=rgb_obs,
                pcd_obs=pcd_obs,
                instruction=lang_emb,
                curr_gripper=eef_state,
                run_inference=True,
            )
            action_axis_angle = quat2axisangle_torch(pred_actions[:, :, 3:7].reshape(-1, 4)).reshape(batch_size, -1, 3)

            gripper_action = 2 * pred_actions[:, :, 7:] - 1 

            return torch.cat(
                [pred_actions[:, :, :3], action_axis_angle, gripper_action],
                dim=-1,
            )

        return self.model(
            gt_trajectory=actions,
            trajectory_mask=trajectory_mask,
            rgb_obs=rgb_obs,
            pcd_obs=pcd_obs,
            instruction=lang_emb,
            curr_gripper=eef_state,
            run_inference=actions is None,
        )

    def save_snapshot(self, save_path):
        torch.save(self.model, save_path, pickle_module=dill)

    def load_snapshot(self, save_path):
        self.model = torch.load(save_path, pickle_module=dill)
