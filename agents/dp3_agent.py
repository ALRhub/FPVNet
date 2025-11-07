from collections import deque
import logging

import dill
import einops
import hydra
from omegaconf import DictConfig
import torch

from agents.base_agent import BaseAgent
from agents.models.dp3.model.common.lr_scheduler import get_scheduler

log = logging.getLogger(__name__)


class DP3Agent(BaseAgent):
    def __init__(
        self,
        model: DictConfig,
        obs_encoders: DictConfig,
        language_encoders: DictConfig,
        optimization: DictConfig,
        lr_scheduler: DictConfig,
        device: str,
        state_dim: int,
        latent_dim: int,
        obs_seq_len: int,
        act_seq_len: int,
        cam_names: list[str]
    ):
        super().__init__(
            model=model,
            obs_encoders=obs_encoders,
            language_encoders=language_encoders,
            device=device,
            state_dim=state_dim,
            latent_dim=latent_dim,
            obs_seq_len=obs_seq_len,
            act_seq_len=act_seq_len,
            cam_names=cam_names
        )

        self.optimizer_config = optimization
        self.lr_scheduler_config = lr_scheduler
        
    def configure_optimizers(self, num_training_steps: int):
        optimizer = hydra.utils.instantiate(
            self.optimizer_config, params=self.parameters()
        )

        lr_scheduler = get_scheduler(
            name=self.lr_scheduler_config.name,
            optimizer=optimizer,
            num_warmup_steps=self.lr_scheduler_config.num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        return optimizer, lr_scheduler

    def forward(self, obs_dict, actions=None):
        #########################################
        # deal with language embedding
        #########################################
        if "lang" in obs_dict:
            obs_dict["lang_emb"] = self.language_encoder(obs_dict["lang"]).float()

        batch = {
            "obs": {
                "point_cloud": obs_dict["point_cloud"].to(self.device),
                "agent_pos": torch.cat(
                    [
                        obs_dict["gripper_state"],
                        obs_dict["eef_pos"],
                        obs_dict["eef_quat"],
                    ],
                    dim=-1,
                ).to(self.device),
                "lang_emb": obs_dict["lang_emb"].squeeze().to(self.device),
            },
        }

        if actions is not None:
            batch["action"] = actions.to(self.device)
            return self.model.compute_loss(batch)
        
        else:
            return self.model.predict_action(batch["obs"])["action"]
        
    @torch.no_grad()
    def predict(self, obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        # initialize self.obs_seq if empty
        if not self.obs_seq:
            for key in obs_dict.keys():
                self.obs_seq[key] = deque(maxlen=self.obs_seq_len)

        for key in obs_dict.keys():
            if "lang" in key:
                continue

            self.obs_seq[key].append(obs_dict[key])

            if key == "lang":
                continue
            obs_dict[key] = torch.concat(list(self.obs_seq[key]), dim=1)

            if obs_dict[key].shape[1] < self.obs_seq_len:
                # left pad repeated first element
                pad = einops.repeat(
                    obs_dict[key][:, 0],
                    "b ... -> b t ...",
                    t=self.obs_seq_len - obs_dict[key].shape[1],
                )
                obs_dict[key] = torch.cat([pad, obs_dict[key]], dim=1)
                
        if self.rollout_step_counter == 0:
            self.eval()

            # predict action sequence
            pred_action_seq = self(obs_dict)[:, :self.act_seq_len]
            self.pred_action_seq = pred_action_seq

        current_action = self.pred_action_seq[0, self.rollout_step_counter]

        self.rollout_step_counter += 1
        if self.rollout_step_counter == self.act_seq_len:
            self.rollout_step_counter = 0

        return current_action

    def save_snapshot(self, save_path):
        torch.save(self.model, save_path, pickle_module=dill)

    def load_snapshot(self, save_path):
        self.model = torch.load(save_path, pickle_module=dill)