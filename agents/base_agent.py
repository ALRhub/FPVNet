import abc
import logging
import os
import pickle
from collections import deque

import einops
import hydra
from termcolor import cprint
import torch
import torch.nn as nn
import wandb
from omegaconf import DictConfig

from agents.utils.scaler import Scaler

# A logger for this file
log = logging.getLogger(__name__)


class BaseAgent(nn.Module, abc.ABC):

    def __init__(
        self,
        model: DictConfig,
        obs_encoders: DictConfig,
        language_encoders: DictConfig,
        device: str,
        state_dim: int,
        latent_dim: int,
        obs_seq_len: int,
        act_seq_len: int,
        cam_names: list[str]
    ):
        super().__init__()

        self.device = device
        self.working_dir = os.getcwd()
        self.scaler = None

        # Initialize model and encoder
        self.img_encoder = hydra.utils.instantiate(obs_encoders).to(device)
        self.language_encoder = hydra.utils.instantiate(language_encoders).to(device)
        self.model = hydra.utils.instantiate(model).to(device)
        self.state_emb = nn.Linear(state_dim, latent_dim)

        # print number of parameters
        cprint(f"Model: {sum(p.numel() for p in self.model.parameters())}", "green")
        cprint(f"Image Encoder: {sum(p.numel() for p in self.img_encoder.parameters())}", "green")
        cprint(f"Language Encoder: {sum(p.numel() for p in self.language_encoder.parameters())}", "green")
        cprint(f"State Embedding: {sum(p.numel() for p in self.state_emb.parameters())}", "green")

        self.cam_names = cam_names

        # for inference
        self.rollout_step_counter = 0
        self.act_seq_len = act_seq_len
        self.obs_seq_len = obs_seq_len

        self.obs_seq: dict[str, deque[torch.Tensor]] = {}

    def set_scaler(self, scaler):
        self.scaler = scaler

    # @abc.abstractmethod
    def compute_input_embeddings(self, obs_dict):
        """
        Compute the required embeddings for the visual ones and the latent goal.
        """
        #########################################
        # deal with language embedding
        #########################################
        if "lang" in obs_dict:
            obs_dict["lang_emb"] = self.language_encoder(obs_dict["lang"]).float()

        for key in obs_dict:
            if "lang" in key:
                continue
            obs_dict[key] = obs_dict[key][:, :self.obs_seq_len]

        latent_goal = obs_dict["lang_emb"]

        #########################################
        # using RGB images or point clouds
        if "pc_neighborhood" in obs_dict:
            obs_dict["pc_neighborhood"] = einops.rearrange(obs_dict["pc_neighborhood"], "b t g n d -> (b t) g n d")
            obs_dict["pc_center"] = einops.rearrange(obs_dict["pc_center"], "b t n d -> (b t) n d")

        if self.cam_names:
            B, T, C, H, W = obs_dict[f"{self.cam_names[0]}_image"].shape

        for camera in self.cam_names:
            obs_dict[f"{camera}_image"] = obs_dict[f"{camera}_image"].view(B * T, C, H, W)

        if self.if_film_condition:
            perceptual_emb = self.img_encoder(obs_dict, latent_goal)
        else:
            perceptual_emb = self.img_encoder(obs_dict)
    

        #########################################
        # add robot states
        #########################################
        if self.if_robot_states and "robot_states" in obs_dict.keys():
            robot_states = obs_dict["robot_states"]
            robot_states = self.state_emb(robot_states)

            perceptual_emb = torch.cat([perceptual_emb, robot_states], dim=1)

        return perceptual_emb, latent_goal

    @abc.abstractmethod
    def forward(self, obs_dict: dict[str, torch.Tensor], actions=None) -> torch.Tensor:
        """
        Forward pass of the model
        """
        pass

    def reset(self):
        """Resets the context of the model."""
        self.rollout_step_counter = 0
        self.obs_seq: dict[str, deque[torch.Tensor]] = {}

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
            pred_action_seq = self.scaler.inverse_scale_output(pred_action_seq)
            self.pred_action_seq = pred_action_seq

        current_action = self.pred_action_seq[0, self.rollout_step_counter]

        self.rollout_step_counter += 1
        if self.rollout_step_counter == self.act_seq_len:
            self.rollout_step_counter = 0

        return current_action

    def load_pretrained_model(self, weights_path: str, sv_name=None) -> None:
        """
        Method to load pretrained weights for the entire agent
        """
        path = os.path.join(
            weights_path,
            "model_state_dict.pth" if sv_name is None else f"{sv_name}.pth",
        )
        self.load_state_dict(torch.load(path))
        log.info("Loaded pre-trained model")
# 
    def store_model_weights(self, store_path: str, sv_name=None) -> None:
        """
        Store the weights of the entire agent
        """
        path = os.path.join(
            store_path, "model_state_dict.pth" if sv_name is None else f"{sv_name}.pth"
        )
        torch.save(self.state_dict(), path)
        log.info(f"Model saved to: {store_path}")

    def store_model_scaler(self, store_path: str, sv_name=None) -> None:
        """
        Store the model scaler inside the store path as model_scaler.pkl
        """
        save_path = os.path.join(
            store_path, "model_scaler.pkl" if sv_name is None else sv_name
        )
        with open(save_path, "wb") as f:
            pickle.dump(self.scaler, f)
        log.info(f"Model scaler saved to: {save_path}")

    def load_model_scaler(self, weights_path: str, sv_name=None) -> None:
        """
        Load the model scaler from the weights path
        """
        if sv_name is None:
            sv_name = "model_scaler.pkl"

        with open(os.path.join(weights_path, sv_name), "rb") as f:
            self.scaler = pickle.load(f)
        log.info("Loaded model scaler")

    def get_params(self):

        total_params = sum(p.numel() for p in self.parameters())

        wandb.log({"model parameters": total_params})

        log.info("The model has a total amount of {} parameters".format(total_params))

    @property
    def get_model_state_dict(self) -> dict:
        return self.state_dict()

    @property
    def get_scaler(self) -> Scaler:
        if self.scaler is None:
            raise AttributeError("Scaler has not been set. Use set_scaler() first.")
        return self.scaler

    @property
    def get_model_state(self) -> tuple[dict, Scaler]:
        if self.scaler is None:
            raise AttributeError("Scaler has not been set. Use set_scaler() first.")
        return (self.state_dict(), self.get_scaler)

    def recover_model_state(self, model_state, scaler):
        self.load_state_dict(model_state)
        self.set_scaler(scaler)
