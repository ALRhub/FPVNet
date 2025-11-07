"""Shared utilities for all main scripts."""

import os
import logging
import wandb
from omegaconf import DictConfig
import hydra
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from agents.utils.scaler import ActionScaler, MinMaxScaler
from agents.utils.ema import ExponentialMovingAverage
from agents.base_agent import BaseAgent

log = logging.getLogger(__name__)


class BaseTrainer:
    """Basic train/test class to be inherited."""

    def __init__(
            self,
            trainset: DictConfig,
            train_batch_size: int = 512,
            num_workers: int = 8,
            device: str = 'cpu',
            epoch: int = 100,
            scale_data: bool = True,
            scaler_type: str = None,
            obs_seq_len: int = 1,
            decay_ema: float = 0.999,
            if_use_ema: bool = False,
            pin_memory: bool = True
    ):
        """Initialize."""

        self.trainset = hydra.utils.instantiate(trainset)

        self.train_dataloader = DataLoader(
            self.trainset,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
        )

        self.obs_seq_len = obs_seq_len
        self.epoch = epoch
        self.device = device
        self.working_dir = os.getcwd()
        self.scaler_type = scaler_type

        self.decay_ema = decay_ema
        self.if_use_ema = if_use_ema

        if self.scaler_type == 'minmax':
            self.scaler = MinMaxScaler(self.trainset.get_all_actions(), scale_data, device)
        else:
            self.scaler = ActionScaler(self.trainset.get_all_actions(), scale_data, device)

        log.info("Number of training samples: {}".format(len(self.trainset)))

    def main(self, agent):
        """Run main training/testing pipeline."""

        # assign scaler to agent calss
        agent.set_scaler(self.scaler)

        if self.if_use_ema:
            self.ema_helper = ExponentialMovingAverage(agent.parameters(), self.decay_ema, self.device)

        # define optimizer
        if agent.use_lr_scheduler:
            self.optimizer, self.scheduler = agent.configure_optimizers()
        else:
            self.optimizer = agent.configure_optimizers()

        for num_epoch in tqdm(range(self.epoch)):

            epoch_loss = torch.tensor(0.0).to(self.device)

            for data in tqdm(self.train_dataloader):
                obs_dict, action = data

                # put data on cuda
                for obs_key in obs_dict.keys():
                    if obs_key == 'lang':
                        continue
                    
                    obs_dict[obs_key] = obs_dict[obs_key].to(self.device).contiguous()

                action = self.scaler.scale_output(action).contiguous()

                batch_loss = self.train_one_step(agent, obs_dict, action)

                epoch_loss += batch_loss

            epoch_loss = epoch_loss / len(self.train_dataloader)

            wandb.log({"train_loss": epoch_loss.item()})
            log.info("Epoch {}: Mean train loss is {}".format(num_epoch, epoch_loss.item()))

        log.info("training done")

        if self.if_use_ema:
            self.ema_helper.store(agent.parameters())
            self.ema_helper.copy_to(agent.parameters())

        agent.store_model_weights(agent.working_dir, sv_name='last_model')

    def train_one_step(self, agent: BaseAgent, obs_dict, action):
        """Run a single training step."""
        agent.train()

        loss = agent(obs_dict, action)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        if agent.use_lr_scheduler:
            self.scheduler.step()

        if self.if_use_ema:
            self.ema_helper.update(agent.parameters())

        return loss

