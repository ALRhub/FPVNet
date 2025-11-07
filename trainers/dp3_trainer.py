import copy
import logging
import os
from pathlib import Path

import hydra
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from agents.models.dp3.model.common.normalizer import LinearNormalizer

log = logging.getLogger(__name__)


class DP3Trainer:
    def __init__(
        self,
        trainset: DictConfig,
        ema_model: DictConfig,
        if_use_ema: bool,
        use_lr_scheduler: bool,
        train_batch_size: int = 512,
        num_workers: int = 8,
        device: str = "cpu",
        epoch: int = 100,
        obs_seq_len: int = 1,
        save_every_n_epochs: int = 100,
    ):

        self.trainset = hydra.utils.instantiate(trainset)
        self.train_dataloader = DataLoader(
            self.trainset,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )

        self.batch_size = train_batch_size
        self.obs_seq_len = obs_seq_len
        self.epoch = epoch
        self.device = device
        self.working_dir = os.getcwd()
        self.use_lr_scheduler = use_lr_scheduler
        self.save_every_n_epochs = save_every_n_epochs

        self.if_use_ema = if_use_ema
        self.ema_model_config = ema_model

        self.normalizer = LinearNormalizer()
        all_data = self.trainset.get_all_observations() | {
            "action": self.trainset.get_all_actions()
        }
        self.normalizer.fit(all_data, last_n_dims=1, mode="limits")

    def main(self, agent):
        agent.model.set_normalizer(self.normalizer)

        if self.if_use_ema:
            ema_agent = copy.deepcopy(agent)
            ema_agent.load_state_dict(agent.state_dict())
            ema_agent = ema_agent.to(self.device)

            self.ema = hydra.utils.instantiate(self.ema_model_config, model=ema_agent)

        # define optimizer
        if self.use_lr_scheduler:
            self.optimizer, self.scheduler = agent.configure_optimizers(num_training_steps=self.epoch * len(self.trainset) // self.batch_size)
        else:
            self.optimizer = agent.configure_optimizers(num_training_steps=self.epoch * len(self.trainset) // self.batch_size)

        for num_epoch in tqdm(range(self.epoch)):
            epoch_loss = torch.tensor(0.0).to(self.device)

            for data in tqdm(self.train_dataloader):
                obs_dict, action = data

                # put data on cuda
                for obs_key in obs_dict.keys():
                    if obs_key == 'lang':
                        continue
                    
                    obs_dict[obs_key] = obs_dict[obs_key].to(self.device).contiguous()

                action = action.to(self.device).contiguous()

                batch_loss = self.train_one_step(agent, obs_dict, action)

                epoch_loss += batch_loss
                    
            epoch_loss = epoch_loss / len(self.train_dataloader)

            wandb.log({"epoch_train_loss": epoch_loss.item()})
            log.info(
                "Epoch {}: Mean train loss is {}".format(num_epoch, epoch_loss.item())
            )
        
        agent.save_snapshot(Path(self.working_dir) / "last_model.pth")
        if self.if_use_ema:
            ema_agent.save_snapshot(Path(self.working_dir) / "last_model_ema.pth")

        log.info("training done")

    def train_one_step(self, agent, obs_dict, action):
        """Run a single training step."""
        agent.train()

        loss, loss_dict = agent(obs_dict, action)

        wandb.log({k: v.item() for k, v in loss_dict.items()})

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        if self.use_lr_scheduler:
            self.scheduler.step()

        if self.if_use_ema:
            self.ema.step(agent)

        return loss