import logging
import random
import hydra
import numpy as np

import wandb
from omegaconf import DictConfig, OmegaConf
import torch


log = logging.getLogger(__name__)

OmegaConf.register_new_resolver("add", lambda *numbers: sum(numbers))
OmegaConf.register_new_resolver("mul", lambda *numbers: np.prod(numbers))
torch.cuda.empty_cache()


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


@hydra.main(
    config_path="configs",
    config_name="robocasa_pc_group_img_config.yaml",
    version_base="1.3",
)
def main(cfg: DictConfig) -> None:

    set_seed_everywhere(cfg.seed)

    # init wandb logger and config from hydra path
    wandb.config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        group=cfg.wandb.group,
        mode=cfg.wandb.mode,
        config=wandb.config,
    )

    # train the agent
    agent = hydra.utils.instantiate(cfg.agents)
    trainer = hydra.utils.instantiate(cfg.trainers)

    if cfg.agent_name == "3da":
        agent.model.set_gripper_loc_bounds(trainer.scaler.y_bounds_tensor[:, :3])

    trainer.main(agent)

    torch.cuda.empty_cache()

    # simulate the model
    env_sim = hydra.utils.instantiate(cfg.simulation)
    env_sim.test_agent(agent)

    log.info("Training done")
    log.info("state_dict saved in {}".format(agent.working_dir))
    
    wandb.finish()


if __name__ == "__main__":
    main()
