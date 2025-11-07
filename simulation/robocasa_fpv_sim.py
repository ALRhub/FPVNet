import logging

import einops
import numpy as np
import torch
import wandb
from omegaconf import ListConfig
from robocasa.utils.env_utils import create_env
from termcolor import cprint
from tqdm import tqdm

from agents.base_agent import BaseAgent
from custom_robocasa.env_wrappers.point_cloud_sampling_wrapper import (
    PointCloudSamplingWrapper,
)
from custom_robocasa.env_wrappers.point_cloud_wrapper import PointCloudWrapper
from custom_robocasa.env_wrappers.robosuite_wrapper import RobosuiteWrapper
from custom_robocasa.env_wrappers.segmentation_wrapper import SegmentationWrapper
from custom_robocasa.env_wrappers.segmented_point_cloud_sampling_wrapper import (
    SegmentedPointCloudSamplingWrapper,
)
from custom_robocasa.utils.point_cloud.sampling.fps_pc_sampler import (
    FPSPointCloudSampler,
)
from simulation.base_sim import BaseSim
from utils.pc_grouping import Group

log = logging.getLogger(__name__)

from robocasa.utils.dataset_registry import SINGLE_STAGE_TASK_DATASETS
from environments.robocasa_env_groups import ENV_GROUPS


class RoboCasaSim(BaseSim):
    def __init__(
        self,
        env_group: str,
        camera_names: ListConfig[str],
        img_height: int,
        img_width: int,
        num_episode: int,
        seed: int,
        device: str,
        render: bool = True,
        if_vision: bool = False,
        pc_num_group: int = 256,
        pc_group_size: int = 32,
        sampled_pc_size: int = 4096,
        use_pc_color=False,
        normalize_neighborhood=False,
    ):
        assert env_group in ENV_GROUPS

        super().__init__(seed, device, render, 1, if_vision)

        self.num_episode = num_episode
        self.camera_names = list(camera_names)

        self.env_names = ENV_GROUPS[env_group]

        self.pc_group_divider = Group(
            pc_num_group, pc_group_size, normalize_neighborhood
        )

        self.pos_key = "robot0_base_to_eef_pos"
        self.quat_key = "robot0_eef_quat"

        self.img_height = img_height
        self.img_width = img_width
        self.pc_num_points = sampled_pc_size

        cprint(f"Using position key: {self.pos_key}", "blue")
        cprint(f"Using quaternion key: {self.quat_key}", "blue")

        self.use_pc_color = use_pc_color

    def test_agent(self, agent: BaseAgent):

        for env_name in self.env_names:

            print(f"Initializing environment {env_name}")

            horizon = SINGLE_STAGE_TASK_DATASETS[env_name]["horizon"]
            env = self._init_env(
                env_name,
                self.img_width,
                self.img_height,
                self.render,
                self.pc_num_points,
                0,
                False,
            )

            success_count = 0
            task_completion_hold_count = -1

            for i in range(self.num_episode):
                obs = env.reset()

                if self.render:
                    env.render()

                agent.reset()

                lang = env.get_ep_meta()["lang"]

                for j in tqdm(range(horizon)):
                    obs_dict = {}
                    obs_dict["lang"] = lang

                    gripper_state = torch.from_numpy(
                        obs["robot0_gripper_qpos"][:1]
                    ).float()
                    obs_dict["gripper_state"] = einops.rearrange(
                        gripper_state, "d -> 1 1 d"
                    ).to(self.device)

                    eef_pos = torch.from_numpy(obs[self.pos_key]).float()
                    obs_dict["eef_pos"] = einops.rearrange(eef_pos, "d -> 1 1 d").to(
                        self.device
                    )

                    eef_quat = torch.from_numpy(obs[self.quat_key]).float()
                    obs_dict["eef_quat"] = einops.rearrange(eef_quat, "d -> 1 1 d").to(
                        self.device
                    )

                    sampled_point_cloud = torch.from_numpy(
                        obs["sampled_point_cloud"][
                            :, : 3 if not self.use_pc_color else 6
                        ]
                    ).float()
                    neighborhood, center = self.pc_group_divider(
                        sampled_point_cloud.unsqueeze(0)
                    )
                    obs_dict["pc_neighborhood"] = (
                        einops.rearrange(neighborhood, "b g n d -> b 1 g n d")
                        .float()
                        .to(self.device)
                    )
                    obs_dict["pc_center"] = (
                        einops.rearrange(center, "b g d -> b 1 g d")
                        .float()
                        .to(self.device)
                    )

                    for cam_name in self.camera_names:
                        rgb = (
                            torch.from_numpy(obs[f"{cam_name}_image"].copy())
                            .float()
                            .permute(2, 0, 1)
                            / 255.0
                        )
                        rgb = einops.rearrange(rgb, "c h w -> 1 1 c h w").to(
                            self.device
                        )
                        obs_dict[f"{cam_name}_image"] = rgb

                    action = agent.predict(obs_dict).cpu().numpy()

                    action = np.concatenate([action, np.array([0, 0, 0, 0, -1])])

                    obs, _, done, _ = env.step(action)

                    if self.render:
                        env.render()

                    if env._check_success():
                        if task_completion_hold_count > 0:
                            task_completion_hold_count -= (
                                1  # latched state, decrement count
                            )
                        else:
                            task_completion_hold_count = (
                                10  # reset count on first success timestep
                            )
                    else:
                        task_completion_hold_count = (
                            -1
                        )  # null the counter if there's no success

                    if task_completion_hold_count == 0:
                        success_count += 1
                        done = True

                    if done:
                        break

            success_rate = success_count / self.num_episode
            print(f"Success rate: {success_rate}")

            wandb.log({f"{env_name}_average_success": success_rate})

            env.close()

    def _init_env(
        self,
        env_name,
        img_width,
        img_height,
        render,
        pc_num_points,
        obj_max_num_points,
        use_segmented_point_cloud,
    ):
        base_env = create_env(
            env_name=env_name,
            camera_widths=img_width,
            camera_heights=img_height,
            camera_names=self.camera_names,
            has_renderer=render,
            seed=self.seed,
        )

        if use_segmented_point_cloud:
            env = SegmentedPointCloudSamplingWrapper(
                PointCloudSamplingWrapper(
                    PointCloudWrapper(
                        SegmentationWrapper(
                            RobosuiteWrapper(base_env),
                            env_name=env_name,
                        ),
                        global_frame=False,
                        get_segmented_pc=True,
                    ),
                    pc_sampler=FPSPointCloudSampler(),
                    num_points=pc_num_points,
                ),
                env_name=env_name,
                obj_sampler=FPSPointCloudSampler(),
                rest_sampler=FPSPointCloudSampler(),
                num_points=pc_num_points,
                obj_max_num_points=obj_max_num_points,
            )
        else:
            env = PointCloudSamplingWrapper(
                PointCloudWrapper(RobosuiteWrapper(base_env), global_frame=False),
                pc_sampler=FPSPointCloudSampler(),
                num_points=pc_num_points,
            )

        return env
