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
from custom_robocasa.custom_robocasa.robocasa.utils.dataset_registry import (
    SINGLE_STAGE_TASK_DATASETS,
)
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
        pc_num_points: int = 1024,
        obj_max_num_points: int = 512,
        use_full_point_cloud: bool = False,
        use_sampled_point_cloud: bool = False,
        use_segmented_point_cloud: bool = False,
        use_pc_color: bool = False,
    ):
        assert env_group in ENV_GROUPS

        super().__init__(seed, device, render, 1, if_vision)

        self.num_episode = num_episode
        self.camera_names = list(camera_names)

        self.env_names = ENV_GROUPS[env_group]

        self.use_pc_color = use_pc_color

        if use_sampled_point_cloud:
            if use_segmented_point_cloud:
                self.sampled_pc_key = "segmented_sampled_point_cloud"
            else:
                self.sampled_pc_key = "sampled_point_cloud"
        else:
            assert (
                use_full_point_cloud
            ), "At least one of sampled or full point cloud must be used"

        self.use_full_point_cloud = use_full_point_cloud
        self.use_sampled_point_cloud = use_sampled_point_cloud
        self.use_segmented_point_cloud = use_segmented_point_cloud

        self.pos_key = "robot0_base_to_eef_pos"
        self.quat_key = "robot0_base_to_eef_quat"

        self.img_height = img_height
        self.img_width = img_width
        self.pc_num_points = pc_num_points
        self.obj_max_num_points = obj_max_num_points
        self.use_segmented_point_cloud = use_segmented_point_cloud

        cprint(f"Using position key: {self.pos_key}", "blue")
        cprint(f"Using quaternion key: {self.quat_key}", "blue")

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
                self.obj_max_num_points,
                self.use_segmented_point_cloud,
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

                    if self.use_full_point_cloud:
                        point_cloud = torch.from_numpy(obs["point_cloud"]).float()

                        if self.use_sampled_point_cloud:
                            point_cloud[:, 3:] /= 255.0

                            sampled_point_cloud = torch.from_numpy(
                                obs[self.sampled_pc_key]
                            ).float()
                            if not self.use_pc_color:
                                sampled_point_cloud = sampled_point_cloud[:, :3]
                            else:
                                sampled_point_cloud[:, 3:] /= 255.0

                            obs_dict["point_cloud"] = einops.rearrange(
                                point_cloud, "n d -> 1 1 n d"
                            ).to(self.device)
                            obs_dict["sampled_point_cloud"] = einops.rearrange(
                                sampled_point_cloud, "n d -> 1 1 n d"
                            ).to(self.device)

                        else:
                            if not self.use_pc_color:
                                point_cloud = point_cloud[:, :3]
                            else:
                                point_cloud[:, 3:] /= 255.0

                            obs_dict["point_cloud"] = einops.rearrange(
                                point_cloud, "n d -> 1 1 n d"
                            ).to(self.device)

                    elif self.use_sampled_point_cloud:
                        point_cloud = torch.from_numpy(obs[self.sampled_pc_key]).float()
                        if not self.use_pc_color:
                            point_cloud = point_cloud[:, :3]
                        else:
                            point_cloud[:, 3:] /= 255.0

                        obs_dict["point_cloud"] = einops.rearrange(
                            point_cloud, "n d -> 1 1 n d"
                        ).to(self.device)
                    else:
                        raise ValueError("Invalid point cloud configuration")

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
