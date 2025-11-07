import json
import os

import h5py
import numpy as np
import torch
from termcolor import cprint
from pathlib import Path

from tqdm import tqdm

from environments.dataset.base_dataset import TrajectoryDataset

import robocasa
from robocasa.utils.dataset_registry import SINGLE_STAGE_TASK_DATASETS
import robocasa.macros as macros

from utils.transform_utils import axisangle2quat_numpy
from environments.robocasa_env_groups import ENV_GROUPS

class RobocasaDataset(TrajectoryDataset):
    def __init__(
        self,
        env_group: str,
        device: str = "cpu",
        obs_dim: int = 20,
        action_dim: int = 7,
        obs_seq_len: int = 1,
        window_size: int = 1,
        img_width: int = 128,
        img_height: int = 128,
    ):
        assert env_group in ENV_GROUPS

        super().__init__(
            device=device,
            obs_dim=obs_dim,
            action_dim=action_dim,
            window_size=window_size,
        )

        self.envs_data = []
        max_horizon = 0

        if macros.DATASET_BASE_PATH is None:
            ds_base_path = os.path.join(
                Path(robocasa.__path__[0]).parent.absolute(), "datasets"
            )
        else:
            ds_base_path = macros.DATASET_BASE_PATH

        for env_name in ENV_GROUPS[env_group]:
            print(f"Getting demos from {env_name}")
            ds_config = SINGLE_STAGE_TASK_DATASETS[env_name]

            max_horizon = max(max_horizon, ds_config["horizon"])

            ds_path = os.path.join(
                ds_base_path,
                ds_config["human_path"],
                f"processed_demo_{img_width}_{img_height}.hdf5",
            )

            f = h5py.File(ds_path, "r")
            self.envs_data.append(f["data"])

        self.data = {
            "gripper_state": [],
            "eef_pos": [],
            "eef_quat": [],
            "lang": [],
            "action": [],
            "mask": [],
            "point_cloud": [],
        }

        self.obs_seq_len = obs_seq_len

        pos_key = "robot0_base_to_eef_pos"
        quat_key = "robot0_base_to_eef_quat"
        action_key = "actions"

        cprint(f"Using position key: {pos_key}", "blue")
        cprint(f"Using quaternion key: {quat_key}", "blue")
        cprint(f"Using action key: {action_key}", "blue")

        self.slices = []

        i = 0
        for j, env_data in enumerate(tqdm(self.envs_data)):
            for demo in tqdm(env_data):
                demo_length = env_data[demo].attrs["num_samples"]

                for start in range(demo_length - window_size + 1):
                    self.slices.append((i, start, start + window_size))

                gripper_state = np.zeros((max_horizon, 1))
                eef_pos = np.zeros((max_horizon, 3))
                eef_quat = np.zeros((max_horizon, 4))
                action = np.zeros((max_horizon, self.action_dim))
                mask = np.zeros((max_horizon))

                gripper_state[:demo_length] = env_data[demo]["obs"][
                    "robot0_gripper_qpos"
                ][:, :1]
                eef_pos[:demo_length] = env_data[demo]["obs"][pos_key][:]
                eef_quat[:demo_length] = env_data[demo]["obs"][quat_key][:]
                mask[:demo_length] = 1

                env_action = env_data[demo][action_key][:, :7]
                action_quat = axisangle2quat_numpy(env_action[:, 3:6])
                action[:demo_length] = np.concatenate([env_action[:, :3], action_quat, env_action[:, 6:]], axis=1)
                action[:demo_length][action[:demo_length, -1] == -1, -1] = 0 # replace the gripper state of -1 with 0

                self.data["gripper_state"].append(gripper_state)
                self.data["eef_pos"].append(eef_pos)
                self.data["eef_quat"].append(eef_quat)
                self.data["action"].append(action)
                self.data["lang"].append(
                    json.loads(env_data[demo].attrs["ep_meta"])["lang"]
                )
                self.data["mask"].append(mask)

                point_cloud = torch.from_numpy(env_data[demo]["obs"]["point_cloud"][:]).float()
                point_cloud[:, :, 3:] /= 255.0
                self.data["point_cloud"].append(point_cloud)

                i += 1

        for key in self.data:
            if key == "lang" or key == "point_cloud":
                continue

            self.data[key] = torch.from_numpy(np.stack(self.data[key], axis=0)).float()

    def get_all_actions(self):
        result = []

        for i in range(len(self.data["mask"])):
            T = int(self.data["mask"][i].sum().item())
            result.append(self.data["action"][i, :T, :])
        
        return torch.cat(result, dim=0).to(self.device)

    def get_all_observations(self):
        assert self.use_sampled_point_cloud and not self.use_full_point_cloud

        result = {"point_cloud": [], "agent_pos": []}

        for i in range(len(self.data["mask"])):
            T = int(self.data["mask"][i].sum().item())

            point_cloud = self.data["sampled_point_cloud"][i, :T]
            result["point_cloud"].append(point_cloud)

            gripper_state = self.data["gripper_state"][i, :T]
            eef_pos = self.data["eef_pos"][i, :T]
            eef_quat = self.data["eef_quat"][i, :T]
            robot_states = torch.cat([gripper_state, eef_pos, eef_quat], dim=1)
            result["agent_pos"].append(robot_states)

        result["point_cloud"] = torch.cat(result["point_cloud"], dim=0).to(self.device)
        result["agent_pos"] = torch.cat(result["agent_pos"], dim=0).to(self.device)

        return result

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        i, start, end = self.slices[idx]

        action = self.data["action"][i, start + self.obs_seq_len - 1 : end]

        gripper_state = self.data["gripper_state"][i, start : start + self.obs_seq_len]
        eef_pos = self.data["eef_pos"][i, start : start + self.obs_seq_len]
        eef_quat = self.data["eef_quat"][i, start : start + self.obs_seq_len]

        robot_states = torch.cat([gripper_state, eef_pos, eef_quat], dim=1)

        point_cloud = self.data["point_cloud"][i][start : start + self.obs_seq_len]

        obs = {
            "gripper_state": gripper_state,
            "eef_pos": eef_pos,
            "eef_quat": eef_quat,
            "robot_states": robot_states,
            "point_cloud": point_cloud,
        }

        obs["lang"] = self.data["lang"][i]

        return obs, action
