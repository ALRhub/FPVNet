import json
import os

import einops
import h5py
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path

from environments.dataset.base_dataset import TrajectoryDataset

import robocasa
from robocasa.utils.dataset_registry import SINGLE_STAGE_TASK_DATASETS
import robocasa.macros as macros

from pytorch3d.ops import sample_farthest_points

from utils.pc_grouping import Group
from environments.robocasa_env_groups import ENV_GROUPS


class RobocasaDataset(TrajectoryDataset):
    def __init__(
        self,
        cam_names: list[str],
        env_group: str,
        device: str = "cpu",
        obs_dim: int = 20,
        action_dim: int = 7,
        obs_seq_len: int = 1,
        window_size: int = 1,
        img_width: int = 128,
        img_height: int = 128,
        pc_num_group: int = 256,
        pc_group_size: int = 32,
        sampled_pc_size: int = 4096,
        use_pc_color: bool = False,
        normalize_neighborhood: bool = False,
    ):
        assert env_group in ENV_GROUPS

        super().__init__(
            device=device,
            obs_dim=obs_dim,
            action_dim=action_dim,
            window_size=window_size,
        )

        pc_group_divider = Group(pc_num_group, pc_group_size, normalize_neighborhood)

        self.envs_data = []
        max_horizon = 0

        if macros.DATASET_BASE_PATH is None:
            ds_base_path = os.path.join(
                Path(robocasa.__path__[0]).parent.absolute(), "datasets"
            )
        else:
            ds_base_path = macros.DATASET_BASE_PATH

        for env_name in ENV_GROUPS[env_group]:
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
            "pc_neighborhood": [],
            "pc_center": [],
            "robot0_agentview_left_image": [],
            "robot0_agentview_right_image": [],
            "robot0_eye_in_hand_image": [],
        }

        self.obs_seq_len = obs_seq_len

        pos_key = "robot0_base_to_eef_pos"
        quat_key = "robot0_base_to_eef_quat"
        action_key = "actions"

        self.slices = []


        i = 0
        for env_data in self.envs_data:
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
                action[:demo_length] = env_data[demo][action_key][:, : self.action_dim]
                mask[:demo_length] = 1

                self.data["gripper_state"].append(gripper_state)
                self.data["eef_pos"].append(eef_pos)
                self.data["eef_quat"].append(eef_quat)
                self.data["action"].append(action)
                self.data["lang"].append(
                    json.loads(env_data[demo].attrs["ep_meta"])["lang"]
                )
                self.data["mask"].append(mask)

                point_cloud = (
                    torch.from_numpy(env_data[demo]["obs"]["point_cloud"][:])
                    .float()
                    .to(device)
                )
                rgbs = (
                    einops.rearrange(
                        point_cloud[:, :, 3:],
                        "t (num_cam h w) c -> t num_cam c h w",
                        num_cam=len(cam_names),
                        h=img_height,
                        w=img_width,
                    )
                    / 255.0
                )

                for cam_idx, cam_name in enumerate(cam_names):
                    self.data[f"{cam_name}_image"].append(rgbs[:, cam_idx])

                with torch.no_grad():
                    if not use_pc_color:
                        point_cloud = point_cloud[:, :, :3]
                    point_cloud = point_cloud.to("cuda")

                    _, sampled_indices = sample_farthest_points(
                        points=point_cloud[:, :, :3],
                        K=sampled_pc_size,
                    )

                    sampled_point_cloud = point_cloud.gather(
                        1,
                        sampled_indices.unsqueeze(-1).expand(
                            -1, -1, 6 if use_pc_color else 3
                        ),
                    )

                    neighborhood, center = pc_group_divider(sampled_point_cloud)
                    neighborhood = neighborhood.float().to(device)
                    center = center.float().to(device)

                self.data["pc_neighborhood"].append(neighborhood)
                self.data["pc_center"].append(center)

                i += 1

        for key in self.data:
            if key == "lang":
                continue

            if "image" not in key and "pc" not in key:
                self.data[key] = (
                    torch.from_numpy(np.stack(self.data[key], axis=0))
                    .float()
                    .to(device)
                )

    def get_all_actions(self):
        result = []

        for i in range(len(self.data["mask"])):
            T = int(self.data["mask"][i].sum().item())
            result.append(self.data["action"][i, :T, :])

        return torch.cat(result, dim=0)

    def get_all_observations(self):
        assert self.use_sampled_point_cloud and self.use_full_point_cloud

        point_clouds = einops.rearrange(
            self.data["sampled_point_cloud"],
            "num_demos seq_len num_points d -> (num_demos seq_len num_points) d",
        )

        gripper_states = einops.rearrange(
            self.data["gripper_state"], "num_demos seq_len d -> (num_demos seq_len) d"
        )
        eef_poss = einops.rearrange(
            self.data["eef_pos"], "num_demos seq_len d -> (num_demos seq_len) d"
        )
        eef_quats = einops.rearrange(
            self.data["eef_quat"], "num_demos seq_len d -> (num_demos seq_len) d"
        )
        robot_states = torch.cat([gripper_states, eef_poss, eef_quats], dim=1)

        return {
            "point_cloud": point_clouds.to(self.device),
            "agent_pos": robot_states.to(self.device),
        }

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        i, start, end = self.slices[idx]

        action = self.data["action"][i, start + self.obs_seq_len - 1 : end]

        gripper_state = self.data["gripper_state"][i, start : start + self.obs_seq_len]
        eef_pos = self.data["eef_pos"][i, start : start + self.obs_seq_len]
        eef_quat = self.data["eef_quat"][i, start : start + self.obs_seq_len]

        robot_states = torch.cat([gripper_state, eef_pos, eef_quat], dim=1)

        pc_neighborhood = self.data["pc_neighborhood"][i][
            start : start + self.obs_seq_len
        ]

        pc_center = self.data["pc_center"][i][start : start + self.obs_seq_len]

        left_img = self.data["robot0_agentview_left_image"][i][
            start : start + self.obs_seq_len
        ]
        right_img = self.data["robot0_agentview_right_image"][i][
            start : start + self.obs_seq_len
        ]
        eye_in_hand_img = self.data["robot0_eye_in_hand_image"][i][
            start : start + self.obs_seq_len
        ]

        obs = {
            "gripper_state": gripper_state,
            "eef_pos": eef_pos,
            "eef_quat": eef_quat,
            "robot_states": robot_states,
            "pc_neighborhood": pc_neighborhood,
            "pc_center": pc_center,
            "robot0_agentview_left_image": left_img,
            "robot0_agentview_right_image": right_img,
            "robot0_eye_in_hand_image": eye_in_hand_img,
        }

        obs["lang"] = self.data["lang"][i]

        return obs, action
