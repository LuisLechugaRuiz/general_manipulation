import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import torch

from rlbench.action_modes.action_mode import ActionMode
from rlbench.backend.scene import Scene
from rvt.utils.peract_utils import CAMERAS


class ACTActionMode(ActionMode):
    """Move arm using ACT policy"""

    def __init__(self, arm_action_mode, gripper_mode, act_executor):
        super(ACTActionMode, self).__init__(arm_action_mode, gripper_mode)
        self.act_executor = act_executor
        self.key_points = {}
        self.last_action = np.zeros(9,)
        self.threshold = 0.05
        self.images = 0  # TODO: Remove when we draw_circle_on_image.

    def action(self, scene: Scene, action: np.ndarray):
        target_reached = False
        # Only fill key point if action is not equal to last, this is due to the stuck condition, remove it after.
        if self.euclidean_distance(self.last_action[:7], action[:7]) > 0.1:
            self.fill_key_point(action[:3], scene.get_observation())
        stuck_iterations = 0
        actions = 0
        prev_qpos = np.zeros(7,)
        while not target_reached:
            obs = scene.get_observation()
            obs_dict = self.get_obs_dict(obs)
            qpos = obs_dict["joint_positions"]
            pred_action = self.act_executor.step(
                obs
            ).cpu().numpy()
            self.arm_action_mode.action(scene, pred_action)
            if (
                self.euclidean_distance(obs_dict["gripper_pose"][:3], action[:3])
                < self.threshold
            ):
                target_reached = True
            if (self.euclidean_distance(qpos, prev_qpos)) < 0.02:
                stuck_iterations += 1
                if stuck_iterations > 20:
                    print("IS STUCK!!")
                    target_reached = True  # Leave the loop, retrieve new action point or fail after timeout.
            else:
                stuck_iterations = 0
            actions += 1
            prev_qpos = qpos
        ee_action = np.atleast_1d(action[7])
        self.gripper_action_mode.action(scene, ee_action)
        self.last_action = action

    def euclidean_distance(self, pos1, pos2):
        return np.linalg.norm(np.array(pos1) - np.array(pos2))

    def get_obs_dict(self, obs):
        obs_dict = vars(obs)
        obs_dict = {k: v for k, v in obs_dict.items() if v is not None}
        return obs_dict

    # TODO: Loop until euclidean distance is lower to threshold.
    def action_OLD(self, scene: Scene, action: np.ndarray):
        # Extract the flattened act_action from continuous_action starting from index 9
        flattened_act_action = action[9:]

        # Reshape it back to its original shape (20, 7)
        all_actions = flattened_act_action.reshape((20, 7))
        pred_action = self.act_executor.step(
            all_actions
        ).cpu().numpy()
        self.arm_action_mode.action(scene, pred_action)
        obs = scene.get_observation()
        obs_dict = self.get_obs_dict(obs)
        # TODO: Add gripper action to act model!
        if (
            self.euclidean_distance(obs_dict["gripper_pose"][:3], action[:3])
            < self.threshold
        ):
            ee_action = np.atleast_1d(action[7])
            self.gripper_action_mode.action(scene, ee_action)

    def get_image_OLD(self, obs):  # TODO: Call after fixes.
        all_cam_images = []
        obs_dict = self.get_obs_dict(obs)
        obs_dict = {
            k: np.transpose(v, [2, 0, 1]) if v.ndim == 3 else np.expand_dims(v, 0)
            for k, v in obs_dict.items()
            if isinstance(v, (np.ndarray, list))
        }

        for cam_name in CAMERAS:
            keypoint_channel = np.zeros_like(obs_dict['%s_rgb' % cam_name][0])
            if self.key_points[cam_name]:
                px, py = self.key_points[cam_name]
                keypoint_channel[int(py), int(px)] = 1.0
            rgba_image = np.concatenate([obs_dict['%s_rgb' % cam_name], keypoint_channel[np.newaxis, :, :]], axis=0)
            rgba = torch.from_numpy(rgba_image).float()
            all_cam_images.append(rgba)
        # construct observations
        image_data = torch.stack(all_cam_images, axis=0)
        image_data[:, :3, :, :] = image_data[:, :3, :, :] / 255.0
        return image_data

    def get_image(self, obs):
        all_cam_images = []
        obs_dict = self.get_obs_dict(obs)
        obs_dict = {
            k: np.transpose(v, [2, 0, 1]) if v.ndim == 3 else np.expand_dims(v, 0)
            for k, v in obs_dict.items()
            if isinstance(v, (np.ndarray, list))
        }

        for cam_name in CAMERAS:
            rgb = torch.from_numpy(obs_dict['%s_rgb' % cam_name]).float()
            all_cam_images.append(rgb)
        # construct observations
        image_data = torch.stack(all_cam_images, axis=0)
        image_data = image_data / 255.0
        return image_data

    # TODO: Add correct shape (with act output addition)
    def action_shape(self, scene: Scene):
        return (9,)
