import numpy as np
import torch

from rlbench.action_modes.action_mode import ActionMode
from rlbench.backend.scene import Scene


class ACTActionMode(ActionMode):
    """Move arm using ACT policy"""

    def __init__(self, arm_action_mode, gripper_mode, act_executor):
        super(ACTActionMode, self).__init__(arm_action_mode, gripper_mode)
        self.act_executor = act_executor
        self.key_points = {}
        self.last_action = np.zeros(
            9,
        )
        self.threshold = 0.05
        self.stuck_threshold = 0.07
        self.device = act_executor.act_agent.device

    def action(self, scene: Scene, action: np.ndarray):
        target_reached = False
        stuck_iterations = 0
        prev_qpos = np.zeros(
            7,
        )

        print("UPDATING ACTION!")
        while not target_reached:
            obs = scene.get_observation()
            obs_dict = self.get_obs_dict(obs)
            pred_action = self.act_executor.step(obs_dict, action[:3]).cpu().numpy()
            self.arm_action_mode.action(scene, pred_action)
            while (
                self.euclidean_distance(
                    scene.robot.arm.get_joint_positions(), pred_action
                )
                > 0.001
            ):
                scene.step()
            obs = scene.get_observation()  # Get obs again after step?
            # DEBUG
            # print("DISTANCE IS:", self.euclidean_distance(obs.gripper_pose[:3], action[:3]))
            if (
                self.euclidean_distance(obs.gripper_pose[:3], action[:3])
                < self.threshold
            ):
                target_reached = True
                print("TARGET REACHED!")
            qpos = obs.joint_positions
            if (self.euclidean_distance(qpos, prev_qpos)) < 0.02:
                stuck_iterations += 1
                if stuck_iterations > 30:
                    print("IS STUCK!!")
                    target_reached = True  # Leave the loop, retrieve new action point or fail after timeout.
            else:
                stuck_iterations = 0
            prev_qpos = qpos
        ee_action = np.atleast_1d(action[7])
        self.gripper_action_mode.action(scene, ee_action)
        self.last_action = action

    def euclidean_distance(self, pos1, pos2):
        return np.linalg.norm(np.array(pos1) - np.array(pos2))

    def get_obs_dict(self, obs):
        obs_dict = vars(obs)
        new_obs_dict = {}

        for k, v in obs_dict.items():
            if v is None:
                continue  # Skip None values
            if isinstance(v, (np.ndarray, list)):
                if v.ndim == 3:
                    v = np.transpose(v, [2, 0, 1])
                v = torch.tensor(np.array([v]), device=self.device).unsqueeze(0)
                if v.dtype == torch.double:
                    v = v.float()
            new_obs_dict[k] = v

        return new_obs_dict

    def action_shape(self, scene: Scene):
        return (9,)
