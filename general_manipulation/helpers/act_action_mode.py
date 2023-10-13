import numpy as np
import torch

from rlbench.action_modes.action_mode import ActionMode
from rlbench.backend.scene import Scene


class ACTActionMode(ActionMode):
    """Move arm using ACT policy"""

    def __init__(self, arm_action_mode, gripper_mode, state_dim, num_queries, device):
        super(ACTActionMode, self).__init__(arm_action_mode, gripper_mode)
        self.key_points = {}
        self.last_action = np.zeros(
            9,
        )
        self.threshold = 0.02
        self.all_time_actions = torch.zeros(
            [num_queries, num_queries, state_dim]
        ).cuda()
        self.num_queries = num_queries
        self.device = device

    def action(self, scene: Scene, action: np.ndarray):
        joint_positions_seq = action[9:].reshape(1, 20, 7)
        joint_positions_seq = torch.tensor(joint_positions_seq).to(self.device).float()
        # print("JOINT POSITIONS SEQ:", joint_positions_seq.shape)
        pred_action = self._temporal_ensembling(joint_positions_seq).cpu().numpy()
        # pred_action = action[:7]
        # print("PRED POSE:", action[:3])
        # print("CURRENT GRIPPER POSE:", scene.get_observation().gripper_pose[:3])
        # print("PRED ACTION:", pred_action)
        try:
            self.arm_action_mode.action(scene, pred_action)
        except Exception as e:
            print("ERROR:", e)
        # while (
        #    self.euclidean_distance(scene.robot.arm.get_joint_positions(), pred_action)
        #    > 0.001
        #):
        #    scene.step()
        obs = scene.get_observation()
        # print("OUT")
        if self.euclidean_distance(obs.gripper_pose[:3], action[:3]) < self.threshold:
            print("TARGET REACHED!")
            ee_action = np.atleast_1d(action[7])
            self.gripper_action_mode.action(scene, ee_action)

    def _temporal_ensembling(self, all_actions):
        # Shift all existing actions to the left in each sequence
        self.all_time_actions[:, :-1] = self.all_time_actions[:, 1:]

        # Append the new all_actions to the end of the buffer along dimension 0
        self.all_time_actions = torch.cat((self.all_time_actions, all_actions), dim=0)

        # If the buffer is too large along dimension 0, remove the oldest all_actions tensor
        if self.all_time_actions.shape[0] > self.num_queries:
            self.all_time_actions = self.all_time_actions[1:]

        # Filter actions
        actions_for_curr_step = self.all_time_actions[:, 0]
        actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
        actions_for_curr_step = actions_for_curr_step[actions_populated]

        # Calculate the exponential weights and the raw action as before
        k = 0.01
        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
        exp_weights = exp_weights / exp_weights.sum()
        exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
        return raw_action.squeeze(0)

    # TODO: REMOVE
    def action_OLD(self, scene: Scene, action: np.ndarray):
        target_reached = False
        stuck_iterations = 0
        prev_qpos = np.zeros(
            7,
        )

        print("UPDATING ACTION!")
        joint_positions_seq = action[9:]
        joint_positions_seq = joint_positions_seq.reshape(1, 20, 7)

        while not target_reached:
            obs = scene.get_observation()
            obs_dict = self.get_obs_dict(obs)
            pred_action = self.act_executor.step(obs_dict, action[:7]).cpu().numpy()
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
            print("OBS GRIPPER POSE:", obs.gripper_pose)
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
