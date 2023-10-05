import torch
import numpy as np


class ACTExecutor:
    def __init__(
        self, act_agent, state_dim, num_queries, max_size=100
    ):
        # self.norm_stats = norm_stats
        self.num_queries = num_queries
        self.max_size = max_size
        self.all_time_actions = torch.zeros(
            [num_queries, num_queries, state_dim]
        ).cuda()
        self.act_agent = act_agent

    # TODO: Move this to ACT Agent (add auxiliar class to normalize data)
    def _pre_process(self, joint_positions):
        return (
            joint_positions - self.norm_stats["joint_abs_position_mean"].cuda()
        ) / self.norm_stats["joint_abs_position_std"].cuda()

    def _post_process(self, action):
        return (
            action * self.norm_stats["joint_abs_position_std"].cuda()
            + self.norm_stats["joint_abs_position_mean"].cuda()
        )

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
        return raw_action

    def step(self, observation, target_pose):
        target_pose = torch.tensor(target_pose).cuda().float().unsqueeze(0).unsqueeze(0)
        a_hat, is_pad_hat, [mu, logvar] = self.act_agent.update(observation, target_pose)
        raw_action = self._temporal_ensembling(a_hat)
        raw_action = raw_action.squeeze(0).cuda()
        return raw_action
