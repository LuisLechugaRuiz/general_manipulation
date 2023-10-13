import numpy as np
import torch
import os
import pickle

from rvt.utils.get_dataset import get_act_dataset


class ACTDataset:
    def __init__(
        self,
        tasks,
        batch_size,
        replay_storage_dir,
        data_folder,
        num_demos,
        num_workers,
        training,
        training_iterations,
        ckpt_dir,
        device,
    ):
        super(ACTDataset).__init__()
        self.dataloader = get_act_dataset(
            tasks,
            batch_size,
            replay_storage_dir,
            data_folder,
            num_demos,
            False,
            num_workers,
            training=training,
        )
        self._iterator = iter(self.dataloader.dataset)
        self.norm_stats = self.get_norm_stats(
            training_iterations=training_iterations,
            ckpt_dir=ckpt_dir,
        )
        self.device = device

    def get_data(self):
        sample = next(self._iterator)
        batch = {
            k: torch.tensor(v).to(self.device) if isinstance(v, np.ndarray) else v.to(self.device)
            for k, v in sample.items()
            if isinstance(v, (torch.Tensor, np.ndarray))
        }
        return batch

    def get_norm_stats(self, training_iterations, ckpt_dir):
        stats_path = os.path.join(ckpt_dir, "dataset_stats.pkl")
        if os.path.exists(stats_path):
            with open(stats_path, "rb") as f:
                return pickle.load(f)

        all_qpos = []
        norm_stats = {}
        print("Training iterations", training_iterations)
        for _ in range(training_iterations):
            raw_batch = next(self._iterator)
            all_qpos.append(torch.tensor(raw_batch["joint_positions"][0][0]))

        # normalize qpos data
        all_qpos = torch.stack(all_qpos)
        qpos_mean = all_qpos.mean(dim=[0, 1], keepdim=True).squeeze()
        norm_stats["qpos_mean"] = qpos_mean
        qpos_std = all_qpos.std(dim=[0, 1], keepdim=True).squeeze()
        qpos_std = torch.clip(qpos_std, 1e-2, np.inf)  # clipping
        norm_stats["qpos_std"] = qpos_std
        print("---Norm stats---")
        print("Joint abs position mean:", qpos_mean)
        print("Joint abs position std:", qpos_std)
        # save dataset stats
        if not os.path.isdir(ckpt_dir):
            os.makedirs(ckpt_dir)
        with open(stats_path, "wb") as f:
            pickle.dump(norm_stats, f)

        return norm_stats
