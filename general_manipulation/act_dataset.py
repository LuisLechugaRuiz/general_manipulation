import numpy as np
import torch
import os
import pickle

from rvt.utils.get_dataset import get_act_dataset
from rvt.utils.peract_utils import (
    CAMERAS,
)

# TODO: REFACTOR! NORMALIZE DATA HERE BEFORE RETURNING THE SAMPLE
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
            device=device,
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
        #return self._retrieve_data(sample)

    # TODO: REMOVE
    def _retrieve_data(self, sample):
        qpos = torch.from_numpy(sample["qpos"].squeeze(1))
        actions = torch.from_numpy(sample["actions"].squeeze(1))
        is_pad = torch.from_numpy(sample["is_pad"].squeeze(1))
        # new axis for different cameras
        all_cam_images = []
        for cam_name in CAMERAS:
            # rgba = torch.from_numpy(sample["%s_rgba" % cam_name].squeeze(1)) TODO: Enable after fix.
            # all_cam_images.append(rgba)
            rgb = torch.from_numpy(sample["%s_rgb" % cam_name].squeeze(1))
            all_cam_images.append(rgb)
        # construct observations
        image_data = torch.stack(all_cam_images, axis=1)

        # normalize only RGB channels
        mean = self.norm_stats["joint_abs_position_mean"]
        std = self.norm_stats["joint_abs_position_std"]
        # image_data[:, :, :3, :, :] = image_data[:, :, :3, :, :] / 255.0 TODO: Enable after fix.
        image_data = image_data / 255.0
        qpos = (qpos - mean) / std
        actions = (actions - mean) / std

        # self.save_image(image_data.numpy())
        return image_data, qpos, actions, is_pad

    def _reset_iterator(self):
        self._iterator = iter(self._replay_dataset)

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
            all_qpos.append(torch.tensor(raw_batch["qpos"].squeeze(1)))

        # normalize qpos data
        all_qpos = torch.stack(all_qpos)
        qpos_mean = all_qpos.mean(dim=[0, 1], keepdim=True).squeeze()
        norm_stats["joint_abs_position_mean"] = qpos_mean
        qpos_std = all_qpos.std(dim=[0, 1], keepdim=True).squeeze()
        qpos_std = torch.clip(qpos_std, 1e-2, np.inf)  # clipping
        norm_stats["joint_abs_position_std"] = qpos_std
        print("---Norm stats---")
        print("Joint abs position mean:", qpos_mean)
        print("Joint abs position std:", qpos_std)
        # save dataset stats
        if not os.path.isdir(ckpt_dir):
            os.makedirs(ckpt_dir)
        with open(stats_path, "wb") as f:
            pickle.dump(norm_stats, f)

        return norm_stats
