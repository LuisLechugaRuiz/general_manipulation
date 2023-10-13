from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import site
from tqdm import tqdm
import argparse

from act.tmp.utils import compute_dict_mean, detach_dict

from rvt.utils.peract_utils import DATA_FOLDER

from general_manipulation.helpers.act_dataset import ACTDataset
from general_manipulation.utils.load_agents import get_act_agent
import general_manipulation.config.act_train_config as act_train_cfg


# TODO: Test it with small number of iterations to verify that it works
class ACTTrain(object):
    def __init__(self, args):
        self.stage = args.stage

        train_cfg = act_train_cfg.get_cfg_defaults()
        self.training_iterations = train_cfg.training_iterations
        self.validation_iterations = train_cfg.val_iterations
        self.seed = train_cfg.seed
        self.epochs = train_cfg.epochs

        device = train_cfg.device
        tasks = train_cfg.tasks
        batch_size_train = train_cfg.batch_size_train
        num_workers = train_cfg.num_workers

        rvt_package_path = self.get_package_path("rvt")
        if rvt_package_path:
            rvt_data_folder = f"{rvt_package_path}/rvt/{DATA_FOLDER}"
            self.ckpt_dir = f"{rvt_data_folder}/act_checkpoint/stage{self.stage}"
            train_replay_storage_dir = f"{rvt_package_path}/rvt/replay/replay_train"
            test_replay_storage_dir = f"{rvt_package_path}/rvt/replay/replay_val"
        else:
            raise RuntimeError("rvt is not installed!!")

        self.train_dataset = ACTDataset(
            tasks,
            batch_size_train,
            train_replay_storage_dir,
            rvt_data_folder,
            train_cfg.num_train,
            num_workers,
            True,
            self.training_iterations,
            self.ckpt_dir,
            device,
        )

        self.test_dataset = ACTDataset(
            tasks,
            batch_size_train,
            test_replay_storage_dir,
            rvt_data_folder,
            train_cfg.num_val,
            num_workers,
            False,
            self.validation_iterations,
            self.ckpt_dir,
            device,
        )

        cos_dec_max_step = self.epochs * self.training_iterations
        self.act_agent = get_act_agent(
            norm_stats=self.train_dataset.norm_stats,
            device=device,
            cos_dec_max_step=cos_dec_max_step,
        )
        self.act_agent.build(training=True, device=device)

    def train_bc(self):
        best_ckpt_info = self.train_act()
        best_epoch, min_val_loss, best_state_dict = best_ckpt_info

        # save best checkpoint
        ckpt_path = os.path.join(self.ckpt_dir, "policy_best.ckpt")
        torch.save(best_state_dict, ckpt_path)
        print(f"Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}")

    def train_act(
        self
    ):
        train_history = []
        validation_history = []
        min_val_loss = np.inf
        best_ckpt_info = None

        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
        ebar = tqdm(range(self.epochs))
        for epoch in ebar:
            print(f"\nEpoch {epoch}")
            # training
            self.act_agent.act_model.train()
            tbar = tqdm(range(self.training_iterations))
            for batch_idx in tbar:
                forward_dict = self.update_agent(eval=False)
                loss = forward_dict["loss"]
                train_history.append(detach_dict(forward_dict))
                tbar.set_description(f"Loss: {loss.item():.4f}")
            epoch_summary = compute_dict_mean(
                train_history[(batch_idx + 1) * epoch : (batch_idx + 1) * (epoch + 1)]
            )
            epoch_train_loss = epoch_summary["loss"]
            ebar.set_description(f"Train loss: {epoch_train_loss:.5f}")

            # validation
            with torch.inference_mode():
                self.act_agent.act_model.eval()
                epoch_dicts = []
                for _ in tqdm(range(self.validation_iterations)):
                    forward_dict = self.update_agent(eval=True)
                    epoch_dicts.append(forward_dict)
                epoch_summary = compute_dict_mean(epoch_dicts)
                validation_history.append(epoch_summary)

                epoch_val_loss = epoch_summary["loss"]
                if epoch_val_loss < min_val_loss:
                    min_val_loss = epoch_val_loss
                    best_ckpt_info = (
                        epoch,
                        min_val_loss,
                        deepcopy(self.act_agent.act_model.state_dict()),
                    )
            ebar.set_description(f"Val loss:   {epoch_val_loss:.5f}")

            ckpt_path = os.path.join(self.ckpt_dir, f"policy_epoch_{epoch}_seed_{self.seed}.ckpt")
            torch.save(self.act_agent.act_model.state_dict(), ckpt_path)
            self.plot_history(train_history, validation_history, epoch, self.ckpt_dir, self.seed)

        ckpt_path = os.path.join(self.ckpt_dir, "policy_last.ckpt")
        torch.save(self.act_agent.act_model.state_dict(), ckpt_path)

        best_epoch, min_val_loss, best_state_dict = best_ckpt_info
        ckpt_path = os.path.join(self.ckpt_dir, f"policy_epoch_{best_epoch}_seed_{self.seed}.ckpt")
        torch.save(best_state_dict, ckpt_path)
        print(
            f"Training finished:\nSeed {self.seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}"
        )

        # save training curves
        self.plot_history(train_history, validation_history, self.ckpt_dir, seed)

        return best_ckpt_info

    def update_agent(self, eval):
        if eval:
            data = self.test_dataset.get_data()
        else:
            data = self.train_dataset.get_data()
        if self.stage == 0:
            target_pose = data["action_last_pose"]
        elif self.stage == 1:
            target_pose = data["target_pose"]
        else:
            raise Exception("Unknown stage!")
        forward_dict = self.act_agent.update(
            observation=data, target_pose=target_pose, eval=eval
        )
        return forward_dict

    def plot_history(self, train_history, validation_history, seed):
        # save training curves
        for key in train_history[0]:
            plot_path = os.path.join(self.ckpt_dir, f"train_val_{key}_seed_{self.seed}.png")
            plt.figure()
            train_values = [summary[key].item() for summary in train_history]
            val_values = [summary[key].item() for summary in validation_history]
            plt.plot(
                np.linspace(0, self.epochs - 1, len(train_history)),
                train_values,
                label="train",
            )
            plt.plot(
                np.linspace(0, self.epochs - 1, len(validation_history)),
                val_values,
                label="validation",
            )
            # plt.ylim([-0.1, 1])
            plt.tight_layout()
            plt.legend()
            plt.title(key)
            plt.savefig(plot_path)
        print(f"Saved plots to {self.ckpt_dir}")

    def get_package_path(self, package_name):
        for site_package_dir in site.getsitepackages():
            # Check for direct package directory (usual case)
            potential_path = os.path.join(site_package_dir, package_name)
            if os.path.exists(potential_path):
                return potential_path

            # Check for egg-link (editable installs)
            egg_link = os.path.join(site_package_dir, f"{package_name}.egg-link")
            if os.path.exists(egg_link):
                with open(egg_link, "r") as f:
                    return f.readline().strip()
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", type=int, default=0,
    )
    args = parser.parse_args()

    ACTTrain(args).train_bc()
