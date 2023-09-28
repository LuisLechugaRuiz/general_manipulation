from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import site
from tqdm import tqdm

from act.tmp.utils import compute_dict_mean, detach_dict

from rvt.utils.peract_utils import CAMERAS, DATA_FOLDER

from general_manipulation.act_dataset import ACTDataset
from general_manipulation.utils import get_act_agent, load_rvt_agent


def main():
    device = "cuda:0"

    # From config: -> TODO: GET FROM CONFIG!!
    BATCH_SIZE_TRAIN = 2
    NUM_TRAIN = 100
    NUM_VAL = 25
    NUM_WORKERS = 3
    NUM_IMAGES = 5
    EPOCHS = 3
    tasks = ["push_buttons"]  # Just testing from now.
    VAL_ITERATIONS = 100
    TRAINING_ITERATIONS = 20000  # Previously: int(10000 // (BATCH_SIZE_TRAIN / 16))

    rvt_package_path = get_package_path("rvt")
    if rvt_package_path:
        RVT_DATA_FOLDER = f"{rvt_package_path}/rvt/{DATA_FOLDER}"
        CKPT_DIR = f"{RVT_DATA_FOLDER}/act_checkpoint"
        TRAIN_REPLAY_STORAGE_DIR = f"{rvt_package_path}/rvt/replay/replay_train"
        TEST_REPLAY_STORAGE_DIR = f"{rvt_package_path}/rvt/replay/replay_val"
    else:
        raise RuntimeError("rvt is not installed!!")

    rvt_agent = load_rvt_agent(device=device)
    rvt_agent.load_clip()
    train_dataset = ACTDataset(
        rvt_agent,
        tasks,
        BATCH_SIZE_TRAIN,
        TRAIN_REPLAY_STORAGE_DIR,
        RVT_DATA_FOLDER,
        NUM_TRAIN,
        NUM_IMAGES,
        NUM_WORKERS,
        True,
        TRAINING_ITERATIONS,
        CKPT_DIR,
        device,
    )

    test_dataset = ACTDataset(
        rvt_agent,
        tasks,
        BATCH_SIZE_TRAIN,
        TEST_REPLAY_STORAGE_DIR,
        RVT_DATA_FOLDER,
        NUM_VAL,
        NUM_IMAGES,
        NUM_WORKERS,
        False,
        TRAINING_ITERATIONS,
        CKPT_DIR,
        device,
    )

    config = {
        "num_epochs": EPOCHS,
        "task_name": tasks[0],
        "temporal_agg": True,
        "camera_names": CAMERAS,
        "ckpt_dir": CKPT_DIR,
        "seed": 0,
    }
    cos_dec_max_step = EPOCHS * TRAINING_ITERATIONS
    act_agent = get_act_agent(device=device, cos_dec_max_step=cos_dec_max_step)
    act_agent.build(training=True, device=device)

    best_ckpt_info = train_bc(
        act_agent,
        train_dataset,
        test_dataset,
        config,
        TRAINING_ITERATIONS,
        VAL_ITERATIONS,
    )
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info

    # save best checkpoint
    ckpt_path = os.path.join(CKPT_DIR, "policy_best.ckpt")
    torch.save(best_state_dict, ckpt_path)
    print(f"Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}")


def train_bc(
    act_agent,
    train_dataset,
    val_dataset,
    config,
    training_iterations,
    val_iterations,
):
    num_epochs = config["num_epochs"]
    ckpt_dir = config["ckpt_dir"]
    seed = config["seed"]

    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    ebar = tqdm(range(num_epochs))
    for epoch in ebar:
        print(f"\nEpoch {epoch}")
        # training
        act_agent.act_model.train()
        tbar = tqdm(range(training_iterations))
        for batch_idx in tbar:
            data = train_dataset.get_data()
            heatmap = data["heatmap"].squeeze(1)
            heatmap = heatmap.view(
                heatmap.shape[0] * heatmap.shape[1],
                heatmap.shape[2],
                heatmap.shape[3],
                heatmap.shape[4],
            )
            forward_dict = act_agent.update(
                observation=data, heatmap=heatmap, eval=False
            )
            loss = forward_dict["loss"]
            train_history.append(detach_dict(forward_dict))
            tbar.set_description(f"Loss: {loss.item():.4f}")
        epoch_summary = compute_dict_mean(
            train_history[(batch_idx + 1) * epoch : (batch_idx + 1) * (epoch + 1)]
        )
        epoch_train_loss = epoch_summary["loss"]
        ebar.set_description(f"Train loss: {epoch_train_loss:.5f}")
        summary_string = ""
        for k, v in epoch_summary.items():
            summary_string += f"{k}: {v.item():.3f} "
        # print(summary_string)

        # validation
        with torch.inference_mode():
            act_agent.act_model.eval()
            epoch_dicts = []
            for _ in tqdm(range(val_iterations)):
                data = val_dataset.get_data()
                forward_dict = act_agent.update(
                    observation=data, hm=data["heatmap"], eval=True
                )
                epoch_dicts.append(forward_dict)
            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)

            epoch_val_loss = epoch_summary["loss"]
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                best_ckpt_info = (
                    epoch,
                    min_val_loss,
                    deepcopy(act_agent.act_model.state_dict()),
                )
        ebar.set_description(f"Val loss:   {epoch_val_loss:.5f}")
        summary_string = ""
        for k, v in epoch_summary.items():
            summary_string += f"{k}: {v.item():.3f} "
        # print(summary_string)

        if epoch % 100 == 0:
            ckpt_path = os.path.join(ckpt_dir, f"policy_epoch_{epoch}_seed_{seed}.ckpt")
            torch.save(act_agent.act_model.state_dict(), ckpt_path)
            plot_history(train_history, validation_history, epoch, ckpt_dir, seed)

    ckpt_path = os.path.join(ckpt_dir, "policy_last.ckpt")
    torch.save(act_agent.act_modell.state_dict(), ckpt_path)

    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f"policy_epoch_{best_epoch}_seed_{seed}.ckpt")
    torch.save(best_state_dict, ckpt_path)
    print(
        f"Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}"
    )

    # save training curves
    plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed)

    return best_ckpt_info


def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    # save training curves
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f"train_val_{key}_seed_{seed}.png")
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        val_values = [summary[key].item() for summary in validation_history]
        plt.plot(
            np.linspace(0, num_epochs - 1, len(train_history)),
            train_values,
            label="train",
        )
        plt.plot(
            np.linspace(0, num_epochs - 1, len(validation_history)),
            val_values,
            label="validation",
        )
        # plt.ylim([-0.1, 1])
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
    print(f"Saved plots to {ckpt_dir}")


def get_package_path(package_name):
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
    main()
