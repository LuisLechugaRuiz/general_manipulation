from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import site
from tqdm import tqdm
import yaml

from act.tmp.utils import compute_dict_mean, detach_dict

import rvt.config as default_exp_cfg
import rvt.models.rvt_agent as rvt_agent
import rvt.mvt.config as default_mvt_cfg
from rvt.mvt.mvt import MVT
from rvt.utils.rvt_utils import load_agent as load_agent_state
from rvt.utils.peract_utils import CAMERAS, DATA_FOLDER, SCENE_BOUNDS, IMAGE_SIZE

from general_manipulation.act_dataset import ACTDataset
import general_manipulation.act_config as default_act_cfg


def main():
    device = "cuda:0"

    # From config:
    BATCH_SIZE_TRAIN = 2
    NUM_TRAIN = 100
    NUM_VAL = 25
    NUM_WORKERS = 3
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

    train_dataset = ACTDataset(
        tasks,
        BATCH_SIZE_TRAIN,
        TRAIN_REPLAY_STORAGE_DIR,
        RVT_DATA_FOLDER,
        NUM_TRAIN,
        NUM_WORKERS,
        True,
        TRAINING_ITERATIONS,
        CKPT_DIR,
        device,
    )

    test_dataset = ACTDataset(
        tasks,
        BATCH_SIZE_TRAIN,
        TEST_REPLAY_STORAGE_DIR,
        RVT_DATA_FOLDER,
        NUM_VAL,
        NUM_WORKERS,
        False,
        TRAINING_ITERATIONS,
        CKPT_DIR,
        device
    )

    config = {
        "num_epochs": EPOCHS,
        "task_name": tasks[0],
        "temporal_agg": True,
        "camera_names": CAMERAS,
        "ckpt_dir": CKPT_DIR,
        "seed": 0,
    }
    rvt_folder = f"{rvt_package_path}/rvt"
    agent = load_agent(rvt_folder, device)

    best_ckpt_info = train_bc(
        agent,
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
    agent,
    train_dataset,
    val_dataset,
    config,
    training_iterations,
    val_iterations,
):
    num_epochs = config["num_epochs"]
    ckpt_dir = config["ckpt_dir"]
    seed = config["seed"]

    optimizer = torch.optim.AdamW(
        agent._network.act_model.parameters(), lr=0.0001, weight_decay=0.0001
    )

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
        agent._network.act_model.train()
        optimizer.zero_grad()
        tbar = tqdm(range(training_iterations))
        for batch_idx in tbar:
            data = train_dataset.get_data()
            forward_dict = agent.train_act(data)
            # backward
            loss = forward_dict["loss"]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
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
            agent._network.act_model.eval()
            epoch_dicts = []
            for _ in tqdm(range(val_iterations)):
                data = val_dataset.get_data()
                forward_dict = agent.train_act(data)
                epoch_dicts.append(forward_dict)
            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)

            epoch_val_loss = epoch_summary["loss"]
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                best_ckpt_info = (
                    epoch,
                    min_val_loss,
                    deepcopy(agent._network.act_model.state_dict()),
                )
        ebar.set_description(f"Val loss:   {epoch_val_loss:.5f}")
        summary_string = ""
        for k, v in epoch_summary.items():
            summary_string += f"{k}: {v.item():.3f} "
        # print(summary_string)

        if epoch % 100 == 0:
            ckpt_path = os.path.join(ckpt_dir, f"policy_epoch_{epoch}_seed_{seed}.ckpt")
            torch.save(agent._network.act_model.state_dict(), ckpt_path)
            plot_history(train_history, validation_history, epoch, ckpt_dir, seed)

    ckpt_path = os.path.join(ckpt_dir, "policy_last.ckpt")
    torch.save(agent._network.act_model.state_dict(), ckpt_path)

    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f"policy_epoch_{best_epoch}_seed_{seed}.ckpt")
    torch.save(best_state_dict, ckpt_path)
    print(
        f"Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}"
    )

    # save training curves
    plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed)

    return best_ckpt_info


def load_agent(
    rvt_folder,
    device,
    use_input_place_with_mean=False,
):
    model_folder = f"{rvt_folder}/runs/rvt"
    model_name = "model_14.pth"
    log_name = "test/1"
    eval_log_dir = os.path.join(model_folder, "eval", log_name)
    model_path = os.path.join(model_folder, model_name)
    exp_cfg_path = None
    mvt_cfg_path = None
    eval_log_dir = None

    # load exp_cfg
    model_folder = os.path.join(os.path.dirname(model_path))

    exp_cfg = default_exp_cfg.get_cfg_defaults()
    if exp_cfg_path is not None:
        exp_cfg.merge_from_file(exp_cfg_path)
    else:
        exp_cfg.merge_from_file(os.path.join(model_folder, "exp_cfg.yaml"))

    # WARNING NOTE: a temporary hack to use place_with_mean in evaluation
    if not use_input_place_with_mean:
        exp_cfg.rvt.place_with_mean = True
    exp_cfg.freeze()

    mvt_cfg = default_mvt_cfg.get_cfg_defaults()
    if mvt_cfg_path is not None:
        mvt_cfg.merge_from_file(mvt_cfg_path)
    else:
        mvt_cfg.merge_from_file(os.path.join(model_folder, "mvt_cfg.yaml"))
    mvt_cfg.freeze()

    act_cfg = default_act_cfg.get_cfg_defaults()
    act_cfg.freeze()
    act_cfg_dict = yaml.safe_load(act_cfg.dump())

    rvt = MVT(
        renderer_device=device,
        act_cfg_dict=act_cfg_dict,
        **mvt_cfg,
    )

    agent = rvt_agent.RVTAgent(
        network=rvt.to(device),
        image_resolution=[IMAGE_SIZE, IMAGE_SIZE],
        add_lang=mvt_cfg.add_lang,
        scene_bounds=SCENE_BOUNDS,
        cameras=CAMERAS,
        log_dir=f"{eval_log_dir}/eval_run",
        **exp_cfg.peract,
        **exp_cfg.rvt,
    )

    agent.build(training=False, device=device)
    load_agent_state(model_path, agent)

    print("Agent Information")
    print(agent)
    return agent


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
