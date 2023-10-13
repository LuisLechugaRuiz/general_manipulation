import os
import site
import yaml

import rvt.config as default_exp_cfg
import rvt.models.rvt_agent as rvt_agent
import rvt.mvt.config as default_mvt_cfg
from rvt.mvt.mvt import MVT
from rvt.utils.peract_utils import CAMERAS, SCENE_BOUNDS, IMAGE_SIZE
from rvt.utils.rvt_utils import load_agent as load_agent_state

from general_manipulation.act_agent import ACTAgent
import general_manipulation.config.act_config as default_act_cfg


def get_act_agent(norm_stats, device, cos_dec_max_step: int = 60000):
    act_cfg = default_act_cfg.get_cfg_defaults()
    act_cfg.freeze()
    act_cfg_dict = yaml.safe_load(act_cfg.dump())

    return ACTAgent(
        norm_stats=norm_stats,
        device=device,
        cfg_dict=act_cfg_dict,
        cos_dec_max_step=cos_dec_max_step,
    )


def load_rvt_agent(
    device,
    use_input_place_with_mean=False,
):
    rvt_package_path = get_package_path("rvt")
    if rvt_package_path:
        rvt_folder = f"{rvt_package_path}/rvt/"
    else:
        raise RuntimeError("rvt is not installed!!")

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

    rvt = MVT(
        renderer_device=device,
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
