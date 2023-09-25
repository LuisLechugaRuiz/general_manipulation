import os

import rvt.config as default_exp_cfg
import rvt.models.rvt_agent as rvt_agent
import rvt.mvt.config as default_mvt_cfg
from rvt.mvt.mvt import MVT
from rvt.utils.peract_utils import (
    CAMERAS,
    SCENE_BOUNDS,
    IMAGE_SIZE,
)
from rvt.utils.rvt_utils import load_agent as load_agent_state

import general_manipulation.act_config as default_act_cfg


def load_agent(
    model_path=None,
    exp_cfg_path=None,
    mvt_cfg_path=None,
    act_cfg_path=None,
    eval_log_dir="",
    device=0,
    use_input_place_with_mean=False,
):
    device = f"cuda:{device}"

    assert model_path is not None

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
    if act_cfg_path is not None:
        act_cfg.merge_from_file(act_cfg_path)
    else:
        act_cfg.merge_from_file(os.path.join(model_folder, "act_cfg.yaml"))

    act_cfg.freeze()
    act_cfg_dict = act_cfg.__dict__['_C'].__dict__

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
    agent.eval()

    print("Agent Information")
    print(agent)
    return agent

# TODO: Adapt this function with train act calling train act from RVT AGENT!.
def train_act():