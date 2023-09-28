import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

import rvt.utils.peract_utils as peract_utils
from rvt.mvt.renderer import BoxRenderer
import rvt.mvt.utils as mvt_utils
import rvt.utils.rvt_utils as rvt_utils
from rvt.utils.lr_sched_utils import GradualWarmupScheduler

from peract_colab.arm.optim.lamb import Lamb

from general_manipulation.models.act_model import ACTModel


class ACTAgent:
    def __init__(
        self,
        device,
        cfg_dict,
        cos_dec_max_step: int = 60000,
        scene_bounds: list = peract_utils.SCENE_BOUNDS,
        cameras: list = peract_utils.CAMERAS,
    ):
        self.act_model = ACTModel(cfg_dict=cfg_dict, num_img=cfg_dict["num_images"]).to(device)
        self.move_pc_in_bound = cfg_dict["move_pc_in_bound"]
        self._place_with_mean = cfg_dict["place_with_mean"]
        self.scene_bounds = scene_bounds
        self.cameras = cameras
        self.img_size = self.act_model.img_size
        self.add_corr = self.act_model.add_corr
        self.renderer = BoxRenderer(
            device=device,
            img_size=(self.img_size, self.img_size),
            with_depth=self.act_model.add_depth,
        )
        self._cos_dec_max_step = cos_dec_max_step
        self._optimizer_type = cfg_dict["optimizer_type"]
        self._lambda_weight_l2 = cfg_dict["lambda_weight_l2"]
        self._warmup_steps = cfg_dict["warmup_steps"]
        self._lr_cos_dec = cfg_dict["lr_cos_dec"]
        self._lr = cfg_dict["lr"]
        self._training = False

    def update(self, observation, heatmap, eval: bool = False):
        obs, pcd = peract_utils._preprocess_inputs(observation, self.cameras)

        with torch.no_grad():
            pc, img_feat = rvt_utils.get_pc_img_feat(
                obs,
                pcd,
            )

            pc, img_feat = rvt_utils.move_pc_in_bound(
                pc, img_feat, self.scene_bounds, no_op=not self.move_pc_in_bound
            )

            # TODO: Vectorize
            pc_new = []
            rev_trans = []
            for _pc in pc:
                a, b = mvt_utils.place_pc_in_cube(
                    _pc,
                    with_mean_or_bounds=self._place_with_mean,
                    scene_bounds=None if self._place_with_mean else self.scene_bounds,
                )
                pc_new.append(a)
                rev_trans.append(b)
            pc = pc_new

            proprio_joint_abs = observation["joint_positions"].squeeze(1)
            actions = observation["actions"].squeeze(1)
            is_pad = observation["is_pad"].squeeze(1)

            img = self.render(
                pc=pc,
                img_feat=img_feat,
                img_aug=0,
                dyn_cam_info=None,
            )

        use_grad = self._training and not eval
        with torch.set_grad_enabled(use_grad):
            act_out = self.act_model(
                img=img, heatmap=heatmap, proprio=proprio_joint_abs, actions=actions, is_pad=is_pad
            )
        if self._training:
            # backward
            loss = act_out["loss"]
            loss.backward()
            self._optimizer.step()
            self._optimizer.zero_grad()

        return act_out

    # TODO: Copied from MVT - Move to common utils when refactor.
    def render(self, pc, img_feat, img_aug, dyn_cam_info):
        with torch.no_grad():
            if dyn_cam_info is None:
                dyn_cam_info_itr = (None,) * len(pc)
            else:
                dyn_cam_info_itr = dyn_cam_info

            if self.add_corr:
                img = [
                    self.renderer(
                        _pc,
                        torch.cat((_pc, _img_feat), dim=-1),
                        fix_cam=True,
                        dyn_cam_info=(_dyn_cam_info,)
                        if not (_dyn_cam_info is None)
                        else None,
                    ).unsqueeze(0)
                    for (_pc, _img_feat, _dyn_cam_info) in zip(
                        pc, img_feat, dyn_cam_info_itr
                    )
                ]
            else:
                img = [
                    self.renderer(
                        _pc,
                        _img_feat,
                        fix_cam=True,
                        dyn_cam_info=(_dyn_cam_info,)
                        if not (_dyn_cam_info is None)
                        else None,
                    ).unsqueeze(0)
                    for (_pc, _img_feat, _dyn_cam_info) in zip(
                        pc, img_feat, dyn_cam_info_itr
                    )
                ]

            img = torch.cat(img, 0)
            img = img.permute(0, 1, 4, 2, 3)

            # image augmentation
            if img_aug != 0:
                stdv = img_aug * torch.rand(1, device=img.device)
                # values in [-stdv, stdv]
                noise = stdv * ((2 * torch.rand(*img.shape, device=img.device)) - 1)
                img = torch.clamp(img + noise, -1, 1)

        return img

    def build(self, training: bool, device: torch.device = None):
        self._training = training
        self._device = device

        if training:
            # Don't use debug during training
            self.act_model.debug = False

        if self._optimizer_type == "lamb":
            # From: https://github.com/cybertronai/pytorch-lamb/blob/master/pytorch_lamb/lamb.py
            self._optimizer = Lamb(
                self.act_model.parameters(),
                lr=self._lr,
                weight_decay=self._lambda_weight_l2,
                betas=(0.9, 0.999),
                adam=False,
            )
        elif self._optimizer_type == "adam":
            self._optimizer = torch.optim.Adam(
                self.act_model.parameters(),
                lr=self._lr,
                weight_decay=self._lambda_weight_l2,
            )
        else:
            raise Exception("Unknown optimizer")

        if self._lr_cos_dec:
            after_scheduler = CosineAnnealingLR(
                self._optimizer,
                T_max=self._cos_dec_max_step,
                eta_min=self._lr / 100,  # mininum lr
            )
        else:
            after_scheduler = None
        self._lr_sched = GradualWarmupScheduler(
            self._optimizer,
            multiplier=1,
            total_epoch=self._warmup_steps,
            after_scheduler=after_scheduler,
        )
