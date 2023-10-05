import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

import rvt.utils.peract_utils as peract_utils
from rvt.mvt.renderer import BoxRenderer
import rvt.mvt.utils as mvt_utils
import rvt.utils.rvt_utils as rvt_utils
from rvt.utils.lr_sched_utils import GradualWarmupScheduler

from peract_colab.arm.optim.lamb import Lamb

from general_manipulation.models.act_model import ACTModel
from general_manipulation.utils.video_recorder import VideoRecorder

import numpy as np
from scipy.spatial.transform import Rotation as R


class ACTAgent:
    def __init__(
        self,
        device,
        cfg_dict,
        cos_dec_max_step: int = 60000,
        scene_bounds: list = peract_utils.SCENE_BOUNDS,
        cameras: list = peract_utils.CAMERAS,
    ):
        self.device = device
        self.act_model = ACTModel(cfg_dict=cfg_dict, num_img=cfg_dict["num_images"]).to(
            device
        )
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
        self.debug = cfg_dict["debug"]
        self._training = False
        self._video_recorder = VideoRecorder(num_img=cfg_dict["num_images"])

    def update(self, observation, target_pose, eval: bool = False):
        with torch.no_grad():
            obs, pcd = peract_utils._preprocess_inputs(observation, self.cameras)

            pc, img_feat = rvt_utils.get_pc_img_feat(
                obs,
                pcd,
            )

            pc, img_feat = rvt_utils.move_pc_in_bound(
                pc, img_feat, self.scene_bounds, no_op=not self.move_pc_in_bound
            )

            pc_new = []
            wpt_local = []
            proj_wpt = []
            for _pc, _wpt in zip(pc, target_pose):
                pc_a, _ = mvt_utils.place_pc_in_cube(
                    _pc,
                    with_mean_or_bounds=self._place_with_mean,
                    scene_bounds=None if self._place_with_mean else self.scene_bounds,
                )
                pc_new.append(pc_a)
                position = _wpt[0, :3]
                quaternion = _wpt[0, 3:7]
                pc_b, _ = mvt_utils.place_pc_in_cube(
                    _pc,
                    position,
                    with_mean_or_bounds=self._place_with_mean,
                    scene_bounds=None if self._place_with_mean else self.scene_bounds,
                )
                wpt_local.append(pc_b.unsqueeze(0))

                # Get direction
                direction = self.get_direction(quaternion.cpu().numpy())
                displaced_point = (
                    position + direction * 0.03
                )  # Add a small displacement
                pc_c, _ = mvt_utils.place_pc_in_cube(
                    _pc,
                    displaced_point,
                    with_mean_or_bounds=self._place_with_mean,
                    scene_bounds=None if self._place_with_mean else self.scene_bounds,
                )
                proj_wpt.append(pc_c.unsqueeze(0))
            pc = pc_new
            wpt_local = torch.cat(wpt_local, axis=0)
            proj_wpt = torch.cat(proj_wpt, axis=0)

            img = self.render(
                pc=pc,
                img_feat=img_feat,
                wpt=wpt_local,
                proj_wpt=proj_wpt,
                img_aug=0,
                dyn_cam_info=None,
            )

        actions = observation.get("actions", None)
        if actions is not None:
            actions = actions.squeeze(1)
        is_pad = observation.get("is_pad", None)
        if is_pad is not None:
            is_pad = is_pad.squeeze(1)
        proprio_joint_abs = observation["joint_positions"].squeeze(1)

        use_grad = self._training and not eval
        with torch.set_grad_enabled(use_grad):
            act_out = self.act_model(
                img=img,
                proprio=proprio_joint_abs,
                actions=actions,
                is_pad=is_pad,
            )
        if use_grad:
            # backward
            loss = act_out["loss"]
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()
            self._lr_sched.step()

        return act_out

    def render(self, pc, img_feat, wpt, proj_wpt, img_aug, dyn_cam_info):
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
            b, n, _, h, w = img.shape  # (bs, num_img, img_feat_dim, h, w)

            wpt = wpt.unsqueeze(1)
            pts = self.renderer.get_pt_loc_on_img(
                pt=wpt, fix_cam=True, dyn_cam_info=None
            )  # (bs, 1, num_img, 2)
            proj_wpt = proj_wpt.unsqueeze(1)
            proj_pts = self.renderer.get_pt_loc_on_img(
                pt=proj_wpt, fix_cam=True, dyn_cam_info=None
            )  # (bs, 1, num_img, 2)
            heatmap = torch.zeros(b, n, 1, h, w).to(
                self.device
            )  # (bs, num_img, 1, h, w)

            def clamp(x, y, min_value=0, max_value=219):
                x = max(min(x, max_value), min_value)
                y = max(min(y, max_value), min_value)

                return x, y

            for sb in range(b):
                for sn in range(n):
                    pt = pts[sb, 0, sn, :].int()
                    x, y = pt
                    x, y = clamp(x, y)
                    heatmap[sb, sn, 0, y, x] = 1

                    proj_pt = proj_pts[sb, 0, sn, :].int()
                    dir_2d = proj_pt - pt
                    x1, y1 = x + dir_2d[0], y + dir_2d[1]
                    dir_pts = self.bresenham_line(int(x), int(y), int(x1), int(y1))
                    for dir_pt in dir_pts:
                        x, y = clamp(dir_pt[0], dir_pt[1])
                        heatmap[sb, sn, 0, y, x] = 0.5

            img = torch.cat((img, heatmap), dim=2)
            if self.debug:
                self._video_recorder.record(
                    img=img
                )  # Record video - Image + heatmap while debugging.

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
            self.debug = False

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

    def get_direction(self, quaternion):
        reference_vector = np.array([0, 0, 1])  # From RVEP ref

        # Convert the quaternion to a rotation matrix
        rotation_matrix = R.from_quat(quaternion).as_matrix()

        # Get the actual direction the gripper is pointing in
        gripper_direction = np.dot(rotation_matrix, reference_vector)

        return torch.tensor(gripper_direction).to(self.device).float()

    def bresenham_line(self, x0, y0, x1, y1):
        """Generate points along a line using Bresenham's algorithm."""
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while True:
            points.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

        return points
