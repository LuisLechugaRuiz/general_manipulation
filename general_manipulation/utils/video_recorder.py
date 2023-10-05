import cv2
import numpy as np
import torch

from rvt.mvt.utils import generate_hm_from_pt


class VideoRecorder(object):
    def __init__(self, num_img, output_filename="output.avi"):
        self.frames = {}
        self.img_frames = {}
        self.num_steps = (
            100  # TODO: Don't finish on hardcode value, trigger end from outside.
        )
        self.output_filename = output_filename
        self.num_img = num_img

    def record(self, img=None):
        bs = img.shape[0]
        for i in range(bs):
            for j in range(self.num_img):
                single_img = img[i, j, 3:6].cpu().numpy()
                # Denormalize img
                single_img = ((single_img + 1.0) * 255.0 / 2.0).astype(np.uint8)
                single_img = np.transpose(single_img, (1, 2, 0))

                single_keypoint = img[i, j, 7].cpu().numpy()
                overlay = self.create_overlay(single_img, single_keypoint)

                if j not in self.frames:
                    self.frames[j] = []
                    self.img_frames[j] = []
                self.frames[j].append(overlay)
                self.img_frames[j].append(single_img)

        # If the number of steps is reached, create the video
        if all(len(frames) >= self.num_steps for frames in self.frames.values()):
            self.create_videos()
            self.frames = {}  # Clear the frames
            self.img_frames = {}

    def create_overlay(self, img, keypoint):
        h, w, _ = img.shape

        y_indices, x_indices = np.where(np.logical_or(keypoint == 1, keypoint == 0.5))
        pt = torch.stack([torch.tensor(x_indices), torch.tensor(y_indices)], dim=1)
        heatmap = (
            generate_hm_from_pt(
                pt,
                (h, w),
                sigma=1.5,
                thres_sigma_times=3,
            )
            .cpu()
            .numpy()
        )
        heatmap = np.sum(heatmap, axis=0)
        heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
        heatmap_colored = cv2.applyColorMap(
            heatmap.astype(np.uint8), cv2.COLORMAP_JET
        ).astype(np.uint8)
        overlay = cv2.addWeighted(img, 0.4, heatmap_colored, 1.0, 0)
        return overlay

    def create_videos(self):
        for camera_idx, frames in self.frames.items():
            if frames:
                self.create_video(frames, f"video_camera_{camera_idx}.avi")

        for camera_idx, img_frames in self.img_frames.items():
            if img_frames:
                self.create_video(img_frames, f"video_camera_{camera_idx}_img_only.avi")

    def create_video(self, frames, output_filename):
        height, width, _ = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        video_out = cv2.VideoWriter(output_filename, fourcc, 20.0, (width, height))

        for frame in frames:
            video_out.write(frame)

        video_out.release()

    def save_image(self, frame, output_filename):
        cv2.imwrite(output_filename, frame)
