import cv2
import numpy as np
from torch.nn import functional as F


class VideoRecorder(object):
    def __init__(self, num_img, output_filename="output.avi"):
        self.frames = {}
        self.img_frames = {}
        self.num_steps = (
            100  # TODO: Don't finish on hardcode value, trigger end from outside.
        )
        self.output_filename = output_filename
        self.num_img = num_img

    def record(
        self,
        img=None,
        attn=None,
        num_lang_tokens=None,
        num_pat_img=None,
        num_heads=None,
        heatmap=None,
    ):
        bs = img.shape[0]
        for i in range(bs):
            for j in range(self.num_img):
                single_img = img[i, j, 3:6].cpu().numpy()
                # Denormalize img
                single_img = ((single_img + 1.0) * 255.0 / 2.0).astype(np.uint8)
                single_img = np.transpose(single_img, (1, 2, 0))

                single_hm = heatmap[i, j].cpu().numpy()

                single_attn = attn[
                    -1
                ]  # Last attention layer - [bs * num_img * num_heads, num_pat_img * num_pat_img + num_lang_tokens, num_pat_img * num_pat_img + num_lang_tokens]
                batch_size = i * num_heads * self.num_img
                start_index = batch_size + j * num_heads
                single_attn_mean = single_attn[
                    start_index : start_index + num_heads
                ].mean(
                    dim=0
                )  # [num_pat_img * num_pat_img + num_lang_tokens, num_pat_img * num_pat_img + num_lang_tokens]
                single_attn_mean = single_attn_mean[
                    num_lang_tokens:, num_lang_tokens:
                ]  # [num_pat_img * num_pat_img, num_pat_img * num_pat_img]
                sum_attn = single_attn_mean.sum(dim=0)  # [num_pat_img * num_pat_img]
                # Reshape the attention weights to a 2D spatial structure
                single_attn_map = sum_attn.reshape(
                    num_pat_img, num_pat_img
                )  # [num_pat_img, num_pat_img]
                single_attn_map = single_attn_map.unsqueeze(0).unsqueeze(0)
                h, w, _ = single_img.shape
                upscaled_attention_map = (
                    F.interpolate(single_attn_map, size=(h, w), mode="bilinear")
                    .squeeze()
                    .cpu()
                    .numpy()
                )
                overlay = self.create_overlay(
                    single_img, single_hm, upscaled_attention_map
                )

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

    def create_overlay(self, img, heatmap, attn):
        heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
        heatmap_colored = cv2.applyColorMap(
            heatmap.astype(np.uint8), cv2.COLORMAP_JET
        ).astype(np.uint8)

        attn = cv2.normalize(attn, None, 0, 255, cv2.NORM_MINMAX)
        attention_colored = cv2.applyColorMap(
            attn.astype(np.uint8), cv2.COLORMAP_HOT
        ).astype(np.uint8)

        overlay1 = cv2.addWeighted(img, 0.6, heatmap_colored, 0.4, 0)
        overlay_final = cv2.addWeighted(overlay1, 0.6, attention_colored, 0.4, 0)
        return overlay_final

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
