import cv2
import numpy as np
from torch.nn import functional as F


class ImgRecorder(object):
    def __init__(self, num_img, output_filename="output.png"):
        self.output_filename = output_filename
        self.num_img = num_img

    def record(self, img=None, attn=None, num_pat_img=None, num_heads=None):
        bs = img.shape[0]
        for i in range(bs):
            for j in range(self.num_img):
                single_img = img[i, j, 3:6].cpu().numpy()
                # Denormalize img
                single_img = ((single_img + 1.0) * 255.0 / 2.0).astype(np.uint8)
                single_img = np.transpose(single_img, (1, 2, 0))

                single_attn = attn[
                    -1
                ]  # Last attention layer - [bs * num_img * num_heads, num_pat_img * num_pat_img, num_pat_img * num_pat_img]
                batch_size = i * num_heads * self.num_img
                start_index = batch_size + j * num_heads
                single_attn_mean = single_attn[
                    start_index : start_index + num_heads
                ].mean(
                    dim=0
                )  # [num_pat_img * num_pat_img, num_pat_img * num_pat_img]
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
                overlay = self.create_overlay(single_img, upscaled_attention_map)
                img_num = i + j
                self.save_image(
                    frame=single_img, output_filename=f"raw_img{img_num}.png"
                )
                self.save_image(frame=overlay, output_filename=f"output{img_num}.png")

    def create_overlay(self, img, attn):
        attn = cv2.normalize(attn, None, 0, 255, cv2.NORM_MINMAX)
        attention_colored = cv2.applyColorMap(
            attn.astype(np.uint8), cv2.COLORMAP_HOT
        ).astype(np.uint8)

        overlay = cv2.addWeighted(img, 0.6, attention_colored, 0.4, 0)
        return overlay

    def save_image(self, frame, output_filename):
        cv2.imwrite(output_filename, frame)
