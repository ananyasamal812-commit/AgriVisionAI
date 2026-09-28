import cv2
import numpy as np
import torch

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


class GradCAMGenerator:

    def __init__(self, model):
        self.model = model

        # Find the correct target layer
        if hasattr(model, "model"):
            backbone = model.model

            if hasattr(backbone, "blocks"):
                # timm EfficientNet
                self.target_layers = [backbone.blocks[-1]]

            elif hasattr(backbone, "features"):
                # torchvision EfficientNet
                self.target_layers = [backbone.features[-1]]

            else:
                raise AttributeError(
                    "Could not locate a valid target layer for GradCAM."
                )

        elif hasattr(model, "features"):
            self.target_layers = [model.features[-1]]

        else:
            raise AttributeError(
                "Model does not contain 'features' or 'model.blocks'."
            )

    def generate(
        self,
        input_tensor,
        rgb_image,
        output_path="outputs/gradcam.png",
    ):

        with GradCAM(
            model=self.model,
            target_layers=self.target_layers,
        ) as cam:

            grayscale_cam = cam(
                input_tensor=input_tensor,
                targets=None,
            )[0]

        grayscale_cam = cv2.resize(
            grayscale_cam, (rgb_image.shape[1], rgb_image.shape[0])
        )
        visualization = show_cam_on_image(
            rgb_image,
            grayscale_cam,
            use_rgb=True,
        )

        visualization = cv2.cvtColor(
            visualization,
            cv2.COLOR_RGB2BGR,
        )

        saved = cv2.imwrite(
            output_path,
            visualization,
        )

        if not saved:
            raise OSError(f"Unable to save heatmap: {output_path}")
        return output_path
