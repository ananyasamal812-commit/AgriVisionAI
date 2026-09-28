import timm
import torch.nn as nn


class EfficientNetModel(nn.Module):

    def __init__(
        self,
        model_name="efficientnet_b3",
        num_classes=38,
        pretrained=True,
    ):

        super().__init__()

        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
        )

    def forward(self, x):
        return self.model(x)