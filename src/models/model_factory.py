from src.models.efficientnet import EfficientNetModel
from src.models.convnext import ConvNeXtModel
from src.models.vit import ViTModel


class ModelFactory:

    @staticmethod
    def create(
        model_name,
        num_classes,
        pretrained=True,
    ):

        model_name = model_name.lower()

        if model_name.startswith("efficientnet"):
            return EfficientNetModel(
                model_name=model_name,
                num_classes=num_classes,
                pretrained=pretrained,
            )

        elif model_name.startswith("convnext"):
            return ConvNeXtModel(
                model_name=model_name,
                num_classes=num_classes,
                pretrained=pretrained,
            )

        elif model_name.startswith("vit"):
            return ViTModel(
                model_name=model_name,
                num_classes=num_classes,
                pretrained=pretrained,
            )

        raise ValueError(f"Unknown model: {model_name}")