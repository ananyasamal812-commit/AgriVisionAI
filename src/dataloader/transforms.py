import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(image_size=224):

    return A.Compose([

        A.RandomResizedCrop(
            size=(image_size, image_size),
            scale=(0.8, 1.0),
        ),

        A.HorizontalFlip(p=0.5),

        A.VerticalFlip(p=0.2),

        A.Rotate(limit=20, p=0.5),

        A.RandomBrightnessContrast(p=0.5),

        A.Normalize(

            mean=(0.485,0.456,0.406),

            std=(0.229,0.224,0.225)

        ),

        ToTensorV2()

    ])


def get_valid_transforms(image_size=224):

    return A.Compose([

        A.Resize(image_size, image_size),

        A.Normalize(

            mean=(0.485,0.456,0.406),

            std=(0.229,0.224,0.225)

        ),

        ToTensorV2()

    ])