from pathlib import Path

import cv2

from torch.utils.data import Dataset


class CropDiseaseDataset(Dataset):

    def __init__(

        self,

        root,

        transform=None

    ):

        self.root = Path(root)

        self.transform = transform

        self.images = []

        self.labels = []
        
        self.classes = sorted(

            [x.name for x in self.root.iterdir() if x.is_dir()]

        )

        self.class_to_idx = {

            cls:i

            for i,cls in enumerate(self.classes)

        }

        for cls in self.classes:

            folder = self.root/cls

            for image in folder.iterdir():

                self.images.append(image)

                self.labels.append(

                    self.class_to_idx[cls]

                )

    def __len__(self):

        return len(self.images)

    def __getitem__(self,index):

        image = cv2.imread(

            str(self.images[index])

        )

        image = cv2.cvtColor(

            image,

            cv2.COLOR_BGR2RGB

        )

        label = self.labels[index]

        if self.transform:

            image = self.transform(

                image=image

            )["image"]

        return image,label