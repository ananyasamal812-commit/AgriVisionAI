from pathlib import Path
from PIL import Image
import pandas as pd
from tqdm import tqdm


class MetadataGenerator:

    def __init__(self, dataset_root):

        self.dataset_root = Path(dataset_root)

    def generate(self):

        rows = []

        image_extensions = {
            ".jpg",
            ".jpeg",
            ".png"
        }

        for split in ["train", "valid", "test"]:

            split_path = self.dataset_root / split

            if not split_path.exists():
                continue

            classes = [x for x in split_path.iterdir() if x.is_dir()]

            for class_folder in tqdm(classes, desc=f"{split}"):

                class_name = class_folder.name

                if "___" in class_name:

                    crop, disease = class_name.split("___", 1)

                else:

                    crop = "Unknown"

                    disease = class_name

                healthy = disease.lower() == "healthy"

                for image in class_folder.iterdir():

                    if image.suffix.lower() not in image_extensions:
                        continue

                    try:

                        with Image.open(image) as img:

                            width, height = img.size

                        rows.append({

                            "split": split,

                            "crop": crop,

                            "disease": disease,

                            "healthy": healthy,

                            "image_path": str(image),

                            "file_name": image.name,

                            "width": width,

                            "height": height,

                            "extension": image.suffix,

                            "size_kb": round(image.stat().st_size / 1024, 2)

                        })

                    except Exception:

                        continue

        df = pd.DataFrame(rows)

        output = Path("datasets/metadata/dataset_metadata.csv")

        output.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(output, index=False)

        print()

        print("=" * 60)

        print("Metadata Generated Successfully")

        print("=" * 60)

        print(f"Total Images : {len(df)}")

        print(f"Output File  : {output}")

        print("=" * 60)

        return df