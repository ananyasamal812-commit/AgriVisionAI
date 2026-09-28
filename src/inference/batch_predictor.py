from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.inference.predictor import Predictor


class BatchPredictor:

    def __init__(
        self,
        predictor=None,
    ):

        if predictor is None:

            predictor = Predictor()

        self.predictor = predictor

        self.supported = {
            ".jpg",
            ".jpeg",
            ".png",
        }

    def predict_folder(
        self,
        input_folder,
        output_csv="outputs/predictions.csv",
    ):

        input_folder = Path(input_folder)

        Path(output_csv).parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        results = []

        images = []

        for ext in self.supported:

            images.extend(
                input_folder.rglob(f"*{ext}")
            )

            images.extend(
                input_folder.rglob(f"*{ext.upper()}")
            )

        print()

        print("=" * 60)
        print("Batch Prediction Started")
        print("=" * 60)

        for image in tqdm(images):

            try:

                prediction = self.predictor.predict(
                    str(image)
                )

                results.append({

                    "image": image.name,

                    "path": str(image),

                    "disease": prediction["disease"],

                    "confidence": prediction["confidence"],

                    "symptoms": prediction["symptoms"],

                    "cause": prediction["cause"],

                    "treatment": prediction["treatment"],

                    "prevention": prediction["prevention"],

                    "gradcam": prediction["gradcam"],

                })

            except Exception as e:

                results.append({

                    "image": image.name,

                    "path": str(image),

                    "error": str(e),

                })

        df = pd.DataFrame(results)

        df.to_csv(
            output_csv,
            index=False,
        )

        print()

        print("=" * 60)
        print("Batch Prediction Finished")
        print("=" * 60)
        print(f"Images Processed : {len(images)}")
        print(f"Results Saved    : {output_csv}")
        print("=" * 60)

        return df