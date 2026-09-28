import numpy as np


class ImageNormalizer:

    def normalize(self, image):

        image = np.array(image).astype("float32")

        image /= 255.0

        return image