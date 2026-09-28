from PIL import Image


class ImageResizer:

    def __init__(self, image_size=224):

        self.image_size = image_size

    def resize(self, image):

        return image.resize(
            (self.image_size, self.image_size),
            Image.Resampling.LANCZOS
        )