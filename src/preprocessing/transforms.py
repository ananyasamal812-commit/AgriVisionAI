from PIL import ImageOps


class ImageTransforms:

    def apply(self, image):

        image = ImageOps.exif_transpose(image)

        image = image.convert("RGB")

        return image