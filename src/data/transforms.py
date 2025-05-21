import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2 as cv


class Transforms:
    def __init__(self, img_size: list[int] | tuple[int, int], mean: tuple[float, ...], std: tuple[float, ...]) -> None:
        """
        Initializes a Transforms object with the given image size.
        Args:
            img_size (list[int] | tuple[int, int]): The size of the images to be transformed.
            mean (tuple[float, ...]): The mean values for normalization.
            std (tuple[float, ...]): The standard deviation values for normalization.
        """
        self.img_size = img_size
        self.mean = mean
        self.std = std
        self.eval = A.Compose([
            A.Resize(height=self.img_size[0], width=self.img_size[1]),
            A.Normalize(mean=self.mean, std=self.std),
            ToTensorV2()
        ])
        # Augmentations for training data
        self.train = A.Compose([
            A.RandomBrightnessContrast(),
            A.Rotate(limit=10, border_mode=cv.BORDER_REPLICATE),
            A.Perspective(),
            A.ChannelShuffle(),
            A.MotionBlur(blur_limit=15),
            A.GaussNoise(),
            self.eval
        ])