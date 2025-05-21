from dataclasses import dataclass

@dataclass
class ImageStatistics:
    """
    Statistics for image normalization.
    """
    MEAN_RGB = (0.485, 0.456, 0.406)
    STD_RGB = (0.229, 0.224, 0.225)
    MEAN_MONOCHROME = .0,
    STD_MONOCHROME = 1.,