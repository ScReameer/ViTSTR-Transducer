from ..net.model import Model

import torch
import albumentations as A
import numpy as np
import plotly.io as pio
import plotly.express as px

from torch.utils.data import DataLoader
from PIL import Image

pio.renderers.default = 'png'
pio.templates.default = 'plotly_dark'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# ImageNet mean and std
MEAN = [0]
STD = [1]
# ResNet input image size
RESIZE_TO = (224, 224)

TRANSFORMS_EVAL = A.Compose([
    A.ToFloat(max_value=255),
    A.Resize(height=224, width=224),
    A.Normalize(mean=[0], std=[1])
])

class Predictor:
    def __init__(self) -> None:
        """Aux class to draw images with predicted caption"""
        self.inv_normalizer = A.Normalize(
            mean=[-m/s for m, s in zip(MEAN, STD)],
            std=[1/s for s in STD]
        )
        self.transforms = TRANSFORMS_EVAL
    
    def caption_dataloader(self, dataloader: DataLoader, model: Model, n_samples=5) -> None:
        """Draw n samples from given dataloader with predicted caption

        Args:
            `dataloader` (`DataLoader`): dataloader with pairs (image, target_caption)
            `model` (`Model`): model to predict caption
            `n_samples` (`int`, optional): number of samples to draw with predicted caption. Defaults to `5`.
        """
        iter_loader = iter(dataloader)
        for _ in range(n_samples):
            img, target = next(iter_loader)
            processed_img = img[0] # [C, H, W]
            target = target[0]
            orig_img = self.inv_normalizer(image=processed_img)['image'].cpu().permute(1, 2, 0).numpy()[..., 0] # [H, W]
            self._show_img_with_caption(processed_img, orig_img, model, target)
            
    def caption_single_image(self, path: str, model: Model) -> None:
        """Draw image from path or URL with predicted caption

        Args:
            `path` (`str`): path to image. Image must have 3 (RGB) or 4 (RGBA) channels, otherwise raises `ValueError`
            `model` (`Model`): model to predict caption

        Raises:
            `ValueError`: image channels < 3
        """
        try:
            orig_img = np.array(Image.open(path).convert('L')) # [H, W]
        except:
            raise ValueError(f'Wrong image path: {path}')
        processed_img = self.transforms(image=orig_img)['image'][None, ...] # [1, H, W]
        self._show_img_with_caption(processed_img, orig_img, model)
        
    def _show_img_with_caption(self, processed_img: torch.Tensor, orig_img: np.ndarray, model: Model, target=None) -> None:
        """Aux func to draw image and print caption

        Args:
            `processed_img` (`torch.Tensor`): transformed image of shape `[C, H, W]`
            `orig_img` (`np.ndarray`): original image of shape `[H, W, C]`
            `model` (`Model`): model to predict caption
        """
        prediction = model.predict(torch.tensor(processed_img, device=DEVICE))
        print(f'Predicted caption: {prediction}')
        if target is not None:
            target = model.vocab.decode_word(target)
            print(f'Target caption: {target}')
        px.imshow(orig_img, color_continuous_scale='gray').update_xaxes(visible=False).update_yaxes(visible=False).show()
        