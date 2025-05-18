import os

import torch
import numpy as np
import plotly.io as pio
import plotly.express as px
from torch.utils.data import DataLoader
from PIL import Image

from ..net.model import ViTSTRTransducer
from ..data_processing.dataset import Transforms, ImageStatistics
from albumentations import Compose

pio.renderers.default = 'png'


class Predictor:
    def __init__(self, img_size: list[int] | tuple[int, int], device: str, input_channels: int, output_path='predictions', ) -> None:
        """
        Initializes the Predictor class.

        Args:
            `img_size` (`int`): The size of the input image.
            `device_idx` (`str` or `int`): The index of the CUDA device to use for computations.
            `output_path` (`str`, optional): The path to save the output. Defaults to 'predictions'.
        """
        self.output_path = output_path
        self.input_channels = input_channels
        if self.input_channels == 1:
            self.mean = ImageStatistics.MEAN_MONOCHROME
            self.std = ImageStatistics.STD_MONOCHROME
        elif self.input_channels == 3:
            self.mean = ImageStatistics.MEAN_RGB
            self.std = ImageStatistics.STD_RGB
        else:
            raise ValueError(f"input_channels cannot be {self.input_channels}, acceptable values is 1 or 3")
        self.transforms: Compose = Transforms(img_size=img_size, mean=self.mean, std=self.std).eval
        self.device = device
    
    def caption_dataloader(self, dataloader: DataLoader, model: ViTSTRTransducer, n_samples: int = 32) -> None:
        """
        Draws samples from a given dataloader with predicted captions.

        Args:
            dataloader (DataLoader): The dataloader to iterate over.
            model (ViTSTRTransducer): The model to predict captions.
            n_samples (int, optional): The number of samples to draw while testing. Defaults to 32.
        """
        iter_loader = iter(dataloader)
        batched_imgs, _ = next(iter_loader)
        if batched_imgs.size(0) == 1:
            batched_imgs = torch.cat([next(iter_loader)[0] for _ in range(n_samples)], dim=0)
        for i in range(n_samples):
            img = batched_imgs[i]
            processed_img = img.unsqueeze(0) # [B, C, H, W]
            orig_img = (processed_img[0].permute(1, 2, 0) * torch.tensor(self.std) + torch.tensor(self.mean)).cpu().numpy() # [H, W, 3]
            self._save_img_with_caption(
                processed_img=processed_img, 
                orig_img=orig_img, 
                model=model, 
                save_img=True,
            )
            
    def caption_single_image(self, path: str, model: ViTSTRTransducer, save_img: bool) -> str:
        """
        Draws an image from a given path with a predicted caption.

        Args:
            `path` (`str`): The path to the image.
            `model` (`ViTSTRTransducer`): The model used to predict the caption.
            `save_img` (`bool`): Whether to save the image with the predicted caption.

        Returns:
            `str`: The predicted caption.

        Raises:
            `ValueError`: If the image path is incorrect.
        """
        try:
            orig_img = np.array(Image.open(path).convert('RGB')) # [H, W, 3]
        except:
            raise ValueError(f'Wrong image path: {path}')
        processed_img = self.transforms(image=orig_img)['image'].unsqueeze(0) # [3, H, W]
        return self._save_img_with_caption(
            processed_img=processed_img,
            orig_img=orig_img,
            model=model, 
            save_img=save_img
        )
        
    def _save_img_with_caption(
        self, 
        processed_img: torch.Tensor, 
        orig_img: np.ndarray, 
        model: ViTSTRTransducer,
        save_img: bool,
    ) -> str:
        """
        Auxiliary function to draw an image with a predicted caption.

        Args:
            `processed_img` (`torch.Tensor`): The transformed image of shape `[C, H, W]`.
            `orig_img` (`np.ndarray`): The original image of shape `[H, W, C]`.
            `model` (`ViTSTRTransducer`): The model used to predict the caption.
            `save_img` (`bool`): A flag indicating whether to save the image with the predicted caption.

        Returns:
            `str`: The predicted caption.
        """
        prediction = model.predict(processed_img[0])
        prediction = model.vocab.decode(prediction)
        if self.input_channels == 1:
            orig_img = orig_img.squeeze(axis=-1)
        fig = px.imshow(
            orig_img,
            title=f'Prediction: {prediction}',
        ).update_xaxes(
            visible=False
        ).update_yaxes(
            visible=False
        ).update_layout(
            coloraxis_showscale=False,
            title_x=0.5
        )
        if save_img:
            if not os.path.exists(self.output_path):
                os.makedirs(self.output_path)
            backslash = '\\'
            fig.write_image(f"{self.output_path}/{prediction.replace('/', '').replace(backslash, '')}.png")
        return prediction
        