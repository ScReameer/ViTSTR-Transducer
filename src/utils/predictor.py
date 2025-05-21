import os

import torch
import numpy as np
import plotly.io as pio
import plotly.express as px
from torch.utils.data import DataLoader
from pathlib import Path

from ..net.model import ViTSTRTransducer
from ..data import ImageStatistics

pio.renderers.default = 'png'


class Predictor:
    def __init__(self, input_channels: int, output_path: str | Path ) -> None:
        """
        Initializes the Predictor class.

        Args:
            `img_size` (`int`): The size of the input image.
            `output_path` (`str`, optional): The path to save the output.
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
    
    def caption_dataloader(self, dataloader: DataLoader, model: ViTSTRTransducer) -> None:
        """
        Draws samples from a given dataloader with predicted captions.

        Args:
            dataloader (DataLoader): The dataloader to iterate over.
            model (ViTSTRTransducer): The model to predict captions.
        """
        iter_loader = iter(dataloader)
        batched_imgs, _ = next(iter_loader)
        n_samples = batched_imgs.size(0) if batched_imgs.size(0) > 1 else 32
        if batched_imgs.size(0) == 1:
            batched_imgs = torch.cat([next(iter_loader)[0] for _ in range(n_samples)], dim=0)
        for i in range(n_samples):
            img = batched_imgs[i]
            processed_img = img.unsqueeze(0) # [B, C, H, W]
            orig_img = (processed_img[0].permute(1, 2, 0) * torch.tensor(self.std) + torch.tensor(self.mean)).cpu().numpy()[np.newaxis, ...] # [H, W, 3]
            self._save_img_with_caption(
                processed_imgs=processed_img, 
                orig_imgs=orig_img, 
                model=model, 
                save_img=True,
            )
        
    def _save_img_with_caption(
        self, 
        processed_imgs: torch.Tensor, 
        orig_imgs: np.ndarray, 
        model: ViTSTRTransducer,
        save_img: bool,
    ) -> tuple[list[str], list[torch.Tensor]]:
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
        prediction, conf = model.predict(processed_imgs)
        
        if save_img:
            for i, pred in enumerate(prediction):
                orig_img = orig_imgs[i]
                if self.input_channels == 1:
                    orig_img = orig_img.squeeze(axis=-1)
                fig = px.imshow(
                    orig_img,
                    title=f'Prediction: {pred}',
                ).update_xaxes(
                    visible=False
                ).update_yaxes(
                    visible=False
                ).update_layout(
                    coloraxis_showscale=False,
                    title_x=0.5
                )
                if not os.path.exists(self.output_path):
                    os.makedirs(self.output_path)
                backslash = '\\'
                fig.write_image(f"{str(self.output_path)}/{pred.replace('/', '').replace(backslash, '')}.png")
        return prediction, conf
        