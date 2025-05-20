from .net.model import ViTSTRTransducer
from .data_processing.vocabulary import Vocabulary
from .utils.predictor import Predictor
from .data_processing.dataset import Transforms, ImageStatistics
import torch
import numpy as np
from albumentations import Compose
from PIL import Image


class ImageCaptioner:
    def __init__(self, checkpoint_path: str, device: str) -> None:
        """
        Initializes the ImageCaptioner class.

        Args:
            checkpoint_path (str): The path to the trained checkpoint of the model.
            device_idx (int | str): The index of the CUDA device to use for inference.
        """
        self.model = ViTSTRTransducer.load_from_checkpoint(checkpoint_path, training=False, map_location=device).eval()
        self.model.freeze()
        self.predictor = Predictor(img_size=self.model.input_size, device=device, output_path='inference', input_channels=self.model.input_channels)
        
    def caption_image(self, path: str, save_img=True) -> str:
        """
        Caption a single image.

        Args:
            path (str): The path to the image.
            save_img (bool, optional): Flag to save the image with the predicted caption. Defaults to True.

        Returns:
            str: The predicted caption.
        """
        return self.predictor.caption_single_image(path, self.model, save_img=save_img)
    
class ImageCaptionerTorchscript:
    def __init__(self, checkpoint_path: str, labels: list[str] | str, device: int | str, dtype: torch.dtype) -> None:
        """
        Initializes an instance of the ImageCaptionerTorchscript class.

        Args:
            checkpoint_path (str): The path to the checkpoint file (torchscript).
            vocab_path (str): The path to the vocabulary file.
            device (int | str): The device to use for inference. It can be an integer representing the CUDA device index or a string representing the CPU.
        """
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.model = torch.jit.load(checkpoint_path, map_location=self.device).eval()
        self.vocab: Vocabulary = Vocabulary(labels=labels)
        self.dtype = dtype
        self.img_size = self.model.input_size
        self.input_channels = self.model.input_channels
        if self.input_channels == 1:
            self.mean = ImageStatistics.MEAN_MONOCHROME
            self.std = ImageStatistics.STD_MONOCHROME
            self.convert_literal = 'L'
        elif self.input_channels == 3:
            self.mean = ImageStatistics.MEAN_RGB
            self.std = ImageStatistics.STD_RGB
            self.convert_literal = 'RGB'
        else:
            raise ValueError(f"input_channels cannot be {self.input_channels}, acceptable values is 1 or 3")
        self.transforms: Compose = Transforms(img_size=self.img_size, mean=self.mean, std=self.std).eval
    
    def preprocess_image(self, path: str) -> torch.Tensor:
        """
        Preprocesses an image from a given path.

        Args:
            path (str): The path to the image.

        Returns:
            torch.Tensor: The preprocessed image tensor of shape [1, 3, H, W].
        """
        try:
            orig_img = np.array(Image.open(path).convert(self.convert_literal)) # [H, W, 3]
        except:
            raise ValueError(f'Cannot open image: {path}')
        processed_img = self.transforms(image=orig_img)['image'].to(self.dtype).unsqueeze(0).to(self.device) # [1, 3, H, W]
        return processed_img
        
    
    @torch.inference_mode()
    def predict(self, image: torch.Tensor, max_length=30) -> str:
        """
        Predicts a price for a given image.

        Args:
            image (torch.Tensor): A preprocessed (resized and normalized) image of shape [1, 3, H, W].
            max_length (int, optional): The maximum output sentence length. Defaults to 30. (2 extra tokens for <START> and <END>)

        Returns:
            str: The predicted price for the image.
        """
        y_input = torch.tensor([[self.vocab.start_token_idx]], dtype=torch.int, device=self.device)
        # Example
        # 1st iteration: [<START>]
        # 2nd iteration: [<START>, 7]
        # 3rd iteration: [<START>, 7, 9]
        # 4th iteration: [<START>, 7, 9, <END>]
        # Predicted price: 79
        indices = []
        for _ in range(max_length):
            pred: torch.Tensor = self.model(image, y_input)
            # Next token for y_input
            next_item = pred.argmax(-1)[0, -1].item()
            next_item = torch.tensor([[next_item]], device=self.device)
            # Concatenate previous input with predicted best word
            y_input = torch.cat((y_input, next_item), dim=1)
            # Stop if model predicts end of sentence
            if next_item.view(-1).item() == self.vocab.end_token_idx:
                confidence = pred.softmax(-1).max(-1).values.squeeze().half().cpu().numpy()[:-1]
                break
            indices.append(next_item.item())
        
        # print('Confidences:\n')
        # print(confidence)
        # print('\nIndices:\n')
        # print(indices, '\n')
        # [1, seq] -> [seq] without <START> and <END>
        result = y_input.view(-1)[1:-1]
        return self.vocab.decode(result)