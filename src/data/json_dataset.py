import json
import glob
import os
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image
from .vocabulary import Vocabulary
from .image_statistics import ImageStatistics
from .transforms import Transforms


class JsonDataset(Dataset):
    def __init__(self, dataset_path: str | Path, vocab: Vocabulary, sample: str, img_size: list[int] | tuple[int, int], input_channels: int) -> None:
        """
        Initializes a JsonDataset object with the given dataset path, vocabulary, sample type, image size, and input channels.
        Args:
            dataset_path (str | Path): The path to the dataset.
            vocab (Vocabulary): The vocabulary to use for encoding labels.
            sample (str): The type of sample to use ('train', 'val', or 'test').
            img_size (list[int] | tuple[int, int]): The size of the images.
            input_channels (int): The number of input channels (1 for grayscale, 3 for RGB).
        """
        super().__init__()
        self.dataset_path = dataset_path
        self.sample = sample
        self.input_channels = input_channels
        if self.input_channels == 1:
            mean = ImageStatistics.MEAN_MONOCHROME
            std = ImageStatistics.STD_MONOCHROME
            self.convert_literal = 'L'
        elif self.input_channels == 3:
            mean = ImageStatistics.MEAN_RGB
            std = ImageStatistics.STD_RGB
            self.convert_literal = 'RGB'
        else:
            raise ValueError(f"input_channels cannot be {input_channels}, acceptable values is 1 or 3")
        transforms = Transforms(img_size=img_size, mean=mean, std=std)
        self.transforms = transforms.train if 'train' in self.sample.lower() else transforms.eval
        self.vocab = vocab
        self.root = os.path.join(self.dataset_path, self.sample)
        self.images_path = os.path.join(self.root, 'img')
        self.annotations_path = os.path.join(self.root, 'ann')
        self.annotations_list = os.listdir(self.annotations_path)
        
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.annotations_list)

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return the sample at the given index.
        """
        with open(os.path.join(self.annotations_path, self.annotations_list[index]), 'r') as f:
            annotation_data = json.load(f)
        base_name = annotation_data['name']
        label = annotation_data['description']

        pattern = os.path.join(self.images_path, base_name + '.*')
        matches = glob.glob(pattern)

        if not matches:
            print(f"No image file found for base name '{base_name}' in '{self.images_path}'. Skipping this image...")
            return self.__getitem__(index=index+1)
        
        img_path = matches[0]
        img = np.array(Image.open(os.path.join(self.images_path, img_path)).convert(self.convert_literal))
        img = self.transforms(image=img)['image'] # [C, H, W]
        target = self.vocab.encode(label) # [Sequence]
        return (img, target)