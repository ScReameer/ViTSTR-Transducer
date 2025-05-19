from .vocabulary import Vocabulary
import torch
import json
import os
import numpy as np
import cv2 as cv
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
import lmdb
import io
from pathlib import Path
from torch.utils.data import get_worker_info
from dataclasses import dataclass

@dataclass
class ImageStatistics:
    MEAN_RGB = (0.485, 0.456, 0.406)
    STD_RGB = (0.229, 0.224, 0.225)
    MEAN_MONOCHROME = .0,
    STD_MONOCHROME = 1.,

class Transforms:
    def __init__(self, img_size: list[int] | tuple[int, int], mean: tuple[float, ...], std: tuple[float, ...]) -> None:
        """
        Initializes a Transforms object with the given image size.

        Args:
            img_size (int): The size of the images to be transformed.
        """
        self.img_size = img_size
        self.mean = mean
        self.std = std
        self.eval = A.Compose([
            A.Resize(height=self.img_size[0], width=self.img_size[1]),
            A.Normalize(mean=self.mean, std=self.std),
            ToTensorV2()
        ])
        self.train = A.Compose([
            A.RandomBrightnessContrast(),
            A.Rotate(limit=10, border_mode=cv.BORDER_REPLICATE),
            A.Perspective(),
            A.ChannelShuffle(),
            A.MotionBlur(blur_limit=15),
            A.GaussNoise(),
            self.eval
        ])

class JsonDataset(Dataset):
    def __init__(self, dataset_path: str | Path, vocab: Vocabulary, sample: str, img_size: list[int] | tuple[int, int], input_channels: int) -> None:
        """
        Initializes a PriceDataset object with the given dataset path, vocabulary, sample, and image size.

        Args:
            dataset_path (str): The path to the dataset.
            vocab (Vocabulary): The vocabulary to be used.
            sample (str): The sample to be used (e.g., 'train' or 'test').
            img_size (int): The size of the images in the dataset.
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
        
    def __len__(self):
        return len(self.annotations_list)

    def __getitem__(self, index):
        with open(os.path.join(self.annotations_path, self.annotations_list[index]), 'r') as f:
            annotation_data = json.load(f)
        img_path = annotation_data['name'] + '.png'
        label = annotation_data['description']
        img = np.array(Image.open(os.path.join(self.images_path, img_path)).convert(self.convert_literal))
        img = self.transforms(image=img)['image'] # [C, H, W]
        target = self.vocab.encode(label) # [Sequence]
        return (img, target)
    
class Collate:
    def __init__(self, pad_idx) -> None:
        """
        Initializes a Collate object with the given pad index.

        Args:
            pad_idx (int): The index to use for padding.
        """
        self.pad_idx = pad_idx
    
    def __call__(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        imgs = [item[0][None, ...] for item in batch]
        imgs = torch.cat(imgs, dim=0)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, padding_value=self.pad_idx, batch_first=True)
        return imgs, targets
    
class Database:
    def __init__(self, root: str, max_readers: int = 1):
        """Initializes a read-only LMDB environment."""
        self.root = root
        self.env = lmdb.open(
            root,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=max_readers
        )

    def close(self):
        self.env.close()

class LmdbDataset(Dataset):
    def __init__(self, db: Database, vocab: Vocabulary, sample: str, img_size: list[int] | tuple[int, int], input_channels: int):
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
        self.transforms = transforms.train if 'train' in sample.lower() else transforms.eval
        self.vocab = vocab
        self.db = db
        with db.env.begin(write=False) as txn:
            self.n_samples = int(txn.get(b'num-samples'))
        self.worker_txn = None  # for multi-worker support

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        if get_worker_info() is not None:
            if self.worker_txn is None:
                self.worker_txn = self.db.env.begin(write=False)
            txn = self.worker_txn
        else:
            txn = self.db.env.begin(write=False)

        index += 1
        label_key = f'label-{index:09d}'.encode()
        img_key = f'image-{index:09d}'.encode()
        label = txn.get(label_key)
        imgbuf = txn.get(img_key)

        if label is None or imgbuf is None:
            return self.__getitem__(index=index+1)

        label = label.decode('utf-8')
        img = np.array(Image.open(io.BytesIO(imgbuf)).convert(self.convert_literal))
        img = self.transforms(image=img)['image']
        target = self.vocab.encode(label)
        return img, target
