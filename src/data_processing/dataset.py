from .vocabulary import Vocabulary

import os
import torch
import lmdb
import io
import pandas as pd
import numpy as np
import plotly.express as px
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms as T
from skimage.io import imread
from PIL import Image
from sklearn.model_selection import train_test_split

TRANSFORMS = T.Compose([
    T.ToTensor(),
    T.Resize((224, 224)),
    T.Normalize(
        # ImageNet mean and std
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Singleton
class Database:
    def __init__(self, root: str):
        self.root = root
        self.env = lmdb.open(root, readonly=True)

class LmdbDataset(Dataset):
    def __init__(self, db: Database, vocab: Vocabulary, transforms=TRANSFORMS):
        self.transforms = transforms
        self.vocab = vocab
        self.db = db
        with self.db.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode()))
            self.nSamples = nSamples

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        index += 1
        try:
            with self.db.env.begin(write=False) as txn:
                label_key = f'label-{index:09d}'.encode()
                label = txn.get(label_key).decode('utf-8')
                img_key = f'image-{index:09d}'.encode()
                imgbuf = txn.get(img_key)
            buf = io.BytesIO()
            buf.write(imgbuf)
            img = self.transforms(np.array(Image.open(buf).convert('RGB')))
            target = self.vocab.numericalize(label)
            return (img, target)
        except:
            return self.__getitem__(np.random.randint(0, self.nSamples))

class FlickrDataset(Dataset):
    def __init__(
        self, 
        images_path: str,
        df: pd.DataFrame,
        vocab: Vocabulary,
        sample: str, 
        random_state=42,
        transforms=TRANSFORMS
    ):
        """Creates dataset with images and captions

        Args:
            `images_path` (`str`): path to images
            `df` (`pd.DataFrame`): dataframe of `result.csv` from Flickr30k dataset which contains image name with corresponding caption
            `vocab` (`Vocabulary`): `Vocabulary` class from `src.data_processing.vocabulary`
            `sample` (`str`): `train`, `valid` or `test` sample
            `random_state` (`int`, optional): random state to split data. Defaults to `42`.
            `transforms` (`T.Compose`, optional): transforms/augmentations for images. Defaults to `TRANSFORMS`.
        """
        self.images_path = images_path
        self.df = df.copy()
        self.vocab = vocab
        # Train: 0.9, Valid: 0.05, Test: 0.05
        train, valid = train_test_split(list(self.df.index), test_size=0.1, random_state=random_state)
        valid, test = train_test_split(list(valid), train_size=0.5, random_state=random_state)
        if sample == 'train':
            self.df = self.df[self.df.index.isin(train)].reset_index()
        elif sample == 'valid':
            self.df = self.df[self.df.index.isin(valid)].reset_index()
        elif sample == 'test':
            self.df = self.df[self.df.index.isin(test)].reset_index()
        self.transforms = transforms
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        caption = self.df.utf8_string[idx]
        target = self.vocab.numericalize(caption)
        image_name = self.df.file_name[idx]
        image_path = os.path.join(self.images_path, image_name)
        x1, y1, x2, y2 = self.df.bbox[idx]
        try:
            image = np.array(Image.open(image_path))[y1:y2, x1:x2, :]
            if image.shape[-1] == 4:
                image = image[..., :3]
            img_tensor = self.transforms(image)
            return img_tensor, target
        except:
            return self.__getitem__(np.random.randint(0, len(self)))
    
class Collate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx
    
    def __call__(self, batch):
        imgs = [item[0][None, ...] for item in batch]
        imgs = torch.cat(imgs)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, padding_value=self.pad_idx, batch_first=True)
        return imgs, targets