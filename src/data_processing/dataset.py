from .vocabulary import Vocabulary
import torch
import lmdb
import io
import numpy as np
import albumentations as A
import cv2 as cv
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from PIL import Image

TRANSFORMS_EVAL = A.Compose([
    A.Resize(height=224, width=224),
    A.Normalize(mean=[0], std=[1]),
    ToTensorV2()
])
TRANSFORMS_TRAIN = A.Compose([
    A.RandomBrightnessContrast(),
    # A.Rotate(limit=15, border_mode=cv.BORDER_REPLICATE),
    A.GaussNoise(),
    TRANSFORMS_EVAL
])


class Database:
    def __init__(self, root: str, max_readers: int):
        """Initializes the class with the given root directory.

        Args:
            `root` (`str`): The root directory of the LMDB database.
        """
        self.root = root
        self.env = lmdb.open(
            root, 
            readonly=True, 
            max_readers=max_readers, 
            lock=False, 
            readahead=False,
            meminit=False
        )

class LmdbDataset(Dataset):
    def __init__(self, db: Database, vocab: Vocabulary, sample, transforms=None):
        """Creates dataset with images and captions

        Args:
            `db` (`Database`): lmdb database
            `vocab` (`Vocabulary`): `Vocabulary` class from `src.data_processing.vocabulary`
            `transforms` (`A.Compose`, optional): transforms/augmentations for images. Defaults to `None`.
        """
        self.sample = sample
        if transforms:
            self.transforms = transforms
        else:
            self.transforms = TRANSFORMS_TRAIN if sample == 'train' else TRANSFORMS_EVAL
        self.vocab = vocab
        self.db = db
        with self.db.env.begin(write=False) as txn:
            n_samples = int(txn.get('num-samples'.encode()))
            # if sample == 'train':
                # self.n_samples = n_samples
            # else:
            self.n_samples = n_samples // 2

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        if self.sample in ('train', 'valid'):
            index += 1
        else:
            index += self.n_samples
        with self.db.env.begin(write=False) as txn:
            label_key = f'label-{index:09d}'.encode()
            label = txn.get(label_key).decode('utf-8')
            img_key = f'image-{index:09d}'.encode()
            imgbuf = txn.get(img_key)
        buf = io.BytesIO()
        buf.write(imgbuf)
        img = np.array(Image.open(buf).convert('L'))
        img = self.transforms(image=img)['image']
        target = self.vocab.encode_word(label)
        return (img, target)
            # return self.__getitem__(np.random.randint(0, self.n_samples))
    
class Collate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx
    
    def __call__(self, batch):
        imgs = [item[0][None, ...] for item in batch]
        imgs = torch.cat(imgs, dim=0)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, padding_value=self.pad_idx, batch_first=True)
        return imgs, targets