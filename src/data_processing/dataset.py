from .vocabulary import Vocabulary

import torch
import lmdb
import io
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms as T
from PIL import Image


TRANSFORMS = T.Compose([
    T.ToTensor(),
    T.Resize((224, 224)),
    T.Normalize(
        # ImageNet mean and std
        # mean=[0.485, 0.456, 0.406],
        # std=[0.229, 0.224, 0.225]
        mean=[0.5],
        std=[0.5]
    )
])

class Database:
    def __init__(self, root: str):
        """Initializes the class with the given root directory.

        Args:
            `root` (`str`): The root directory of the LMDB database.
        """
        self.root = root
        self.env = lmdb.open(root, readonly=True, max_readers=20, lock=False, readahead=False, meminit=False)

class LmdbDataset(Dataset):
    def __init__(self, db: Database, vocab: Vocabulary, transforms=TRANSFORMS, sample=None):
        """Creates dataset with images and captions

        Args:
            `db` (`Database`): lmdb database
            `vocab` (`Vocabulary`): `Vocabulary` class from `src.data_processing.vocabulary`
            `transforms` (`T.Compose`, optional): transforms/augmentations for images. Defaults to `TRANSFORMS`.
        """
        self.transforms = transforms
        self.vocab = vocab
        self.db = db
        self.sample = sample
        with self.db.env.begin(write=False) as txn:
            n_samples = int(txn.get('num-samples'.encode()))
            if sample is None:
                self.n_samples = n_samples
            else:
                self.n_samples = n_samples // 2

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        if self.sample == 'valid':
            index += 1
        elif self.sample == 'test':
            index += self.n_samples
        try:
            with self.db.env.begin(write=False) as txn:
                label_key = f'label-{index:09d}'.encode()
                label = txn.get(label_key).decode('utf-8')
                img_key = f'image-{index:09d}'.encode()
                imgbuf = txn.get(img_key)
            buf = io.BytesIO()
            buf.write(imgbuf)
            img = self.transforms(np.array(Image.open(buf).convert('L')))
            target = self.vocab.encode_word(label)
            return (img, target)
        except:
            return self.__getitem__(np.random.randint(0, self.n_samples))
    
class Collate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx
    
    def __call__(self, batch):
        imgs = [item[0][None, ...] for item in batch]
        imgs = torch.cat(imgs)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, padding_value=self.pad_idx, batch_first=True)
        return imgs, targets