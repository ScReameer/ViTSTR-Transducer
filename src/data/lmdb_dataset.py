from torch.utils.data import get_worker_info, Dataset
import numpy as np
from PIL import Image
import io

from .image_statistics import ImageStatistics
from .transforms import Transforms
from .lmdb_database import Database
from .vocabulary import Vocabulary


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