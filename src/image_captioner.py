from .data_processing.vocabulary import Vocabulary
from .net.model import Model
from .utils.predictor import Predictor

import os
import joblib
import pandas as pd

class ImageCaptioner:
    def __init__(self, checkpoint_path: str, vocab_path='vocab/vocab.joblib') -> None:
        """Top-level class for image caption task

        Args:
            `checkpoint_path` (`str`): path to trained checkpoint of model
            `captions_path` (`None` or `str`, optional): path to csv file with captions if vocabulary is not provided. Defaults to `None`.
            `vocab_path` (`str`, optional): path to vocabulary serialized file. Defaults to 'vocab/vocab.joblib'.

        Raises:
            `ValueError`: an empty `captions_path` is provided without an existing vocabulary in `vocab_path`
        """
        if os.path.exists(vocab_path):
            self.vocab = joblib.load(vocab_path)
        else:
            raise ValueError(f"{vocab_path} doesn't exist")
        self.model = Model.load_from_checkpoint(checkpoint_path, vocab=self.vocab)
        self.predictor = Predictor()
        
    def caption_image(self, path_or_url: str) -> None:
        """Print predicted caption and draw image

        Args:
            `path_or_url` (`str`): path or URL to img
        """
        self.predictor.caption_single_image(path_or_url, self.model)