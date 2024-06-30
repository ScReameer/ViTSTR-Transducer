from .data_processing.vocabulary import Vocabulary
from .net.model import Model
from .utils.predictor import Predictor

class ImageCaptioner:
    def __init__(self, checkpoint_path: str) -> None:
        """Top-level class for image caption task

        Args:
            `checkpoint_path` (`str`): path to trained checkpoint of model
        """
        self.vocab = Vocabulary()
        self.model = Model.load_from_checkpoint(checkpoint_path, vocab=self.vocab)
        self.predictor = Predictor()
        
    def caption_image(self, path_or_url: str) -> None:
        """Print predicted caption and draw image

        Args:
            `path_or_url` (`str`): path or URL to img
        """
        self.predictor.caption_single_image(path_or_url, self.model)