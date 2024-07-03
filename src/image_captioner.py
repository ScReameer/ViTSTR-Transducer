from .data_processing.vocabulary import Vocabulary
from .net.model import ViTSTRTransducer
from .utils.predictor import Predictor

class ImageCaptioner:
    def __init__(self, checkpoint_path: str) -> None:
        """Top-level text recognition class using ViTSTR-Transducer

        Args:
            `checkpoint_path` (`str`): path to trained checkpoint of model
        """
        self.vocab = Vocabulary()
        self.model = ViTSTRTransducer.load_from_checkpoint(checkpoint_path, vocab=self.vocab).eval()
        self.predictor = Predictor()
        
    def caption_image(self, path: str, show_img=True) -> str:
        """Caption single image

        Args:
            `path` (`str`): path to img
            `show_img` (`bool`, optional): flag to show image with predicted caption. Defaults to `True`.
            
        Returns:
            `output` (`str`): predicted caption
        """
        return self.predictor.caption_single_image(path, self.model, show_img=show_img)