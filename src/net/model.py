from ..data_processing.vocabulary import Vocabulary
from .encoder import ViTSTR
import torch
import lightning as L
from torch import nn, optim

class Model(L.LightningModule):
    def __init__(
        self,
        input_channels: int,
        d_model: int,
        num_heads: int,
        vocab: Vocabulary,
        lr_max: float,
        lr_min: float,
        t_max : int,
    ) -> None:
        """Encoder-decoder model with Transformer for image captioning task

        Args:
            `vocab` (`Vocabulary`): vocabulary instance of `src.data_processing.vocabulary.Vocabulary`
            `d_model` (`int`): image feature map size, text embedding size and also hidden size of Transformer
            `num_heads` (`int`): heads of Transformer, `d_model` must be divisible by `num_heads` without remainder
            `lr_max` (`float`): maximum learning rate
            `lr_min` (`float`): minimum learning rate
            `dropout_rate` (`float`, optional): droupout regularization. Defaults to `0.1`.
        """
        super().__init__()
        self.vocab = vocab
        self.vocab_size = len(self.vocab)
        self.pad_idx = self.vocab.char2idx['<PAD>']
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.t_max = t_max
        self.input_channels = input_channels
        self.d_model = d_model
        self.num_heads = num_heads
        self.save_hyperparameters(dict(
            vocab_size=self.vocab_size,
            lr_max=self.lr_max,
            lr_min=self.lr_min,
            d_model=self.d_model,
            num_heads=self.num_heads,
            input_channels=self.input_channels
        ))
        self.encoder = ViTSTR(
            in_chans=self.input_channels,
            embed_dim=self.d_model, 
            num_classes=self.vocab_size,
            num_heads=self.num_heads
        )

       
    def training_step(self, batch, batch_idx) -> torch.Tensor:
        self.train()
        imgs, captions = batch 
        predicted = self.forward(imgs=imgs, max_length=captions.size(1))
        loss: torch.Tensor = nn.functional.cross_entropy(
            predicted.contiguous().view(-1, self.vocab_size), # [B*seq_output, vocab_size]
            captions.contiguous().view(-1), # [B*seq_output]
            ignore_index=self.pad_idx, # Ignore <PAD> token
        )
        self.log('train_CE', loss, prog_bar=True, logger=self.logger, on_epoch=False, on_step=True)
        return loss
    
    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            imgs, captions = batch 
            predicted = self.forward(imgs=imgs, max_length=captions.size(1))
            loss: torch.Tensor = nn.functional.cross_entropy(
                predicted.contiguous().view(-1, self.vocab_size), # [B*seq_output, vocab_size]
                captions.contiguous().view(-1), # [B*seq_output]
                ignore_index=self.pad_idx, # Ignore <PAD> token
            )
            self.log('val_CE', loss, prog_bar=True, logger=self.logger, on_epoch=True, on_step=False)
            return loss
        
    def test_step(self, batch, batch_idx) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            imgs, captions = batch 
            predicted = self.forward(imgs=imgs, max_length=captions.size(1))
            loss: torch.Tensor = nn.functional.cross_entropy(
                predicted.contiguous().view(-1, self.vocab_size), # [B*seq_output, vocab_size]
                captions.contiguous().view(-1), # [B*seq_output]
                ignore_index=self.pad_idx, # Ignore <PAD> token
            )
            self.log('test_CE', loss, prog_bar=True, logger=None)
            return loss
        
    def configure_optimizers(self) -> dict:
        optimizer = optim.Adam(self.parameters(), lr=self.lr_max)
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.t_max, eta_min=self.lr_min)
        return {
            'optimizer': optimizer,
            'lr_scheduler': cosine_scheduler
        }
    
    def forward(self, imgs, max_length) -> torch.Tensor:
        return self.encoder.forward(imgs, max_length) # [B, H*W, vocab_size]
    
    def predict(self, image: torch.Tensor, max_length=27) -> str:
        """Predict caption to image

        Args:
            `image` (`Tensor`): preprocessed (resized and normalized) image of shape `[C, H, W]`
            `max_length` (`int`, optional): max output sentence length. Defaults to `27`.

        Returns:
            `caption` (`str`): predicted caption for image
        """
        device = image.device
        self.eval().to(device)
        # image = image.unsqueeze(0)
        with torch.no_grad():
            predicted = self.forward(imgs=image, max_length=max_length)[0].argmax(-1)
            return self.vocab.decode_word(predicted)