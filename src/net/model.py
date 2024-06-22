from ..data_processing.vocabulary import Vocabulary
from .modules.decoder import Decoder
from .modules.encoder import ViTSTR

import torch
import lightning as L
from torch import nn, optim

class Model(L.LightningModule):
    def __init__(
        self,
        # d_model: int,
        vocab: Vocabulary,
        lr_start: float,
        gamma: float,
    ) -> None:
        """Encoder-decoder model with Transformer for image captioning task

        Args:
            `vocab` (`Vocabulary`): vocabulary instance of `src.data_processing.vocabulary.Vocabulary`
            `d_model` (`int`): image feature map size, text embedding size and also hidden size of Transformer
            `num_heads` (`int`): heads of Transformer, `d_model` must be divisible by `num_heads` without remainder
            `lr_start` (`float`): initial learning rate
            `gamma` (`float`): gamma for exponential learning rate scheduler
            `dropout_rate` (`float`, optional): droupout regularization. Defaults to `0.1`.
        """
        super().__init__()
        # self.d_model = d_model
        self.vocab = vocab
        # self.pad_idx = self.vocab.char2idx['<PAD>']
        # self.sos_idx = self.vocab.char2idx['<START>']
        # self.eos_idx = self.vocab.char2idx['<END>']
        # self.unk_idx = self.vocab.char2idx['<UNK>']
        self.vocab_size = len(self.vocab)
        self.lr_start = lr_start
        self.gamma = gamma
        self.save_hyperparameters(dict(
            vocab_size=self.vocab_size,
            lr_start=self.lr_start,
            gamma=self.gamma
        ))
        self.encoder = ViTSTR()
        self.encoder.reset_classifier(num_classes=self.vocab_size)
        # self.encoder.requires_grad_(False)
        # self.encoder.head.requires_grad_(True)

       
    def training_step(self, batch, batch_idx) -> torch.Tensor:
        self.train()
        imgs, captions = batch 
        predicted = self.forward(imgs=imgs, max_length=captions.size(1))
        print(captions.shape, predicted.shape)
        # if batch_idx % 500 == 0:
        #     cap = ''.join([self.vocab.idx2char[idx] for idx in captions[0].tolist()])
        #     pred = ''.join([self.vocab.idx2char[idx] for idx in predicted[0].argmax(-1).tolist()])
        #     print(f'Target: {cap}')
        #     print(f'Predicted: {pred}')
        loss: torch.Tensor = nn.functional.cross_entropy(
            predicted.contiguous().view(-1, self.vocab_size), # [B*seq_output, vocab_size]
            captions.contiguous().view(-1), # [B*seq_output]
            ignore_index=0, # Ignore <PAD> token
        )
        self.log('train_CE', loss, prog_bar=True, logger=self.logger, on_epoch=False, on_step=True)
        return loss
    
    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            imgs, captions = batch 
            predicted = self.forward(imgs=imgs, max_length=captions.size(1))
            print(captions.shape, predicted.shape)
            loss: torch.Tensor = nn.functional.cross_entropy(
                predicted.contiguous().view(-1, self.vocab_size), # [B*seq_output, vocab_size]
                captions.contiguous().view(-1), # [B*seq_output]
                ignore_index=0, # Ignore <PAD> token
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
                ignore_index=0, # Ignore <PAD> token
            )
            self.log('test_CE', loss, prog_bar=True, logger=None)
            return loss
        
    def configure_optimizers(self) -> dict:
        optimizer = optim.Adam(self.parameters(), lr=self.lr_start)
        exp_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.gamma)
        return {
            'optimizer': optimizer,
            'lr_scheduler': exp_scheduler
        }
    
    def forward(self, imgs, max_length) -> torch.Tensor:
        return self.encoder.forward(imgs, max_length) # [B, H*W, vocab_size]
    
    def predict(self, image: torch.Tensor, max_length=50) -> str:
        """Predict caption to image

        Args:
            `image` (`Tensor`): preprocessed (resized and normalized) image of shape `[C, H, W]`
            `max_length` (`int`, optional): max output sentence length. Defaults to `50`.

        Returns:
            `caption` (`str`): predicted caption for image
        """
        device = image.device
        self.eval().to(device)
        image = image.unsqueeze(0)
        y_input = torch.tensor([[self.sos_idx]], dtype=torch.long, device=device)

        for _ in range(max_length):
            # Get source mask
            tgt_mask = self.get_tgt_mask(y_input.size(1)).to(device)
            with torch.no_grad():
                pred: torch.Tensor = self(image, y_input, tgt_mask)
                next_item = pred.argmax(-1)[0, -1].item()
                next_item = torch.tensor([[next_item]], device=device)
                # Concatenate previous input with predicted best word
                y_input = torch.cat((y_input, next_item), dim=1)
                # Stop if model predicts end of sentence
                if next_item.view(-1).item() == self.eos_idx:
                    break
                
        result = y_input.view(-1).tolist()[1:-1]
        return ' '.join([self.vocab.idx2word[idx] for idx in result])
    
    def get_tgt_mask(self, size: int) -> torch.Tensor:
        """Generates a square matrix where the each row allows one word more to be seen

        Args:
            `size` (`int`): sequence length of target, for example if target have shape `[B, S]` then `size = S`

        Returns:
            `mask` (`torch.Tensor`): target mask for transformer
        """
        mask = torch.tril(torch.ones(size, size) == 1).float() # Lower triangular matrix
        mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0
        
        # EX for size=5:
        # [[0., -inf, -inf, -inf, -inf],
        #  [0.,   0., -inf, -inf, -inf],
        #  [0.,   0.,   0., -inf, -inf],
        #  [0.,   0.,   0.,   0., -inf],
        #  [0.,   0.,   0.,   0.,   0.]]
        
        return mask