from ..data_processing.vocabulary import Vocabulary
from .encoder import ViTSTR
import torch
import lightning as L
from torch import nn, optim
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np

# ImageNet mean and std
MEAN = [0]
STD = [1]
# ResNet input image size
RESIZE_TO = (224, 224)

TRANSFORMS_EVAL = A.Compose([
    # A.ToFloat(max_value=255),
    A.Resize(height=224, width=224),
    A.Normalize(mean=[0], std=[1]),
    ToTensorV2()
    
])

class Model(L.LightningModule):
    def __init__(
        self,
        input_channels: int,
        d_model: int,
        num_heads: int,
        vocab: Vocabulary,
        lr_max: float,
        lr_min: float,
        t_max: int,
        dropout_rate=0.1
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
        self.dropout_rate = dropout_rate
        self.save_hyperparameters(dict(
            vocab_size=self.vocab_size,
            lr_max=self.lr_max,
            lr_min=self.lr_min,
            t_max=self.t_max,
            d_model=self.d_model,
            num_heads=self.num_heads,
            input_channels=self.input_channels
        ))
        self.encoder = ViTSTR(
            in_chans=self.input_channels,
            embed_dim=self.d_model, 
            num_classes=self.vocab_size,
            num_heads=self.num_heads,
            drop_rate=self.dropout_rate
        )
        self.embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.transformer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=self.num_heads,
            dropout=self.dropout_rate,
            batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.Sigmoid()
        )
        self.linear = nn.Linear(self.d_model, self.vocab_size)

       
    def training_step(self, batch, batch_idx) -> torch.Tensor:
        self.train()
        imgs, target = batch
        target_input = target[:, :-1]
        target_expected = target[:, 1:]
        tgt_mask = self.get_tgt_mask(target_input.size(1))
        predicted = self.forward(imgs, target_input, tgt_mask)
        loss: torch.Tensor = nn.functional.cross_entropy(
            predicted.contiguous().view(-1, self.vocab_size), # [B*seq_output, vocab_size]
            target_expected.contiguous().view(-1), # [B*seq_output]
            ignore_index=self.pad_idx, # Ignore <PAD> token
        )
        if batch_idx % 100 == 0:
            orig_img = np.array(Image.open('/mnt/s/CV/text_recognition_examples/pepsi.jpg').convert('L'))
            processed_img = TRANSFORMS_EVAL(image=orig_img)['image'].cuda()
            predicted_string = self.predict(processed_img)
            # predicted_string = self.vocab.decode_word(predicted[64].argmax(dim=-1))
            print(f'Target: pepsi')
            print(f'Predicted: {predicted_string}\n')
        self.log('train_CE', loss, prog_bar=True, logger=self.logger, on_epoch=False, on_step=True)
        return loss
    
    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            imgs, target = batch
            target_input = target[:, :-1]
            target_expected = target[:, 1:]
            tgt_mask = self.get_tgt_mask(target_input.size(1))
            predicted = self.forward(imgs, target_input, tgt_mask)
            loss: torch.Tensor = nn.functional.cross_entropy(
                predicted.contiguous().view(-1, self.vocab_size), # [B*seq_output, vocab_size]
                target_expected.contiguous().view(-1), # [B*seq_output]
                ignore_index=self.pad_idx, # Ignore <PAD> token
            )
            self.log('val_CE', loss, prog_bar=True, logger=self.logger, on_epoch=True, on_step=False)
            return loss
        
    def test_step(self, batch, batch_idx) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            imgs, target = batch
            target_input = target[:, :-1]
            target_expected = target[:, 1:]
            tgt_mask = self.get_tgt_mask(target_input.size(1))
            predicted = self.forward(imgs, target_input, tgt_mask)
            loss: torch.Tensor = nn.functional.cross_entropy(
                predicted.contiguous().view(-1, self.vocab_size), # [B*seq_output, vocab_size]
                target_expected.contiguous().view(-1), # [B*seq_output]
                ignore_index=self.pad_idx, # Ignore <PAD> token
            )
            self.log('test_CE', loss, prog_bar=True, logger=None)
            return loss
        
    def configure_optimizers(self) -> dict:
        optimizer = optim.RAdam(self.parameters(), lr=self.lr_max)
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.t_max, eta_min=self.lr_min)
        return {
            'optimizer': optimizer,
            'lr_scheduler': cosine_scheduler
        }
    
    def forward(self, imgs, target, tgt_mask) -> torch.Tensor:
        seqlen = target.shape[1]
        visual_features = self.encoder.forward(imgs, seqlen) # [B, seq, d_model]
        embedding = self.embedding(target) # [B, seq, d_model]
        linguistic_features = self.transformer(memory=visual_features, tgt=embedding, tgt_mask=tgt_mask) # [B, seq, d_model]
        alpha = self.ffn(visual_features * linguistic_features)
        output = self.linear((1 - alpha)*visual_features + alpha*linguistic_features)
        return output
    
    def predict(self, image: torch.Tensor, max_length=27) -> str:
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
        y_input = torch.tensor([[self.vocab.char2idx['<START>']]], dtype=torch.long, device=device)

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
                if next_item.view(-1).item() == self.vocab.char2idx['<END>']:
                    break
                
        result = y_input.view(-1).tolist()[1:-1]
        return ''.join([self.vocab.idx2char[idx] for idx in result])
    
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
        
        return mask.to(self.device)