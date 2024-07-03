from ..data_processing.vocabulary import Vocabulary
from .vit_encoder import ViTSTR
import torch
import lightning as L
from torch import nn, optim

class ViTSTRTransducer(L.LightningModule):
    def __init__(
        self,
        input_channels: int,
        d_model: int,
        num_heads: int,
        vocab: Vocabulary,
        lr_max: float,
        lr_min: float,
        t_max: int,
        dropout_rate=0.2
    ) -> None:
        """ViTSTR-Transducer implementation for text recognition

        Args:
            `input_channels` (`int`): number of channels of image
            `d_model` (`int`): image feature map size, text embedding size and also hidden size of Transformer
            `num_heads` (`int`): heads of TransformerDecoder and VisionTransformer, `d_model` must be divisible by `num_heads` without remainder
            `vocab` (`Vocabulary`): vocabulary instance of `src.data_processing.vocabulary.Vocabulary`
            `lr_max` (`float`): maximum learning rate
            `lr_min` (`float`): minimum learning rate
            `t_max` (`int`): cosine annealing T_max
            `dropout_rate` (`float`, optional): droupout regularization. Defaults to `0.2`.
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
        # Save parameters into hparams.yaml
        self.save_hyperparameters(dict(
            vocab_size=self.vocab_size,
            lr_max=self.lr_max,
            lr_min=self.lr_min,
            t_max=self.t_max,
            d_model=self.d_model,
            num_heads=self.num_heads,
            input_channels=self.input_channels
        ))
        # Visual Transformer as encoder
        self.encoder = ViTSTR(
            in_chans=self.input_channels,
            embed_dim=self.d_model, 
            num_classes=self.vocab_size,
            num_heads=self.num_heads,
            drop_rate=self.dropout_rate
        )
        # Transformer as decoder
        self.transformer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=self.num_heads,
            dropout=self.dropout_rate,
            batch_first=True
        )
        self.embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.Sigmoid()
        )
        self.linear = nn.Linear(self.d_model, self.vocab_size)

    def _compute_loss(self, imgs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target_input = target[:, :-1] # [B, seq] without <END>
        target_expected = target[:, 1:] # [B, seq] without <START>
        tgt_mask = self._get_tgt_mask(target_input.size(1))
        # We want to predict target_expected from target_input
        predicted = self.forward(imgs, target_input, tgt_mask) # [B, seq, vocab_size]
        loss = nn.functional.cross_entropy(
            predicted.contiguous().view(-1, self.vocab_size), # [B*seq, vocab_size]
            target_expected.contiguous().view(-1), # [B*seq]
            ignore_index=self.pad_idx, # Ignore <PAD> token
        )
        return loss
       
    def training_step(self, batches, batch_idx) -> torch.Tensor:
        self.train()
        loss = .0
        for batch in batches: # (dl_1, dl_2, ..., dl_n) for multiple dataloaders
            imgs, target = batch
            loss += self._compute_loss(imgs, target)
        loss /= len(batches)
        self.log('train_CE', loss, prog_bar=True, logger=self.logger, on_epoch=False, on_step=True)
        return loss
    
    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            imgs, target = batch
            loss = self._compute_loss(imgs, target)
            self.log('val_CE', loss, prog_bar=True, logger=self.logger, on_epoch=True, on_step=False)
            return loss
        
    def test_step(self, batch, batch_idx) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            imgs, target = batch
            loss = self._compute_loss(imgs, target)
            self.log('test_CE', loss, prog_bar=True, logger=None)
            return loss
        
    def configure_optimizers(self) -> dict:
        optimizer = optim.RAdam(self.parameters(), lr=self.lr_max)
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.t_max, eta_min=self.lr_min)
        return {
            'optimizer': optimizer,
            'lr_scheduler': cosine_scheduler
        }
    
    def forward(self, imgs: torch.Tensor, target: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass of the ViTSTR-Transducer.

        Args:
            `imgs` (`torch.Tensor`): Input images of shape `[B, C, H, W].`
            `target` (`torch.Tensor`): Target captions of shape `[B, seq]`.
            `tgt_mask` (`torch.Tensor`): Mask for target captions of shape `[B, seq]`.

        Returns:
            `output` (`torch.Tensor`): Output tensor of shape `[B, seq, d_model]`

        Description:
            This function performs the forward pass of the model. It takes the input images, target captions, and target mask as input.
            It first extracts the visual features using the encoder module. Then, it embeds the target captions using the embedding module.
            Next, it computes the linguistic features by passing the visual features and embedded target captions through the transformer module.
            The alpha value is computed by applying a feed-forward neural network (ffn) to the element-wise product of the visual features and linguistic features.
            Finally, the output tensor is computed by combining the visual features and linguistic features using the linear module, with the contribution of each factor controlled by the alpha value.
        """
        
        seqlen = target.shape[1]
        visual_features = self.encoder.forward(imgs, seqlen) # [B, seqlen, d_model]
        embedding = self.embedding(target) # [B, seqlen, d_model]
        # Predict target embedding using visual_features from encoder as memory
        linguistic_features = self.transformer(memory=visual_features, tgt=embedding, tgt_mask=tgt_mask)
        alpha = self.ffn(visual_features * linguistic_features)
        output = self.linear((1 - alpha)*visual_features + alpha*linguistic_features)
        return output # [B, seqlen, vocab_size]
    
    def predict(self, image: torch.Tensor, max_length=27) -> str:
        """Predict caption to image

        Args:
            `image` (`Tensor`): preprocessed (resized and normalized) image of shape `[C, H, W]`
            `max_length` (`int`, optional): max output sentence length. Defaults to `27`. (2 extra tokens for `<START>` and `<END>`)

        Returns:
            `caption` (`str`): predicted caption for image
        """
        device = image.device
        self.eval().to(device)
        image = image.unsqueeze(0)
        y_input = torch.tensor([[self.vocab.char2idx['<START>']]], dtype=torch.long, device=device)

        for _ in range(max_length):
            # Get target mask
            tgt_mask = self._get_tgt_mask(y_input.size(1)).to(device)
            with torch.no_grad():
                pred: torch.Tensor = self(image, y_input, tgt_mask)
                next_item = pred.argmax(-1)[0, -1].item()
                next_item = torch.tensor([[next_item]], device=device)
                # Concatenate previous input with predicted best word
                y_input = torch.cat((y_input, next_item), dim=1)
                # Stop if model predicts end of sentence
                if next_item.view(-1).item() == self.vocab.char2idx['<END>']:
                    break
                
        result = y_input.view(-1)[1:-1]
        # return ''.join([self.vocab.idx2char[idx] for idx in result])
        return result
    
    def _get_tgt_mask(self, size: int) -> torch.Tensor:
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