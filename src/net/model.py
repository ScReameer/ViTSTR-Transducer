import torch
from pytorch_lightning import LightningModule
from torchmetrics.functional.classification import multiclass_accuracy, multiclass_f1_score
from torch import nn, optim
from torch.nn.attention import sdpa_kernel

from .losses import CrossEntropyLossSequence, FocalLoss
from ..data_processing.vocabulary import Vocabulary
from .encoder import ViTEncoder


class ViTSTRTransducer(LightningModule):
    def __init__(
        self,
        input_size: tuple[int, int],
        input_channels: int,
        d_model: int,
        num_heads: int,
        vocab: Vocabulary,
        lr: float,
        weight_decay: float,
        gamma: float,
        weights_type: str,
        dropout_rate: float,
        loss: str,
        training: bool,
        sdp_backend
    ) -> None:
        """
        Initializes the ViTSTRTransducer model.

        Args:
            input_size (tuple[int, int]): The size of the input image.
            input_channels (int): The number of channels in the input image.
            d_model (int): The size of the input feature map, text embedding, and hidden size of the Transformer.
            num_heads (int): The number of heads in the TransformerDecoderLayer.
            vocab (Vocabulary): The vocabulary instance.
            lr (float): The learning rate.
            gamma (float): The gamma value for the ExponentialLR scheduler.
            dropout_rate (float, optional): The dropout rate.
            pretrained (bool, optional): Whether to use a pre-trained model.
            loss (str, optional): The loss function to use.
        """
        super().__init__()
        self.vocab = vocab
        self.vocab_size = len(self.vocab)
        self.pad_idx = self.vocab.token2idx['<PAD>']
        self.lr = lr
        self.loss = loss
        self.weight_decay = weight_decay
        self.gamma = gamma
        self.input_size = input_size
        self.input_channels = input_channels
        self.d_model = d_model
        self.num_heads = num_heads
        self.weights_type = weights_type
        self.dropout_rate = dropout_rate
        self.training = training
        self.sdp_backend = sdp_backend
        # Save params
        self.save_hyperparameters(dict(
            vocab_size=self.vocab_size,
            lr=self.lr,
            weight_decay=self.weight_decay,
            loss=self.loss,
            gamma=self.gamma,
            d_model=self.d_model,
            num_heads=self.num_heads,
            input_channels=self.input_channels,
            vocab=self.vocab,
            input_size=self.input_size,
            weights_type=self.weights_type,
            dropout_rate=self.dropout_rate
        ))
        # VisualTransformer as encoder
        self.encoder = ViTEncoder(
            weights_type=self.weights_type,
            training=self.training,
            img_size=self.input_size,
            in_chans=self.input_channels,
            embed_dim=self.d_model,
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
        if loss == 'focal_loss':
            self.loss = FocalLoss(gamma=2, ignore_index=self.pad_idx)
        elif loss == 'cross_entropy':
            self.loss = CrossEntropyLossSequence(ignore_index=self.pad_idx)
        else:
            raise ValueError(f"Unsupported loss: {loss}")
       
    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        self.train()
        imgs, target = batch
        loss, f1, acc = self._compute_loss_and_metrics(imgs, target)
        self.log('train_loss', loss, prog_bar=True, logger=self.logger, on_epoch=False, on_step=True)
        self.log('train_f1', f1, prog_bar=False, logger=self.logger, on_epoch=False, on_step=True)
        self.log('train_acc', acc, prog_bar=False, logger=self.logger, on_epoch=False, on_step=True)
        return loss
    
    @torch.no_grad()
    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        self.eval()
        imgs, target = batch
        loss, f1, acc = self._compute_loss_and_metrics(imgs, target)
        self.log('val_loss', loss, prog_bar=True, logger=self.logger, on_epoch=True, on_step=False)
        self.log('val_f1', f1, prog_bar=True, logger=self.logger, on_epoch=True, on_step=False)
        self.log('val_acc', acc, prog_bar=True, logger=self.logger, on_epoch=True, on_step=False)
        return loss
        
    @torch.no_grad()
    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        self.eval()
        imgs, target = batch
        loss, f1, acc = self._compute_loss_and_metrics(imgs, target)
        self.log('test_loss', loss, prog_bar=True, logger=None)
        self.log('test_f1', f1, prog_bar=True, logger=None)
        self.log('test_acc', acc, prog_bar=True, logger=None)
        return loss
        
    def configure_optimizers(self):
        # RAdamW
        optimizer = optim.RAdam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay, decoupled_weight_decay=True)
        exponential_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.gamma)
        return {
            'optimizer': optimizer,
            'lr_scheduler': exponential_scheduler
        }
    
    def forward(self, imgs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass of the ViTSTR-Transducer.

        Args:
            `imgs` (`torch.Tensor`): Input images of shape `[B, C, H, W].`
            `target` (`torch.Tensor`): Target captions of shape `[B, seq]`.

        Returns:
            `output` (`torch.Tensor`): Output tensor of shape `[B, seq, d_model]`

        Description:
            This function performs the forward pass of the model. It takes the input images, target captions, and target mask as input.
            It first extracts the visual features using the encoder module. Then, it embeds the target captions using the embedding module.
            Next, it computes the linguistic features by passing the visual features and embedded target captions through the transformer module.
            The alpha value is computed by applying a feed-forward neural network (ffn) to the element-wise product of the visual features and linguistic features.
            Finally, the output tensor is computed by combining the visual features and linguistic features using the linear module, 
            with the contribution of each factor controlled by the alpha value.
        """
        seqlen = target.shape[1]
        visual_features = self.encoder.forward(imgs, seqlen) # [B, seqlen, d_model]
        embedding = self.embedding(target) # [B, seqlen, d_model]
        attn_mask = torch.triu(torch.full((seqlen, seqlen), float('-inf'), device=target.device), diagonal=1)
        linguistic_features = self.transformer.forward(memory=visual_features, tgt=embedding, tgt_is_causal=True, tgt_mask=attn_mask)
        alpha = self.ffn(visual_features * linguistic_features)
        output = self.linear((1 - alpha)*visual_features + alpha*linguistic_features)
        return output # [B, seqlen, vocab_size]
    
    @torch.no_grad()
    def predict(self, image: torch.Tensor, max_length=30):
        """
        Predicts a price for a given image.

        Args:
            image (torch.Tensor): A preprocessed (resized and normalized) image of shape [C, H, W].
            max_length (int, optional): The maximum output sentence length. Defaults to 30. (2 extra tokens for <START> and <END>)

        Returns:
            str: The predicted price for the image.
        """
        device = image.device
        self.eval().to(device)
        image = image.unsqueeze(0)
        y_input = torch.tensor([[self.vocab.token2idx['<START>']]], dtype=torch.int, device=device)

        for _ in range(max_length):
            # Get target mask
            # tgt_mask = self._get_tgt_mask(y_input.size(1)).to(device)
            pred: torch.Tensor = self.forward(image, y_input)
            next_item = pred.argmax(-1)[0, -1].item()
            next_item = torch.tensor([[next_item]], device=device)
            # Concatenate previous input with predicted best word
            y_input = torch.cat((y_input, next_item), dim=1)
            # Stop if model predicts end of sentence
            if next_item.view(-1).item() == self.vocab.token2idx['<END>']:
                break
                
        result = y_input.view(-1)[1:-1]
        return result
    
    def _compute_loss_and_metrics(self, imgs: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        target_input = target[:, :-1] # [B, seq] without <END>
        target_expected = target[:, 1:] # [B, seq] without <START>
        # We want to predict target_expected from target_input
        with sdpa_kernel(self.sdp_backend):
            predicted = self.forward(imgs, target_input) # [B, seq, vocab_size]
        loss = self.loss(predicted, target_expected)
        predicted = predicted.permute(0, 2, 1) # [B, vocab_size, seq]
        # Compute metrics across sequence dimension
        f1 = multiclass_f1_score(
            predicted, 
            target_expected, 
            num_classes=self.vocab_size,
            ignore_index=self.pad_idx,
            multidim_average='samplewise'
        ).mean()
        acc = multiclass_accuracy(
            predicted,
            target_expected, 
            num_classes=self.vocab_size,
            ignore_index=self.pad_idx,
            multidim_average='samplewise'
        ).mean()
        return loss, f1, acc