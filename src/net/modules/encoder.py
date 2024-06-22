import torch
from torch import nn
from torchvision import models

class Encoder(nn.Module):
    def __init__(self, d_model: int):
        """Encoder class to extract feature maps from images

        Args:
            `d_model` (`int`): size of text embedding
        """
        super().__init__()
        resnet = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
        modules = list(resnet.children())[:-2]
        # [B, 2048, H, W]
        resnet_output_channels = 2048
        self.resnet = nn.Sequential(*modules).eval()
        self.resnet.requires_grad_(False)
        self.linear = nn.Linear(
            in_features=resnet_output_channels,
            out_features=d_model
        )

        
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features: torch.Tensor = self.resnet(images)
        # [B, feature_maps, H, W] -> [B, feature_maps, H*W] -> [B, H*W, feature_maps]
        features = features.flatten(start_dim=-2, end_dim=-1).movedim(-1, 1)
        # [B, H*W, feature_maps] -> [B, H*W, d_model]
        features = self.linear(features)
        return features
    
from timm.models.vision_transformer import VisionTransformer


class ViTSTR(VisionTransformer):
    '''
    ViTSTR is basically a ViT that uses DeiT weights.
    Modified head to support a sequence of characters prediction for STR.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x

    def forward(self, x, seqlen: int =25):
        x = self.forward_features(x)
        x = x[:, :seqlen]

        # batch, seqlen, embsize
        b, s, e = x.size()
        x = x.reshape(b*s, e)
        x = self.head(x).view(b, s, self.num_classes)
        return x
    
class ViTEncoder(nn.Module):
    def __init__(self, vocab_size: int):
        """Encoder class to extract feature maps from images

        Args:
            `d_model` (`int`): size of text embedding
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.vit = models.vision_transformer.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        # modules = list(vit.children())
        # [B, 768, H, W]
        vit_output_channels = 768
        # self.vit = nn.Sequential(*modules)
        # self.vit.requires_grad_(False)
        # self.linear = nn.Linear(
        #     in_features=vit_output_channels,
        #     out_features=self.vocab_size
        # )

        
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features: torch.Tensor = self.vit(images)
        # [B, feature_maps, H, W] -> [B, feature_maps, H*W] -> [B, H*W, feature_maps]
        # features = features.flatten(start_dim=-2, end_dim=-1).movedim(-1, 1)
        # # [B, H*W, feature_maps] -> [B, H*W, d_model]
        # features = self.linear(features)
        return features