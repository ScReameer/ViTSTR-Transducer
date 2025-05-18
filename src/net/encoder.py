from .vitstr import vitstr_tiny_patch16_224, vitstr_small_patch16_224, vitstr_base_patch16_224
from collections import OrderedDict
import os

import torch
from torch import nn
from timm.models.vision_transformer import VisionTransformer
from timm.models.vision_transformer import vit_tiny_patch16_224


class ViTEncoder(nn.Module):
    def __init__(self, weights_type: str, training: bool, img_size: tuple[int, int], **kwargs) -> None:
        """
        Initializes a ViTSTR model with the specified weights and training mode.

        Args:
            weights_type (str): The type of weights to use, either 'vit' or 'vitstr'.
            training (bool): Whether the model is in training mode.
        """
        super().__init__()
        assert weights_type in ['vit', 'vitstr_tiny', 'vitstr_small', 'vitstr_base'], 'ViT weights should be either "vit" or "vitstr_tiny" or "vitstr_small" or "vitstr_base"'
        if weights_type == 'vit':
            model = weights_type
        else:
            model, size = weights_type.split('_')
        self.weights = weights_type
        self.img_size = img_size
        
        match model:
            
            case 'vitstr':
                # Original ViTSTR hyperparameters
                vitstr_pretrained_config = dict(
                    num_classes=96,
                    out_channels=kwargs['embed_dim'],
                    kernel_size=(16, 16),
                    stride=(16, 16)
                )
                if size == 'tiny':
                    self.vit: VisionTransformer = vitstr_tiny_patch16_224(pretrained=False, num_classes=vitstr_pretrained_config['num_classes'])
                    backbone = 'vitstr_backbone/vitstr_tiny_patch16_224_aug.pth'
                    link = r'https://github.com/roatienza/deep-text-recognition-benchmark/releases/download/v0.1.0/vitstr_tiny_patch16_224_aug.pth'
                elif size == 'small':
                    self.vit: VisionTransformer = vitstr_small_patch16_224(pretrained=False, num_classes=vitstr_pretrained_config['num_classes'])
                    backbone = 'vitstr_backbone/vitstr_small_patch16_224_aug.pth'
                    link = r'https://github.com/roatienza/deep-text-recognition-benchmark/releases/download/v0.1.0/vitstr_small_patch16_224_aug.pth'
                elif size == 'base':
                    self.vit: VisionTransformer = vitstr_base_patch16_224(pretrained=False, num_classes=vitstr_pretrained_config['num_classes'])
                    backbone = 'vitstr_backbone/vitstr_base_patch16_224_aug.pth'
                    link = r'https://github.com/roatienza/deep-text-recognition-benchmark/releases/download/v0.1.0/vitstr_base_patch16_224_aug.pth'
                
                if not os.path.exists(backbone):
                    os.system(f'wget {link} -P {backbone.split(r"/")[0]}')

                if training:
                    loaded_backbone: OrderedDict = torch.load(backbone, weights_only=True)
                    fixed_weights = OrderedDict()
                    # Align keys for compatibility
                    for key in loaded_backbone.keys():
                        fixed_key_name = key.replace('module.vitstr.', '')
                        fixed_weights[fixed_key_name] = loaded_backbone[key]
                    self.vit.load_state_dict(fixed_weights)
                    del loaded_backbone, fixed_weights
                    
                # 1 input channel -> any input channels
                self.vit.patch_embed.proj = nn.Conv2d(
                    kwargs['in_chans'], 
                    vitstr_pretrained_config['out_channels'], 
                    kernel_size=vitstr_pretrained_config['kernel_size'], 
                    stride=vitstr_pretrained_config['stride']
                )
                # Adapt image size. Default is 224x224 and dont need to adapt
                if self.img_size != (224, 224):
                    total_patches = self.img_size[0] // vitstr_pretrained_config['stride'][0] * self.img_size[1] // vitstr_pretrained_config['stride'][1]
                    self.vit.pos_embed.data = self.vit.pos_embed.data[:, :total_patches+1, :]
                    self.vit.patch_embed.img_size = self.img_size
                if training: print('Successfully loaded ViTSTR backbone!\n')
                
            case 'vit':
                self.vit: VisionTransformer = vit_tiny_patch16_224(pretrained=training, **kwargs)
                if training: print('Successfully loaded ViT backbone!\n')

    def forward(self, x, seqlen: int) -> torch.Tensor:
        x = self.vit.forward_features(x)
        x = x[:, :seqlen] # [batch_size, seqlen, d_model]
        return x