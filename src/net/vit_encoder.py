from timm.models.vision_transformer import VisionTransformer

class ViTSTR(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, seqlen: int):
        x = self.forward_features(x)
        x = x[:, :seqlen] # [batch_size, seqlen, d_model]
        return x