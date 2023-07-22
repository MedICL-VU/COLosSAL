from monai.networks.nets import UNETR


class unetr(nn.Module):
    def __init__(self, crop_size, num_classes=1):
        super(DI2IN, self).__init__()
        self.model = UNETR(
            in_channels=1, 
            out_channels=num_classes, 
            img_size=crop_size, 
            feature_size=16,
            hidden_size=768,
            mlp_dim=3072, 
            num_heads=12, 
            pos_embed='conv', 
            norm_name='instance', 
            conv_block=True, 
            res_block=True, 
            dropout_rate=0.0, 
            spatial_dims=3)

    def forward(self, x):
        return self.model(x)
