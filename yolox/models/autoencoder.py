
# yolox/models/autoencoder.py

import torch.nn as nn

# Use relative imports to integrate with the YOLOX project structure
from .darknet import CSPDarknet
from .network_blocks import BaseConv

class Decoder(nn.Module):
    """
    Decoder module to reconstruct an image from features extracted by CSPDarknet.
    Symmetrical to the encoder's architecture.
    """
    def __init__(self, wid_mul=1.0, act="silu"):
        super().__init__()
        base_channels = int(wid_mul * 64)

        # Upsample from dark5 (1024 channels for width=1.0) to dark4 resolution
        self.up_block1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            BaseConv(base_channels * 16, base_channels * 8, ksize=3, stride=1, act=act)
        )
        # Upsample to dark3 resolution
        self.up_block2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            BaseConv(base_channels * 8, base_channels * 4, ksize=3, stride=1, act=act)
        )
        # Upsample to dark2 resolution
        self.up_block3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            BaseConv(base_channels * 4, base_channels * 2, ksize=3, stride=1, act=act)
        )
        # Upsample to stem resolution
        self.up_block4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            BaseConv(base_channels * 2, base_channels, ksize=3, stride=1, act=act)
        )
        # Final upsample to reverse the initial 'Focus' layer
        self.up_block5 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            BaseConv(base_channels, base_channels // 2, ksize=3, stride=1, act=act)
        )
        # Final convolution to get 3-channel image
        self.final_conv = nn.Conv2d(base_channels // 2, 3, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.up_block1(x)
        x = self.up_block2(x)
        x = self.up_block3(x)
        x = self.up_block4(x)
        x = self.up_block5(x)
        x = self.final_conv(x)
        # Scale output to [0, 255] to match the input from preproc
        return self.sigmoid(x) * 255.0


class Autoencoder(nn.Module):
    """
    Wraps the CSPDarknet encoder and the custom Decoder for pre-training.
    """
    def __init__(self, dep_mul=1.0, wid_mul=1.0, act="silu"):
        super().__init__()
        self.encoder = CSPDarknet(
            dep_mul=dep_mul, 
            wid_mul=wid_mul, 
            out_features=("dark5",),  # Only need the final feature map from the backbone
            act=act
        )
        self.decoder = Decoder(wid_mul=wid_mul, act=act)

    def forward(self, x):
        features = self.encoder(x)
        latent_rep = features["dark5"]
        reconstructed_x = self.decoder(latent_rep)
        return reconstructed_x