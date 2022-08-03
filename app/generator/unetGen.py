import torch
import torch.nn as nn
from app.generator import unetParts


class UNET(torch.nn.Module):
    """ Implementation of unet """

    def __init__(
            self,
    ) -> None:
        """
        Create the UNET here
        """
        super().__init__()
        self.enc_layer1: unetParts.EncoderLayer = unetParts.EncoderLayer(
            in_channels=3,
            out_channels=64
        )
        self.enc_layer2: unetParts.EncoderLayer = unetParts.EncoderLayer(
            in_channels=64,
            out_channels=128
        )
        self.enc_layer3: unetParts.EncoderLayer = unetParts.EncoderLayer(
            in_channels=128,
            out_channels=256
        )
        self.enc_layer4: unetParts.EncoderLayer = unetParts.EncoderLayer(
            in_channels=256,
            out_channels=512
        )
        # Middle layer
        self.middle_layer: unetParts.MiddleLayer = unetParts.MiddleLayer(
            in_channels=512,
            out_channels=1024,
        )
        # Decoding layer
        self.dec_layer1: unetParts.DecoderLayer = unetParts.DecoderLayer(
            in_channels=1024,
            out_channels=512,
        )
        self.dec_layer2: unetParts.DecoderLayer = unetParts.DecoderLayer(
            in_channels=512,
            out_channels=256,
        )

        self.dec_layer3: unetParts.DecoderLayer = unetParts.DecoderLayer(
            in_channels=256,
            out_channels=128,
        )
        self.dec_layer4: unetParts.DecoderLayer = unetParts.DecoderLayer(
            in_channels=128,
            out_channels=64,
        )
        self.final_layer: torch.nn.Conv2d = torch.nn.Conv2d(
            in_channels=64,
            out_channels=3,
            kernel_size=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward function
        :param x:
        :return:
        """
        # enc layers
        enc1, conv1 = self.enc_layer1(x=x)  # 64
        enc2, conv2 = self.enc_layer2(x=enc1)  # 128
        enc3, conv3 = self.enc_layer3(x=enc2)  # 256
        enc4, conv4 = self.enc_layer4(x=enc3)  # 512
        # middle layers
        mid = self.middle_layer(x=enc4)  # 1024
        # expanding layers
        # 512
        dec1 = self.dec_layer1(
            input_layer=mid,
            cropping_layer=conv4,
        )
        # 256
        dec2 = self.dec_layer2(
            input_layer=dec1,
            cropping_layer=conv3,
        )
        # 128
        dec3 = self.dec_layer3(
            input_layer=dec2,
            cropping_layer=conv2,
        )
        # 64
        dec4 = self.dec_layer4(
            input_layer=dec3,
            cropping_layer=conv1,
        )
        # 3
        fin_layer = self.final_layer(
            dec4,
        )
        # Interpolate to retain size
        fin_layer_resized = torch.nn.functional.interpolate(fin_layer, 572)
        return fin_layer_resized


class Generator(nn.Module):
    def __init__(self, in_channels=3, features=64):
        super().__init__()
        # Encoder
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1, padding_mode='reflect'),
            nn.LeakyReLU(.2),
        )
        self.down1 = Block(features, features * 2, down=True, act='leaky', use_dropout=False)  # 64
        self.down2 = Block(features * 2, features * 4, down=True, act='leaky', use_dropout=False)  # 32
        self.down3 = Block(features * 4, features * 8, down=True, act='leaky', use_dropout=False)  # 16
        self.down4 = Block(features * 8, features * 8, down=True, act='leaky', use_dropout=False)  # 8
        self.down5 = Block(features * 8, features * 8, down=True, act='leaky', use_dropout=False)  # 4
        self.down6 = Block(features * 8, features * 8, down=True, act='leaky', use_dropout=False)  # 2
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, 4, 2, 1, padding_mode='reflect'),
            nn.ReLU(),  # 1x1
        )
        # Decoder
        self.up1 = Block(features * 8, features * 8, down=False, act='relu', use_dropout=True)
        self.up2 = Block(features * 8 * 2, features * 8, down=False, act='relu', use_dropout=True)
        self.up3 = Block(features * 8 * 2, features * 8, down=False, act='relu', use_dropout=True)
        self.up4 = Block(features * 8 * 2, features * 8, down=False, act='relu', use_dropout=False)
        self.up5 = Block(features * 8 * 2, features * 4, down=False, act='relu', use_dropout=False)
        self.up6 = Block(features * 4 * 2, features * 2, down=False, act='relu', use_dropout=False)
        self.up7 = Block(features * 2 * 2, features, down=False, act='relu', use_dropout=False)
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features * 2, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        # Encoder
        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        d7 = self.down6(d6)
        bottleneck = self.bottleneck(d7)

        # Decoder
        u1 = self.up1(bottleneck)
        u2 = self.up2(torch.cat([u1, d7], 1))
        u3 = self.up3(torch.cat([u2, d6], 1))
        u4 = self.up4(torch.cat([u3, d5], 1))
        u5 = self.up5(torch.cat([u4, d4], 1))
        u6 = self.up6(torch.cat([u5, d3], 1))
        u7 = self.up7(torch.cat([u6, d2], 1))
        return self.final_up(torch.cat([u7, d1], 1))


# block will be use repeatly later
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, act='relu', use_dropout=False):
        super().__init__()
        self.conv = nn.Sequential(
            # the block will be use on both encoder (down=True) and decoder (down=False)
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False, padding_mode='reflect')
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if act == 'relu' else nn.LeakyReLU(.2)
        )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(.5)

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x
