import torch
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
