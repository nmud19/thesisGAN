import torch


class PatchGan(torch.nn.Module):
    """ Patch GAN Architecture """
    @staticmethod
    def create_contracting_block(in_channels: int, out_channels: int):
        """
        Create encoding layer
        :param in_channels:
        :param out_channels:
        :return:
        """
        conv_layer = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
            ),
            torch.nn.ReLU(),
        )
        max_pool = torch.nn.Sequential(
            torch.nn.MaxPool2d(
                stride=2,
                kernel_size=2,
            ),
        )
        layer = torch.nn.Sequential(
            conv_layer,
            max_pool,
        )
        return layer

    def __init__(self, input_channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.resize_channels = torch.nn.Conv2d(
            in_channels=input_channels,
            out_channels=hidden_channels,
            kernel_size=1,
        )

        self.enc1 = self.create_contracting_block(
            in_channels=hidden_channels,
            out_channels=hidden_channels * 2
        )

        self.enc2 = self.create_contracting_block(
            in_channels=hidden_channels * 2,
            out_channels=hidden_channels * 4
        )

        self.enc3 =  self.create_contracting_block(
            in_channels=hidden_channels * 4,
            out_channels=hidden_channels * 8
        )
        self.enc4 =  self.create_contracting_block(
            in_channels=hidden_channels * 8,
            out_channels=hidden_channels * 16
        )

        self.final_layer = torch.nn.Conv2d(
            in_channels=hidden_channels * 16,
            out_channels=1,
            kernel_size=1,
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """ Forward patch gan layer """
        inpt = torch.cat([x, y], axis=1)
        resize_img = self.resize_channels(inpt)
        enc1 = self.enc1(resize_img)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        final_layer = self.final_layer(enc4)
        return final_layer
