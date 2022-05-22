import torch
from typing import Tuple


class DecoderLayer(torch.nn.Module):
    """Decoder model"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up_sample_layer = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=2,
                stride=2,
                bias=False,
            )
        )
        self.conv_layer = EncoderLayer(
            in_channels=in_channels,
            out_channels=out_channels,
        ).conv_layer

    @staticmethod
    def _get_cropping_shape(previous_layer_shape: torch.Size, current_layer_shape: torch.Size) -> int:
        """ Get the shape to crop """
        return (previous_layer_shape[2] - current_layer_shape[2]) // 2 * -1

    def forward(
            self,
            input_layer: torch.Tensor,
            cropping_layer: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward function to concatenate and conv the figure
        :param cropping_layer:
        :param input_layer:
        :return:
        """
        input_layer = self.up_sample_layer(input_layer)

        cropping_shape = self._get_cropping_shape(
            current_layer_shape=input_layer.shape,
            previous_layer_shape=cropping_layer.shape,
        )

        cropping_layer = torch.nn.functional.pad(
            input=cropping_layer,
            pad=[cropping_shape for _ in range(4)]
        )
        combined_layer = torch.cat(
            tensors=[input_layer, cropping_layer],
            dim=1
        )
        result = self.conv_layer(combined_layer)
        return result


class EncoderLayer(torch.nn.Module):
    """Encoder Layer"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv_layer = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
            ),
            torch.nn.ReLU(),
        )
        self.max_pool = torch.nn.Sequential(
            torch.nn.MaxPool2d(2),
        )
        self.layer = torch.nn.Sequential(
            self.conv_layer,
            self.max_pool,
        )

    def get_conv_layers(self, x: torch.Tensor) -> torch.Tensor:
        """Need to concatenate the layer"""
        return self.conv_layer(x)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass to return conv layer and the max pool layer"""
        conv_output: torch.tensor = self.conv_layer(x)
        fin_out: torch.Tensor = self.max_pool(conv_output)
        return fin_out, conv_output


class MiddleLayer(EncoderLayer):
    """Middle layer only"""

    def forward(self, x: torch.tensor) -> torch.tensor:
        """Forward pass"""
        return self.conv_layer(x)
