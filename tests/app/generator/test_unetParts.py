from app.generator import unetParts
import torch
import pytest


@pytest.mark.parametrize(
    "in_channels, out_channels, input_size, expected_output_size",
    [
        (
                1,
                64,
                torch.randn(1, 1, 572, 572),
                torch.Size([1, 64, 284, 284])
        ),  # Layer 1
        (
            64,
            128,
            torch.randn(1, 64, 284, 284),
            torch.Size([1, 128, 140, 140]),
        ),  # Layer 2
        (
            128,
            256,
            torch.randn(1, 128, 140, 140),
            torch.Size([1, 256, 68, 68]),
        ),  # Layer 3
        (
            256,
            512,
            torch.randn(1, 256, 68, 68),
            torch.Size([1, 512, 32, 32]),
        ),  # Layer 4
    ],
)
def test_layer_encoder_should_return_shape(
    in_channels: int,
    out_channels: int,
    input_size: torch.tensor,
    expected_output_size: torch.Size,
) -> None:
    """Checkout layer 1"""
    layer = unetParts.EncoderLayer(in_channels=in_channels, out_channels=out_channels)
    actual_final_layer, _ = layer(input_size)
    assert actual_final_layer.size() == expected_output_size


@pytest.mark.parametrize(
    "in_channels, out_channels, input_size, expected_output_size",
    [
        (
            512,
            1024,
            torch.randn(1, 512, 32, 32),
            torch.Size([1, 1024, 28, 28]),
        ),  # Middle layer
    ],
)
def test_layer_middle_should_return_good_values(
    in_channels: int,
    out_channels: int,
    input_size: torch.tensor,
    expected_output_size: torch.Size,
) -> None:
    layer = unetParts.MiddleLayer(in_channels=in_channels, out_channels=out_channels)
    assert layer(input_size).shape == expected_output_size


@pytest.mark.parametrize(
    "in_channels, out_channels, input_size, cropping_layer, expected_output_size",
    [
        (
            1024,
            512,
            torch.randn(1, 1024, 28, 28),
            torch.randn([1, 512, 64, 64]),
            torch.randn([1, 512, 52, 52]),
        ),
        (
            512,
            256,
            torch.randn(1, 512, 52, 52),
            torch.randn([1, 256, 136, 136]),
            torch.randn([1, 256, 100, 100]),
        ),
        (
            256,
            128,
            torch.randn([1, 256, 100, 100]),
            torch.randn([1, 128, 280, 280]),
            torch.randn([1, 128, 196, 196]),
        ),
        (
            128,
            64,
            torch.randn([1, 128, 196, 196]),
            torch.randn([1, 64, 568, 568]),
            torch.randn([1, 64, 388, 388]),
        ),

    ],
)
def test_up_layer_should_return_good_values(
    in_channels: int,
    out_channels: int,
    input_size: torch.tensor,
    cropping_layer: torch.tensor,
    expected_output_size: torch.tensor,
) -> None:
    layer = unetParts.DecoderLayer(in_channels=in_channels, out_channels=out_channels)
    actual_output = layer(input_layer=input_size, cropping_layer=cropping_layer)
    assert actual_output.shape == expected_output_size.shape
