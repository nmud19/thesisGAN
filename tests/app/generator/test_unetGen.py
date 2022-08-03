from app.generator import unetGen
import pytest
import torch


@pytest.mark.parametrize(
    "input_size, expected_output_size",
    [
        # (
        #        torch.randn(1, 3, 572, 572),
        #        torch.Size([1, 3, 572, 572])
        # ),
        (
                torch.randn(1, 3, 286, 286),
                torch.Size([1, 3, 286, 286])
        ),
    ]
)
def test_unet_returns_appropriate_image(
        input_size: torch.Tensor,
        expected_output_size: torch.Size,
) -> None:
    """
    UNET returns appropriate shape
    :return:
    """
    unet_obj = unetGen.UNET()
    res = unet_obj(input_size)
    assert res.size() == expected_output_size


# Copied from https://www.kaggle.com/code/iamolivia/pix2pix-anime-sketch-colorization-pytorch
@pytest.mark.parametrize(
    "input_size, expected_output_size",
    [
        (
                torch.randn(1, 3, 256, 256),
                torch.Size([1, 3, 256, 256])
        ),
        (
                torch.randn(1, 3, 512, 512),
                torch.Size([1, 3, 512, 512])
        ),
    ]
)
def test_generator(
        input_size: torch.Tensor,
        expected_output_size: torch.Size,
) -> None:
    model = unetGen.Generator(in_channels=3, features=64)
    preds = model(input_size)
    assert preds.size() == expected_output_size
