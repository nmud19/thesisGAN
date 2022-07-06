import torch

from app.discriminator import patch_gan
import pytest


@pytest.mark.parametrize(
    "in_channels, hidden_channels, x, y, expected_size",
    [
        (
                10,
                1,
                torch.randn(1, 5, 256, 256),
                torch.randn(1, 5, 256, 256),
                torch.Size([1, 1, 16, 16]),
        ),
        (
                6,
                1,
                torch.randn(1, 3, 572, 572),
                torch.randn(1, 3, 572, 572),
                torch.Size([1, 1, 35, 35]),
        )
    ]
)
def test_patch_gan_returns_proper_shape(
        in_channels: int,
        hidden_channels: int,
        x: torch.Tensor,
        y: torch.Tensor,
        expected_size: torch.Size
) -> None:
    obj = patch_gan.PatchGan(
        input_channels=in_channels,
        hidden_channels=hidden_channels
    )
    actual = obj(x, y)
    assert actual.size() == expected_size


@pytest.mark.parametrize(
    "generated, real, expected_size",
    [
        (
                torch.randn(1, 3, 256, 256),
                torch.randn(1, 3, 256, 256),
                torch.Size([1, 1, 26, 26]),
        )
    ]
)
def test_discriminator(
        generated: torch.Tensor,
        real: torch.Tensor,
        expected_size:torch.Size,
) -> None:
    model = patch_gan.Discriminator()
    preds = model(generated, real)
    assert preds.size() == expected_size
