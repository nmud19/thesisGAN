from app.generator import unetGen
import pytest
import torch


@pytest.mark.parametrize(
    "input_size, expected_output_size",
    [
        (
                torch.randn(1, 3, 572, 572),
                torch.Size([1, 3, 572, 572])
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
