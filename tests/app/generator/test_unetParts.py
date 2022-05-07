from app.generator import unetParts

def test_layer1_encoder_should_return_shape()->None:
    """Checkout layer 1"""
    layer = unetParts.encoderLayer1( 
        in_channels=1, 
        out_channels=64
    )
    print(layer.shape)

test_layer1_encoder_should_return_shape()