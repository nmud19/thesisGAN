from helpers.simpleGAN.discriminator import Discriminator
import torch
import torch.nn as nn

# Verify the discriminator block function
def test_discriminator_block() : 
    def test_disc_block(in_features, out_features, num_test=10000)->None:
        disc = Discriminator(in_features, out_features)
        block = disc.get_block(in_features, out_features)

        # Check there are two parts
        assert len(block) == 2
        test_input = torch.randn(num_test, in_features)
        test_output = block(test_input)

        # Check that the shape is right
        assert tuple(test_output.shape) == (num_test, out_features)
        
        # Check that the LeakyReLU slope is about 0.2
        assert -test_output.min() / test_output.max() > 0.1
        assert -test_output.min() / test_output.max() < 0.3
        assert test_output.std() > 0.3
        assert test_output.std() < 0.5

    test_disc_block(25, 12)
    test_disc_block(15, 28)

# Verify the discriminator class
def test_disc_works_fine()->None : 
    def test_discriminator(z_dim, hidden_dim, num_test=100):
        
        disc = Discriminator(z_dim, hidden_dim).get_disc()

        # Check there are four(+1 for sigmoid) parts
        assert len(disc) == 5

        # Check the linear layer is correct
        test_input = torch.randn(num_test, z_dim)
        test_output = disc(test_input)
        assert tuple(test_output.shape) == (num_test, 1)
        
        # Don't use a block
        assert not isinstance(disc[-1], nn.Sequential)

    test_discriminator(5, 10)
    test_discriminator(20, 8)