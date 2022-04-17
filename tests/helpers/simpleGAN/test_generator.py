from helpers.simpleGAN.generator import Generator
import torch

# Verify the generator class
def test_genarator_scenarios()->None :
    def test_generator(z_dim, im_dim, hidden_dim, num_test=10000):
        gen = Generator(z_dim, im_dim, hidden_dim).get_gen()
        
        # Check there are six modules in the sequential part
        assert len(gen) == 6
        test_input = torch.randn(num_test, z_dim)
        test_output = gen(test_input)

        # Check that the output shape is correct
        assert tuple(test_output.shape) == (num_test, im_dim)
        assert test_output.max() < 1, "Make sure to use a sigmoid"
        assert test_output.min() > 0, "Make sure to use a sigmoid"
        assert test_output.min() < 0.5, "Don't use a block in your solution"
        assert test_output.std() > 0.05, "Don't use batchnorm here"
        assert test_output.std() < 0.15, "Don't use batchnorm here"


    test_generator(5, 10, 20)
    test_generator(20, 8, 24)