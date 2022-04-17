import torch.nn as nn
import torch

class Generator(nn.Module):
    @staticmethod
    def create_block(input_dim:int, output_dim:int)->nn.Sequential:
        """
        Create a block for code. 
        This is a sequential block 
            Linear
            Batch Norm
            RELU
        """
        return nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=output_dim),
            nn.BatchNorm1d(num_features=output_dim), 
            nn.ReLU(inplace=True)
        )

    def __init__(self, latent_dim:int, img_shape:int, hidden_dim:int=128):
        super().__init__()
        self.img_shape = img_shape
        self.gen = nn.Sequential(
            self.create_block(input_dim=latent_dim, output_dim=hidden_dim),
            self.create_block(input_dim=hidden_dim, output_dim=hidden_dim*2),
            self.create_block(input_dim=hidden_dim*2, output_dim=hidden_dim*4),
            self.create_block(input_dim=hidden_dim*4, output_dim=hidden_dim*8),
            nn.Linear(hidden_dim*8, img_shape),
            nn.Sigmoid(),
        )

    def forward(self, noise:torch.Tensor)->torch.Tensor:
        """Forward pass for generator"""
        img = self.gen(noise)
        # why??
        # img = img.view(img.size(0), *self.img_shape)
        return img

    def get_gen(self) : 
        """get generator"""
        return self.gen
