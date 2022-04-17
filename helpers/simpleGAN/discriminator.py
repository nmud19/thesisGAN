import torch
import torch.nn as nn

class Discriminator(nn.Module):
    
    @staticmethod
    def get_block(input_dim:int, output_dim:int)->nn.Sequential:
        """Get discriminator block"""
        return nn.Sequential(
            nn.Linear(
                in_features=input_dim, 
                out_features=output_dim
            ),
            nn.LeakyReLU(negative_slope=.2)
        )


    def __init__(self, img_dim:int, hidden_dim:int=128) -> None:
        super().__init__()

        self.disc = nn.Sequential(
            self.get_block(
                input_dim=img_dim, 
                output_dim=hidden_dim*4
            ),
            self.get_block(
                input_dim=hidden_dim*4, 
                output_dim=hidden_dim*2
            ),
            self.get_block(
                input_dim=hidden_dim*2, 
                output_dim=hidden_dim
            ),
            nn.Linear(
                in_features=hidden_dim, 
                out_features=1
            ),
            nn.Sigmoid(),
        )

    def forward(self, img:torch.Tensor)-> torch.Tensor:
        # why???
        img_flat = img.view(img.size(0), -1)
        validity = self.disc(img_flat)
        return validity

    def get_disc(self):
        '''
        Returns:
            the sequential model
        '''
        return self.disc


