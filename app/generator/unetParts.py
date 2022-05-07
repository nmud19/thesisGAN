import torch


class encoderLayer1(torch.nn.Module) : 
    """Layer 1 of encoder"""
    def __init__(self, in_channels:int, out_channels:int) -> None:
        super().__init__()
        self.layer = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding="same",
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding="same",
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
        )
    
    def forward(self, x:torch.tensor)->torch.tensor:
        """Forward pass"""
        return self.layer(x)
        

