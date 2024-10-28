import torch.nn as nn
from torch import Tensor

class GoPolicy(nn.Module):
    def __init__(self) -> None:
        super(GoPolicy, self).__init__()
        
        BOARD_SIZE: int = 361  # 19x19 board
        INPUT_CHANNELS: int = 4*BOARD_SIZE # TODO do I want 3D input? think about this
        
        self.network = nn.Sequential(
            nn.Conv2d(in_channels=INPUT_CHANNELS, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            # TODO Calculate output dim, put more layers here
            nn.Flatten(),
            nn.Linear(BOARD_SIZE, BOARD_SIZE) 
            # TODO softmax???
        )
    
    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)
    
    def loss_imitation() -> float:
        pass
    
    def loss_reinforcement() -> float:
        pass