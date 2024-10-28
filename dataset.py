from typing import List, Tuple
from torch.utils.data import Dataset
from torch import Tensor


class GoMoveDataset(Dataset):
    def __init__(self, games_data: str) -> None:
        """
        games_data should be a path that leads to a file containing board states and expert moves
        """
        self.states: List[Tensor] = []  # Board positions
        self.actions: List[int] = []  # Expert moves
        # TODO: Load Go games here
    
    def __len__(self) -> int:
        return len(self.states)
    
    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        return self.states[idx], self.actions[idx]