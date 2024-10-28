from typing import NamedTuple, Literal, List, Tuple, Optional, Any
import argparse

import torch
import numpy as np
from torch.utils.data import DataLoader
from torch import Tensor

from dataset import GoMoveDataset
from model import GoPolicy
from train import train_imitation, train_reinforcement


class TrainingArgs(NamedTuple):
   batch_size: int
   num_epochs: int
   learning_mode: Literal["reinforcement", "imitation", "both"]
   learning_rate: float
   save_path: str
   load_path: str

def parse_args() -> TrainingArgs:
   parser = argparse.ArgumentParser()
   
   # Required arguments
   parser.add_argument(
       "--batch-size", 
       type=int, 
       required=True,
       help="Batch size for training"
   )
   parser.add_argument(
       "--num-epochs", 
       type=int, 
       required=True,
       help="Number of epochs to train"
   )
   parser.add_argument(
       "--mode",
       type=str,
       required=True,
       choices=["reinforcement", "imitation", "both"],
       help="Training mode: reinforcement or imitation learning (type 'both' for both)"
   )
   
   # Optional arguments with defaults
   parser.add_argument(
       "--learning-rate",
       type=float,
       default=0.001,
       help="Learning rate (default: 0.001)"
   )
   parser.add_argument(
       "--save-path",
       type=str,
       default="models/go_policy.pth",
       help="Path to save the trained model (default: models/go_policy.pth)"
   )
   parser.add_argument(
       "--load-path",
       type=str,
       default="none",
       help="Path to load a pretrained model from (default: 'none' which initialises a new model)"
   )
   parser.add_argument(
       "--dataset-path",
       type=str,
       default="data",
       help="Path to the dataset (default: data)"
   )

   args = parser.parse_args()
   
   return TrainingArgs(
       batch_size=args.batch_size,
       num_epochs=args.num_epochs,
       learning_mode=args.mode,
       learning_rate=args.learning_rate,
       save_path=args.save_path
       load_path=args.load_path
       dataset_path=args.dataset_path
   )



def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        device: torch.device = torch.device("mps")  # Apple Silicon GPU
        print("Using Apple Silicon GPU")
    elif torch.cuda.is_available():
        device: torch.device = torch.device("cuda")
        print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    else:
        device: torch.device = torch.device("cpu")
        print("Using CPU")
    return device


# TODO: don't know if I need these two functions yet
def prepare_state(board_state: np.ndarray) -> Tensor:
    # TODO: Implement this
    # input will look like this: 
    # 'current_stones' 19x19 1: my stone, 0: free, -1 oponent stone, 
    # 'previous_stones' 19x19 1: my stone, 0: free, -1 oponent stone, 
    # 'valid_moves' 19x19 0 or 1 for (im)possible, 
    # 'liberties' 19x19 gets 0 (for fields without a stone) to any other field the number of freedoms of its group
    pass

def process_model_output(action_probs: Tensor) -> int:
    # TODO: Implement this
    pass

def get_model_move(model: GoPolicy, board_state: np.ndarray) -> int:
    model.eval()
    with torch.no_grad():
        state_tensor: Tensor = prepare_state(board_state)
        action_probs: Tensor = model(state_tensor)
        return process_model_output(action_probs)

def main() -> None:
    device: torch.device = get_device()
    
    args: TrainingArgs = parse_args()
    
    # Load data
    dataset: GoMoveDataset = GoMoveDataset(args.dataset_path)
    train_loader: DataLoader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True
    )
    
    # Initialize model
    if args.load_path != "none":
        with open(args.load_path, "r") as file:
            model: GoPolicy = torch.load(file)
        model.to(device)
    else:
        model: GoPolicy = GoPolicy().to(device)
    
    # Train
    if args.learning_mode in ["imitation", "both"]:
        train_imitation(model, train_loader, num_epochs=args.num_epochs, learning_rate=args.learning_rate, device=device)
        
    if args.learning_mode in ["reinforcement", "both"]:
        train_reinforcement(model, num_epochs=args.num_epochs, learning_rate=args.learning_rate, device=device)
    
    # Save model
    torch.save(model.state_dict(), args.save_path)

if __name__ == "__main__":
    main()