import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch import Tensor

from model import GoPolicy


def train_imitation(
    model: GoPolicy,
    train_loader: DataLoader,
    num_epochs: int,
    learning_rate: float,
    device: torch.device
) -> None:
    criterion = nn.CrossEntropyLoss() # TODO is this a good choice? Check!
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    model.train()
    for epoch in range(num_epochs):
        total_loss: float = 0.0
        for states, expert_actions in train_loader:
            states = states.to(device)
            expert_actions = expert_actions.to(device)
            
            # Forward pass
            predicted_actions: Tensor = model(states)
            
            # Compute loss
            loss: Tensor = criterion(predicted_actions, expert_actions)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Average Loss: {total_loss/len(train_loader)}")

def train_reinforcement(
    model: GoPolicy,
    num_epochs: int,
    learning_rate: float,
    device: torch.device
) -> None:
    pass