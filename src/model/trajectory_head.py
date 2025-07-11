import torch
import torch.nn as nn

class TrajectoryHead(nn.Module):
    def __init__(self, hidden_size: int, output_dim: int):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 4, output_dim)
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.regressor(hidden_states)