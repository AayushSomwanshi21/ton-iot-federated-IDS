import torch.nn as nn


class BinaryClassifier(nn.Module):
    def __init__(self, input_dim: int = 34):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        # Return logits; BCEWithLogitsLoss applies the sigmoid internally.
        return self.network(x)
