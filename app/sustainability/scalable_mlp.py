import torch
import torch.nn as nn


class ScalableMLP(nn.Module):
    def __init__(self, input_dim, hidden_scale=1.0, dropout=0.3):
        super().__init__()

        h1 = max(32, int(256 * hidden_scale))
        h2 = max(16, int(128 * hidden_scale))
        h3 = max(8, int(64 * hidden_scale))

        self.block1 = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.BatchNorm1d(h1),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.exit1 = nn.Linear(h1, 1)

        self.block2 = nn.Sequential(
            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.exit2 = nn.Linear(h2, 1)

        self.block3 = nn.Sequential(
            nn.Linear(h2, h3),
            nn.BatchNorm1d(h3),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.exit3 = nn.Linear(h3, 1)

        # Initialize weights properly
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, exit_level=3):
        x = self.block1(x)
        if exit_level == 1:
            return self.exit1(x).squeeze(1)

        x = self.block2(x)
        if exit_level == 2:
            return self.exit2(x).squeeze(1)

        x = self.block3(x)
        return self.exit3(x).squeeze(1)
