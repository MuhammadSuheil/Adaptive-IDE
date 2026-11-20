"""Small tabular and sensor-sequence networks; no training on import."""

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class TabularMLP(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 3),
        )

    def forward(self, x):
        return self.layers(x)


class FeatureBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(64), nn.LayerNorm(64)
        self.attention = nn.MultiheadAttention(64, 4, dropout=0.1, batch_first=True)
        self.linear1, self.linear2 = nn.Linear(64, 256), nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        normalized = self.norm1(x)
        attended, _ = self.attention(
            normalized, normalized, normalized, need_weights=False
        )
        x = x + self.dropout(attended)
        gate, value = self.linear1(self.norm2(x)).chunk(2, dim=-1)
        return x + self.dropout(self.linear2(self.dropout(torch.relu(gate) * value)))


class FeatureTransformer(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, 64))
        self.bias = nn.Parameter(torch.zeros(n_features, 64))
        self.cls = nn.Parameter(torch.zeros(1, 1, 64))
        nn.init.normal_(self.weight, std=0.02)
        nn.init.normal_(self.cls, std=0.02)
        self.blocks = nn.Sequential(FeatureBlock(), FeatureBlock())
        self.head = nn.Sequential(nn.LayerNorm(64), nn.ReLU(), nn.Linear(64, 3))

    def forward(self, x):
        tokens = x.unsqueeze(-1) * self.weight + self.bias
        tokens = torch.cat([self.cls.expand(len(x), -1, -1), tokens], dim=1)
        return self.head(self.blocks(tokens)[:, 0])


class SensorGRU(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.gru = nn.GRU(n_features, 32, num_layers=1, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(32, 3))

    def forward(self, x, lengths):
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.gru(packed)
        return self.head(hidden[-1])
