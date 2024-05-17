# -*- coding: utf-8 -*-
import torch
from torch import nn

# Define the gating model
class Gating(nn.Module):
    def __init__(self, input_dim, num_experts, dropout_rate=0.1):
        super(Gating, self).__init__()

        # Layers
        self.layer1 = nn.Linear(input_dim, 128)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.layer2 = nn.Linear(128, 256)
        self.leaky_relu1 = nn.LeakyReLU()
        self.dropout2 = nn.Dropout(dropout_rate)

        self.layer3 = nn.Linear(256, 128)
        self.leaky_relu2 = nn.LeakyReLU()
        self.dropout3 = nn.Dropout(dropout_rate)

        self.layer4 = nn.Linear(128, num_experts)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.dropout1(x)

        x = self.layer2(x)
        x = self.leaky_relu1(x)
        x = self.dropout2(x)

        x = self.layer3(x)
        x = self.leaky_relu2(x)
        x = self.dropout3(x)

        top_values, top_indices = torch.topk(self.layer4(x), k=2)
        normalized_weights = torch.zeros_like(self.layer4(x))
        normalized_weights[0, top_indices[0]] = top_values[0] / top_values.sum()
        return normalized_weights