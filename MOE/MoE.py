# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import torch
from torch import nn
from MOE.Gating import Gating 

class MoE(nn.Module):
    def __init__(self, trained_experts):
        super(MoE, self).__init__()
        self.experts = trained_experts  
        num_experts = len(trained_experts)
        input_dim = 291  
        self.gating = Gating(input_dim, num_experts)  

    def forward(self, x):
        x = torch.tensor(x)  

        # Get the weights from the gating network
        weights = self.gating(x)  

        # Calculate the agent outputs
        experts_outputs = torch.stack([torch.tensor(expert.predict(observation=x, deterministic=True)[0].tolist()[0]) for expert in self.experts], dim=0)


        actions = np.zeros(29)
        for i in range(len(self.experts)):
            weighted_output = experts_outputs[i] * weights[0][i]
            actions += weighted_output.tolist()
        # print('--moe output actions:', actions)

        return torch.tensor(actions)


