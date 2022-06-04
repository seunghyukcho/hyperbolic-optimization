import torch
import wandb
from math import sqrt
from torch import nn


class Model(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()

        self.parameter_size = args.parameter_size
        self.regularizer_term = args.regularizer_term
        self.regularizer_power = args.regularizer_power

        x_0 = torch.zeros(self.parameter_size)
        x_0[0] = 1
        self.x = nn.parameter.Parameter(x_0)

    def get_x(self):
        x = self.x.detach().cpu().numpy().copy()
        return x

    def forward(self, objective):
        difference = objective(self.x)

        regularizer = -self.x[0].pow(2) + self.x[1:].pow(2).sum() + 1
        regularizer = regularizer.pow(2)
        
        loss = difference + self.regularizer_term * regularizer

        return loss, difference

