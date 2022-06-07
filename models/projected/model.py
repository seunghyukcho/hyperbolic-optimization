import torch
from torch import nn


class Model(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()

        x_initial = torch.zeros(args.parameter_size + 1)
        x_initial[0] = 1
        self.x: torch.Tensor = nn.parameter.Parameter(x_initial)

    def get_x(self):
        x = self.x.detach().cpu().numpy().copy()
        return x

    def forward(self, objective):
        difference = objective(self.x)
        loss = difference
        return loss, difference

