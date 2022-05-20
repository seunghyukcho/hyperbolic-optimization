import torch
from torch import nn


class Model(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()

        self.parameter_size = args.parameter_size
        self.regularizer_term = args.regularizer_term

        self.x: torch.Tensor = nn.parameter.Parameter(torch.zeros(self.parameter_size))

    def get_x(self):
        x = self.x.detach().cpu().numpy()
        return x

    def forward(self, objective):
        difference = objective(self.x)
        regularizer = -self.x[0].pow(2) + self.x[1:].pow(2).sum() + 1
        regularizer = regularizer.pow(2) / 2 / self.parameter_size
        loss = difference + self.regularizer_term * regularizer
        print(regularizer.item())
        # print(self.x.max())

        return loss, difference

