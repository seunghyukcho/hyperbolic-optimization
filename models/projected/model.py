import torch
from torch import nn


class Model(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()

        self.x: torch.Tensor = nn.parameter.Parameter(torch.ones(args.parameter_size))

    def get_x(self):
        x = self.x.detach().cpu().numpy()
        return x

    def forward(self, objective):
        difference = objective(self.x)
        loss = difference
        return loss, difference

    def project(self):
        p = torch.linalg.norm(self.x[1:])
        q = self.x[0]
        alpha = self.x[1:] / p
        r = ((p + q) ** 2 - 1) / (2 * (p + q))
        s = ((p + q) ** 2 + 1) / (2 * (p + q))
        self.x.data[0] = s
        self.x.data[1:] = alpha * r

