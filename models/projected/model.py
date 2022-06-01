import torch
from torch import nn


class Model(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()

        x_initial = torch.zeros(args.parameter_size)
        x_initial[0] = 1
        self.x: torch.Tensor = nn.parameter.Parameter(x_initial)

    def get_x(self):
        x = self.x.detach().cpu().numpy().copy()
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

        if p + q >= 1:
            self.x.data[0] = s
            self.x.data[1:] = alpha * r
        else:
            self.x.data[0] = (p ** 2 + 1).sqrt()

