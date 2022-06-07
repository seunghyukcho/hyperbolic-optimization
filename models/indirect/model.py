import torch
import geoopt
from torch import nn
from torch.nn import functional as F


class Model(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()

        self.parameter_size = args.parameter_size

        self.x = nn.parameter.Parameter(
            torch.zeros(self.parameter_size)
        )
        self.manifold = geoopt.manifolds.Lorentz()

    def get_x(self):
        x = F.pad(self.x, (1, 0))
        x = self.manifold.expmap0(x)
        x = x.detach().cpu().numpy().copy()
        return x

    def forward(self, objective):
        x = F.pad(self.x, (1, 0))
        x = self.manifold.expmap0(x)
        difference = objective(x)
        loss = difference

        return loss, difference

