import torch
import geoopt
from torch import nn


class Model(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()

        self.parameter_size = args.parameter_size + 1

        initial_x = torch.zeros(self.parameter_size)
        initial_x[..., 0] = 1
        self.x = geoopt.ManifoldParameter(data=initial_x, manifold=geoopt.manifolds.Lorentz()) 

    def get_x(self):
        x = self.x.detach().cpu().numpy().copy()
        return x

    def forward(self, objective):
        difference = objective(self.x)
        loss = difference

        return loss, difference

