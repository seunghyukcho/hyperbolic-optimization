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
        print(self.x[None, 1:].pow(2).sum().sqrt())
        p = torch.linalg.norm(self.x[1:])
        print(p)
        q = self.x[0]
        alpha = self.x[1:] / p
        r = ((p + q) ** 2 - 1) / (2 * (p + q))
        s = ((p + q) ** 2 + 1) / (2 * (p + q))

        # mask = (p + q >= 2)
        # self.x.data[0] = s[mask]
        # self.x.data[~mask, 0] = (p ** 2 + 1).sqrt()[~mask]

        # self.embed.weight.data[mask, 1:] = (alpha * r[:, None])[mask]
        # print(self.manifold.inner(self.embed.weight.data, self.embed.weight.data).min())
        # # print(self.embed.weight.data[:, 0].min())
        # if self.embed.weight.data.isnan().sum():
        #     exit()
        if p + q >= 2:
            self.x.data[0] = s
            self.x.data[1:] = alpha * r
        else:
            self.x.data[0] = (p ** 2 + 1).sqrt()
           
        import geoopt
        m = geoopt.manifolds.Lorentz()
        print(m.inner(self.x.data, self.x.data).max(), m.inner(self.x.data, self.x.data).min())

