import torch
import geoopt
from torch import nn
from copy import deepcopy
from torch.nn import functional as F

class Model(nn.Module):
    def __init__(self, args, n_words) -> None:
        super().__init__()

        self.latent_dim = args.latent_dim
        self.n_words = n_words
        self.initial_sigma = args.initial_sigma

        self.manifold = geoopt.manifolds.Lorentz()

        x_0 = torch.empty([self.n_words, self.latent_dim])
        nn.init.normal_(x_0, std=self.initial_sigma)
        x_0 = F.pad(x_0, (1, 0))
        x_0 = self.manifold.expmap0(x_0)
        # self.embed = nn.parameter.Parameter(
        #     x_0,
        #     requires_grad=True
        # )
        self.embed = nn.Embedding.from_pretrained(x_0, freeze=False)

    def project(self):
        # p = torch.linalg.norm(self.embed[:, 1:], dim=-1)
        p = self.embed.weight.data[:, 1:].pow(2).sum(dim=-1).sqrt()
        q = self.embed.weight.data[:, 0]
        # print(p.max(), p.min())
        # print(q.max(), q.min())
        # print((p + q).min(), (p + q).max())
        # print()
        alpha = self.embed.weight.data[:, 1:] / p[:, None]
        # print((p + q).min(), (p + q).max())
        # print(alpha.isnan().sum())
        # print(alpha.max())
        r1 = ((p + q) ** 2 - 1) / (2 * (p + q))
        s1 = ((p + q) ** 2 + 1) / (2 * (p + q))

        # tmp = (q - 2) / p
        # a = 1 - tmp ** 2
        # b = -2 * tmp
        # c = -3
        # r2 = (-b + (b ** 2 - a * c).sqrt()) / a
        # r2 = (-2 * tmp + (4 * tmp * tmp + 3 * (1 - tmp * tmp))) / (1 - tmp * tmp)
        # s2 = 2 + tmp * r2
        # print((-a*c).min())
        # print(r2.min(), r2.max())
        # print(s2.min(), s2.max())

        # self.embed.weight.data[:, 0] = s
        # self.embed.weight.data[:, 1:] = (alpha * r[:, None])
        # mask = (p + q >= 1)
        # print(mask.sum())

        # self.embed.weight.data[mask, 0] = s1[mask]
        # self.embed.weight.data[mask, 1:] = (alpha * r1[:, None])[mask]

        # self.embed.weight.data[~mask, 0] = (p.pow(2) + 1).sqrt()[~mask]
        self.embed.weight.data[:, 0] = (p.pow(2) + 1).sqrt()
        # self.embed.weight.data[~mask, 0] = s2[~mask]
        # self.embed.weight.data[~mask, 1:] = (alpha * r2[:, None])[~mask]
        # print(mask.sum())
        # tmp = (p[~mask].pow(2) + 1).sqrt() - q[~mask]
        # print((tmp > 0).sum(), tmp.size())
        # new_data = torch.zeros_like(self.embed)

        # new_data[mask, 0] = s[mask]
        # new_data[mask, 1:] = (alpha * r[:, None])[mask]

        # new_data[~mask, 0] = (p ** 2 + 1).sqrt()[~mask]
        # new_data[~mask, 1:] = self.embed[~mask, 1:].data
        # self.embed.copy_(new_data)
        # print((p[~mask].pow(2) + 1).max())
        # print((p.pow(2) + 1).max())
        # self.embed.weight.data[:, 0] = (p.pow(2) + 1).sqrt()
        # p = torch.linalg.norm(self.embed.weight.data[:, 1:], dim=-1)
        # q = self.embed.weight.data[:, 0]
        # print(p.max(), p.min())
        # print(q.max(), q.min())
        # print()
        
        # mask = (p + q >= 2)
        # self.embed.weight.data[mask, 0] = s[mask]
        # self.embed.weight.data[~mask, 0] = (p ** 2 + 1).sqrt()[~mask]
        # self.embed.weight.data[mask, 1:] = (alpha * r[:, None])[mask]

        # q = self.embed.weight.data[:, 0]
        # print(q.min(), q.max())
        # lp = self.manifold.inner(
        #     self.embed.weight.data, 
        #     self.embed.weight.data
        # )
        # print(lp.min(), lp.max())
        # print()
        # if lp.isnan().sum():
        #     exit()

    def forward(self, x):
        x = self.embed(x)
        return x

