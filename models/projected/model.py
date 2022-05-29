import torch
import geoopt
from torch import nn
from torch.nn import functional as F

class Model(nn.Module):
    def __init__(self, args, n_words) -> None:
        super().__init__()

        self.latent_dim = args.latent_dim
        self.n_words = n_words
        self.initial_sigma = args.initial_sigma

        self.manifold = geoopt.manifolds.Lorentz()

        x_0 = torch.empty([self.n_words, self.latent_dim])
        nn.init.uniform_(x_0, a=-self.initial_sigma, b=self.initial_sigma)
        x_0 = F.pad(x_0, (1, 0))
        x_0 = self.manifold.expmap0(x_0)

        self.embed = nn.Embedding.from_pretrained(x_0, freeze=False)

    def project(self):
        p = torch.linalg.norm(self.embed.weight.data[:, 1:], dim=-1)
        q = self.embed.weight.data[:, 0]
        # print(p.max(), p.min())
        # print(q.max(), q.min())
        # print((p + q).min(), (p + q).max())
        # print()
        alpha = self.embed.weight.data[:, 1:] / p[:, None]
        # print((p + q).min(), (p + q).max())
        # print(alpha.isnan().sum())
        # print(alpha.max())
        r = ((p + q) ** 2 - 1) / (2 * (p + q))
        s = ((p + q) ** 2 + 1) / (2 * (p + q))

        mask = (p + q >= 2)

        self.embed.weight.data[mask, 0] = s[mask]
        self.embed.weight.data[mask, 1:] = (alpha * r[:, None])[mask]
        
        self.embed.weight.data[~mask, 0] = (p[~mask].pow(2) + 1).sqrt()
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

