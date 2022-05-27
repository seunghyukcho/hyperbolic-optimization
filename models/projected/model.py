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
        nn.init.normal_(x_0, std=self.initial_sigma)
        x_0 = F.pad(x_0, (1, 0))
        x_0 = self.manifold.expmap0(x_0)

        self.embed = nn.Embedding.from_pretrained(x_0, freeze=False)

    def project(self):
        p = torch.linalg.norm(self.embed.weight.data[:, 1:])
        q = self.embed.weight.data[:, 0]
        alpha = self.embed.weight.data[:, 1:] / p
        r = ((p + q) ** 2 - 1) / (2 * (p + q))
        s = ((p + q) ** 2 + 1) / (2 * (p + q))
        
        mask = (p + q >= 2)
        self.embed.weight.data[mask, 0] = s[mask]
        self.embed.weight.data[~mask, 0] = (p ** 2 + 1).sqrt()[mask]
        self.embed.weight.data[mask, 1:] = (alpha * r[:, None])[mask]

    def forward(self, x):
        x = self.embed(x)
        return x

