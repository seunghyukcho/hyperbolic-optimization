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
        self.regularizer_term = args.regularizer_term

        manifold = geoopt.manifolds.Lorentz()

        x_0 = torch.empty([self.n_words, self.latent_dim])
        nn.init.normal_(x_0, std=self.initial_sigma)
        x_0 = F.pad(x_0, (1, 0))
        x_0 = manifold.expmap0(x_0)
        self.embed = nn.Embedding.from_pretrained(x_0, freeze=False)

    def forward(self, x):
        x = self.embed(x)
        regularizer = -x[..., 0].pow(2) + x[..., 1:].pow(2).sum(dim=-1) + 1
        regularizer = regularizer.pow(2).sum(dim=-1).mean() / 2

        return x, regularizer

