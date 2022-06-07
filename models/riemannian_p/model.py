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

        x_0 = torch.empty([self.n_words, self.latent_dim])
        nn.init.normal_(x_0, std=self.initial_sigma)
        self.embed = geoopt.ManifoldParameter(data=x_0, manifold=geoopt.manifolds.PoincareBall()) 

    def forward(self, x):
        x = self.embed[x]
        return x

