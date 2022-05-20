import torch
import geoopt
import argparse
import importlib
import numpy as np
from torch.optim import SGD, Adam
from torch.nn import functional as F
from geoopt.optim import RiemannianSGD, RiemannianAdam

from models import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--parameter_size', type=int, default=10)
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=7777)
    parser.add_argument('--regularizer_term', type=float, default=0)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--model', type=str, choices=['riemannian', 'landing', 'indirect'])
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.set_default_tensor_type(torch.DoubleTensor)

    manifold = geoopt.manifolds.Lorentz()
    ground_truth = torch.randn(args.parameter_size)
    ground_truth = F.pad(ground_truth, (1, 0))
    ground_truth = manifold.expmap0(ground_truth)

    def objective(x):
        return (x - ground_truth).pow(2).mean()

    model_module = importlib.import_module(f'models.{args.model}')
    args.parameter_size += 1
    model = getattr(model_module, 'Model')(args)

    if args.model == 'riemannian':
        optimizer = RiemannianAdam(model.parameters(), args.lr)
    else:
        optimizer = Adam(model.parameters(), args.lr)

    model.train()
    for epoch in range(1, args.epoch + 1):
        optimizer.zero_grad()
        loss, difference = model(objective)
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch:3d} | Loss: {loss.item():.5f} | Difference: {difference.item():.5f}")

    print(f'===========> Ground truth: {ground_truth}')
    print(f'===========> Final results: {model.get_x()}')

