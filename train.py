import torch
import wandb
import geoopt
import argparse
import importlib
import numpy as np
from torch import nn
from pathlib import Path
from torch.optim import SGD
from geoopt.optim import RiemannianSGD
from plotly import graph_objects as go
from wordnet import WordNet, calculate_metrics
from torch.utils.data import DataLoader

from models import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--latent_dim', type=int, default=10)
    parser.add_argument('--path', type=str, default='./data/wordnet')
    parser.add_argument('--n_epochs', type=int, default=1000)
    parser.add_argument('--eval_interval', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=7777)
    parser.add_argument('--regularizer_term', type=float, default=0)
    parser.add_argument('--regularizer_power', type=int, default=4)
    parser.add_argument('--clip_grad', type=float, default=1e9)
    parser.add_argument('--initial_sigma', type=float, default=0.01)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--model', type=str, 
        choices=['riemannian', 'landing', 'indirect', 'projected']
    )
    parser.add_argument('--exp_name', type=str, default='UNTITLED')
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.set_default_tensor_type(torch.DoubleTensor)

    dataset = WordNet(args)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    model_module = importlib.import_module(f'models.{args.model}')
    model = getattr(model_module, 'Model')(args, dataset.n_words)
    model = model.cuda()

    if args.model == 'riemannian':
        optimizer = RiemannianSGD(model.parameters(), args.lr)
    else:
        optimizer = SGD(model.parameters(), args.lr)

    manifold = geoopt.manifolds.Lorentz()
    loss_fn = nn.CrossEntropyLoss()

    wandb.init(project='hyperbolic-optimization')
    wandb.run.name = args.exp_name
    wandb.config.update(args)

    for epoch in range(1, args.n_epochs + 1):
        total_loss = 0.
        model.train()
        for x in loader:
            for param in model.parameters():
                param.grad = None
            x = x.cuda()
            embeds = model(x)
            
            dists = manifold.dist(embeds[:, :1], embeds[:, 1:])
            loss = loss_fn(dists, torch.zeros(dists.size(0), device=x.device).long())
            loss.backward()
            optimizer.step()
            if args.model == 'projected':
                model.project()
                # print(model.embed.weight.data.size())

            total_loss += loss.item() * dists.size(0)

        print(f"{epoch:8d} Epoch | Total loss: {total_loss / dataset.n_relations}")
        wandb.log({
            'loss': total_loss / dataset.n_relations
        })

        if epoch % args.eval_interval == 0 or epoch == args.n_epochs:
            with torch.no_grad():
                model.eval()
                rank, ap = calculate_metrics(dataset, model)
                print(f"===========> Mean rank: {rank} | MAP: {ap}")
                wandb.log({
                    'rank': rank,
                    'map': ap
                })

