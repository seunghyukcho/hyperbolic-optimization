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

from torch import log, exp, norm, mean, arccosh

def log21(x):
    return log(2*x-1)

def exp24(x):
    return exp(x**2+x**4)

def exp2(x):
    return exp(x**2)

def dist2(x, y):
    tmp = 1+2*norm(x-y, dim=-1)*exp2(norm(x, dim=-1))*exp2(norm(y, dim=-1))
    return arccosh(tmp + 1e-9)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--latent_dim', type=int, default=10)
    parser.add_argument('--path', type=str, default='./data/wordnet')
    parser.add_argument('--n_epochs', type=int, default=1000)
    parser.add_argument('--eval_interval', type=int, default=10)
    parser.add_argument('--power', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--seed', type=int, default=7777)
    parser.add_argument('--initial_sigma', type=float, default=0.01)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--model', type=str, 
        choices=['riemannian_p', 'poincare']
    )
    parser.add_argument('--exp_name', type=str, default='UNTITLED')
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.set_default_tensor_type(torch.DoubleTensor)

    dataset = WordNet(args)
    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True
    )
    
    model_module = importlib.import_module(f'models.{args.model}')
    model = getattr(model_module, 'Model')(args, dataset.n_words)
    model = model.cuda()

    if args.model == 'riemannian_p':
        optimizer = RiemannianSGD(model.parameters(), args.lr)
    else:
        optimizer = SGD(model.parameters(), args.lr)

    manifold = geoopt.manifolds.PoincareBall()
    loss_fn = nn.CrossEntropyLoss()

    wandb.init(project='hyperbolic-optimization')
    wandb.run.name = args.exp_name
    wandb.config.update(args)
    # wandb.watch(model, log_freq=10)

    for epoch in range(1, args.n_epochs + 1):
        total_loss = 0.
        model.train()
        for x in loader:
            x = x.cuda()
            for param in model.parameters():
                param.grad = None
            embeds = model(x)
            dists = -dist2(embeds[:, :1], embeds[:, 1:])
            
            loss = loss_fn(
                dists, 
                torch.zeros(dists.size(0), device=x.device).long()
            )
            # print(loss.item())
            if loss.isnan():
                exit()
            
            loss.backward()
            _ = nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            if args.model == 'projected':
                with torch.no_grad():
                    model.project()

            total_loss += loss.item() * dists.size(0)

        print(f"{epoch:8d} Epoch | Total loss: {total_loss / dataset.n_relations}")
        wandb.log({
            'loss': total_loss / dataset.n_relations
        })

        if epoch % args.eval_interval == 0 or epoch == args.n_epochs:
            model.eval()
            with torch.no_grad():
                rank, ap = calculate_metrics(dataset, model, dist2)
                print(f"===========> Mean rank: {rank} | MAP: {ap}")
                wandb.log({
                    'rank': rank,
                    'map': ap
                })

                # print(model(torch.LongTensor([[1]])))
                # print(model.embed.data[1])

                if args.latent_dim == 2:
                    coors = torch.LongTensor(list(range(dataset.n_words))).cuda()
                    coors = model(coors).detach().cpu().numpy()
                    fig = go.Figure()
                    fig.add_trace(
                        go.Scatter(
                            mode='markers',
                            x=coors[:, 0],
                            y=coors[:, 1],
                            showlegend=False
                        )
                    )

                    lines = torch.LongTensor(dataset.relations).cuda()
                    coors = model(lines).detach().cpu().numpy()[:500]
                    for i in range(coors.shape[0]):
                        fig.add_trace(
                            go.Scatter(
                                mode='lines',
                                x=[coors[i, 0, 0], coors[i, 1, 0]],
                                y=[coors[i, 0, 1], coors[i, 1, 1]],
                                showlegend=False,
                                line=dict(color='black')
                            )
                        )

                    fig.update_yaxes(
                        scaleanchor='x',
                        scaleratio=1
                    )
                    wandb.log({
                        'latents': fig
                    })


