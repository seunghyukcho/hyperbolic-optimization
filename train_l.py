import torch
import wandb
import geoopt
import argparse
import importlib
import numpy as np
from torch import nn
from torch.optim import SGD
from geoopt.optim import RiemannianSGD
from plotly import graph_objects as go
from wordnet import WordNet, calculate_metrics
from torch.utils.data import DataLoader

from models import *


def project(x):
    # p = torch.linalg.norm(self.x[1:])
    p = x[..., 1:].pow(2).sum(dim=-1).sqrt() + 1e-9
    q = x[..., 0]
    alpha = x[..., 1:] / p[..., None]

    r1 = ((p + q)**2 - 1) / (2*(p + q))
    s1 = ((p + q)**2 + 1) / (2*(p + q))
    
    tmp = (q - 2) / p
    a = 1 - tmp ** 2
    b = -2 * tmp
    c = -3
    r2 = (-b + (b ** 2 - a * c).sqrt()) / a
    s2 = 2 + tmp * r2

    print(p.max(), q.max(), r1.max(), r2.max())

    y = torch.zeros_like(x)
    mask = (p + q >= 2)
    y[mask, 0] = s1[mask]
    y[~mask, 0] = s2[~mask]
    # print(alpha.size(), r1[mask, None].size())
    y[mask, 1:] = (alpha * r1[:, None])[mask]
    y[~mask, 1:] = (alpha * r2[:, None])[~mask]
    inner = -y[..., 0].pow(2) + y[..., 1:].pow(2).sum(dim=-1)
    print(inner.min(), inner.max())
    # print((inner != -1).sum())
    # y[..., 0] = s
    # y[..., 1:] = alpha * r[..., None]

    return y


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--latent_dim', type=int, default=2)
    parser.add_argument('--path', type=str, default='./data/wordnet')
    parser.add_argument('--n_epochs', type=int, default=300)
    parser.add_argument('--n_epochs_proj', type=int, default=100)
    parser.add_argument('--eval_interval', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--seed', type=int, default=7777)
    parser.add_argument('--regularizer_term', type=float, default=0)
    parser.add_argument('--clip_grad', type=float, default=1e9)
    parser.add_argument('--initial_sigma', type=float, default=0.01)
    parser.add_argument('--lr', type=float, default=0.5)
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
    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True
    )
    
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
    # wandb.watch(model, log_freq=10)

    for epoch in range(1, args.n_epochs + 1):
        total_loss = 0.
        model.train()
        for x in loader:
            for param in model.parameters():
                param.grad = None
            x = x.cuda()
            if args.model == 'landing':
                embeds, regularizer = model(x)
            else:
                embeds = model(x)
                regularizer = 0

            # inner = -embeds[:, :1, 0] * embeds[:, 1:, 0] + embeds[:, :1, 1:] * embeds[:, 1:, 1:]
            # dists = ACosH.apply(-inner)
            dists = manifold.dist(embeds[:, :1], embeds[:, 1:])
            # print(dists.isnan().sum())
            loss = loss_fn(
                -dists, 
                torch.zeros(dists.size(0), device=x.device).long()
            )
            # print(loss)
            if loss.isnan():
                exit()
            loss = loss + regularizer
            loss.backward()
            # grad = nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            # print(grad)
            optimizer.step()
            if args.model == 'projected':
                # x_prev = torch.tensor(model.get_x())
                x_prev = model.embed.weight.data.clone()
                # print(x_prev.isnan().sum())
                # print(x_prev)
                x_proj = project(x_prev)
                # print(x_proj.isnan().sum())
                # print(x_proj)
                x_proj = geoopt.ManifoldParameter(
                    data=x_proj, 
                    manifold=geoopt.manifolds.Lorentz()
                )
                optimizer_proj = RiemannianSGD([x_proj], lr=0.001)
                for _ in range(args.n_epochs_proj):
                    optimizer_proj.zero_grad()
                    loss_ = (x_prev - x_proj).pow(2).sum() / 2
                    loss_.backward()
                    optimizer_proj.step()
                print(x_proj.isnan().sum())
                print()
                
                model.embed.weight.data.copy_(x_proj.data)

            total_loss += loss.item() * dists.size(0)

        print(f"{epoch:8d} Epoch | Total loss: {total_loss / dataset.n_relations}")
        wandb.log({
            'loss': total_loss / dataset.n_relations
        })

        if epoch % args.eval_interval == 0 or epoch == args.n_epochs:
            model.eval()
            with torch.no_grad():
                rank, ap = calculate_metrics(dataset, model, manifold.dist)
                print(f"===========> Mean rank: {rank} | MAP: {ap}")
                wandb.log({
                    'rank': rank,
                    'map': ap
                })

                if args.latent_dim == 2:
                    coors = torch.LongTensor(list(range(dataset.n_words))).cuda()
                    if args.model == 'landing':
                        coors, _ = model(coors)
                    else:
                        coors = model(coors)
                    coors = coors / (coors[:, :1] + 1)
                    coors = coors[:, 1:].detach().cpu().numpy()
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
                    if args.model == 'landing':
                        coors, _ = model(lines)
                    else:
                        coors = model(lines)
                    coors = coors[:500]
                    coors = coors / (coors[..., :1] + 1)
                    coors = coors[..., 1:].detach().cpu().numpy()
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


