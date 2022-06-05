import torch
import wandb
import geoopt
import argparse
import importlib
import numpy as np
from plotly import graph_objects as go
from torch.optim import SGD
from torch.nn import functional as F
from geoopt.optim import RiemannianSGD

from models import *


class ACosH(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        if x <= 1:
            return x * 0
        else:
            return x.arccosh()
    
    @staticmethod
    def backward(ctx, grad_out):
        x, = ctx.saved_tensors
        if x <= 1:
            return grad_out * 0
        else:
            return grad_out * 1 / ((x.pow(2) - 1).sqrt())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--parameter_size', type=int, default=10)
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=7777)
    parser.add_argument('--regularizer_term', type=float, default=0)
    parser.add_argument('--clip_grad', type=float, default=1e9)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--norm', type=float, default=1)
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

    manifold = geoopt.manifolds.Lorentz()

    ground_truth = torch.randn(args.parameter_size)
    norm = ground_truth.pow(2).sum().sqrt()
    ground_truth = ground_truth / norm * args.norm
    ground_truth = F.pad(ground_truth, (1, 0))
    ground_truth = manifold.expmap0(ground_truth)

    def objective(x):
        # loss = manifold.dist(x, ground_truth).pow(2)
        inner = -x[0] * ground_truth[0] + (x[1:] * ground_truth[1:]).sum()
        # loss = LN.apply(-inner).pow(2)
        loss = ACosH.apply(-inner).pow(2)
        return loss

    model_module = importlib.import_module(f'models.{args.model}')
    model = getattr(model_module, 'Model')(args)

    if args.model == 'riemannian':
        optimizer = RiemannianSGD(model.parameters(), args.lr)
    else:
        optimizer = SGD(model.parameters(), args.lr)

    wandb.init(project='hyperbolic-optimization')
    wandb.run.name = args.exp_name
    wandb.config.update(args)

    trajectory = []
    model.train()
    steps = 1e9
    for epoch in range(1, args.epoch + 1):
        optimizer.zero_grad()
        loss, difference = model(objective)
        if loss.isnan():
            break
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), 
            args.clip_grad
        )
        optimizer.step()

        if args.model == 'projected':
            model.project()
        trajectory.append(model.get_x())
        difference = (model.x - ground_truth).pow(2).mean()
        if difference < 1e-5:
            steps = min(steps, epoch)

        wandb.log({
            'loss': loss.item(),
            'difference': difference.item(),
            'grad_norm': grad_norm.item(),
            'epoch': epoch
        })

        print(f"Epoch {epoch:3d} | Loss: {loss.item():.5f} | Difference: {difference.item():.5f} | Grad: {grad_norm.item()}")

    print(f"==========> Converged at {steps} epoch")
    wandb.log({
        'converged': steps
    })
   
    if args.parameter_size == 2:
        trajectory = np.stack(trajectory)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                mode='lines+markers',
                x=trajectory[:, 1],
                y=trajectory[:, 0],
                showlegend=False
            )
        )
        fig.add_trace(
            go.Scatter(
                mode='markers',
                x=[ground_truth[1].item()],
                y=[ground_truth[0].item()],
                showlegend=False
            )
        )
        fig.update_yaxes(
            scaleanchor='x',
            scaleratio=1
        )

        wandb.log({
            'trajectory': fig
        })
    

    print(f'===========> Ground truth: {ground_truth}')
    print(f'===========> Final results: {model.get_x()}')

