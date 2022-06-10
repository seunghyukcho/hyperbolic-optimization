# Numerically Stable Optimization on Hyperbolic Space

This repository contains the code for the team project of CSED490Y.

## Reproduction
To reproduce our experiments, follow the commands.

### Convergence Test
`python train.py --parameter_size 1 --n_epochs 250 --n_epochs_proj 100 --lr 0.01 --norm 5 --model <model>`.

Form the `<model>`, you can use `riemannian`, `indirect`, `projected`, and `poincare`.

### WordNet
To run the WordNet task, you need to switch the branch using the following command:

`git checkout -t origin/wordnet`.

You can reproduce the experiments on our final report by using the following commands:

`python train_l.py --latent_dim 2 --model riemannian`

`python train_l.py --latent_dim 2 --model indirect`

`python train_p_org.py --latent_dim 2 --model riemannian_p`

`python train_p_new.py --latent_dim 2 --model riemannian_p`
