import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch.optim as optim
import firedrake as fd

from tqdm.auto import tqdm, trange

from firedrake import *
from firedrake_adjoint import *

from dataset_processing.generate_random_conductivity import random_field
from models.autoencoder import EncoderDecoder
from models.cnn import CNN
from training.utils import TrainingConfig
from evaluation.evaluate import evaluate


# Retrieve arguments
parser = argparse.ArgumentParser()
parser.add_argument("--resources_dir", default='', type=str, help="Resources directory")
parser.add_argument("--model", default="encoder", type=str, help="one of [encoder, cnn]")
parser.add_argument("--conductivity", default="circle", type=str, help="one of [circle, random]")
parser.add_argument("--scale_noise", default=5e-3, type=float, help="Noise scaling")
parser.add_argument("--alpha", default=1e-3, type=float, help="Regularisation parameter")
parser.add_argument("--ntrain", default=100, type=int, help="Number of training samples")
parser.add_argument("--epochs", default=100, type=int, help="Epochs")
parser.add_argument("--learning_rate", default=1e-3, type=float, help="Learning rate")
parser.add_argument("--evaluation_metric", default="L2", type=str, help="Evaluation metric: one of [Lp, H1, Hdiv, Hcurl]")
parser.add_argument("--Lx", default=1., type=float, help="Width of the domain")
parser.add_argument("--Ly", default=1., type=float, help="Length of the domain")

args = parser.parse_args()
config = TrainingConfig(**dict(args._get_kwargs()))


def solve_poisson(k, f, V, u_exact):
    """Solve Poisson problem"""
    u = Function(V)
    v = TestFunction(V)
    F = (inner(exp(k) * grad(u), grad(v)) - inner(f, v)) * dx
    bcs = [DirichletBC(V, Constant(0.0), "on_boundary")]
    # Solve PDE
    solve(F == 0, u, bcs=bcs, solver_parameters={'ksp_type': 'preonly', 'pc_type': 'lu'})
    # Assemble Firedrake L2-loss
    return assemble( 0.5 * (u - u_exact) ** 2 * dx)


def residual(k, f, V, u_exact):
    """Assemble Poisson residual"""
    v = TestFunction(V)
    F = (inner(exp(k) * grad(u_exact), grad(v)) - inner(f, v)) * dx
    bcs = [DirichletBC(V, Constant(0.0), "on_boundary")]
    return assemble(F, bcs=bcs)


mesh = RectangleMesh(50, 50, config.Lx, config.Ly)
V = FunctionSpace(mesh, "CG", 1)
v = TestFunction(V)
x, y = SpatialCoordinate(mesh)
f = Function(V).interpolate(sin(pi * x) * sin(pi * y))

with stop_annotating():
    if config.conductivity == "circle":
        k_exact = Function(V).interpolate(conditional((x - Lx / 2)**2 + (y - Ly / 2)**2 < 0.1, 2, 1))
    elif config.conductivity == "random":
        k_exact, = random_field(V)
    u_exact = Function(V)
    F = (inner(exp(k_exact) * grad(u_exact), grad(v)) - inner(f, v)) * dx
    bcs = [DirichletBC(V, Constant(0.0), "on_boundary")]
    # Solve PDE
    solve(F == 0, u_exact, bcs=bcs)
    u_obs = Function(V).assign(u_exact)
    noise = config.scale_noise * np.random.rand(V.dim())
    u_obs.dat.data[:] += noise

# fig, axes = plt.subplots(1, 3, figsize=(15, 5))
# collection = tripcolor(k_exact, axes=axes[0], alpha=1, cmap='jet')
# fig.colorbar(collection);
# collection = tripcolor(u_exact, axes=axes[1], alpha=1, cmap='jet')
# fig.colorbar(collection);
# collection = tripcolor(u_obs, axes=axes[2], alpha=1, cmap='jet')
# fig.colorbar(collection);

# plt.show()

fd_backend = fd.get_backend()

if config.model == 'encoder':
    model = EncoderDecoder(V.dim())
elif config.model == 'cnn':
    model = CNN(V.dim())
# Set double precision
model.double()

k = Function(V).assign(1)
F = ReducedFunctional(solve_poisson(k, f, V, u_exact), Control(k))
G = fd.torch_op(F)

u_obs_P = fd_backend.to_ml_backend(u_obs)
k_exact_P = fd_backend.to_ml_backend(k_exact)

optimiser = optim.AdamW(model.parameters(), lr=config.learning_rate, eps=1e-8)

k_learned = []
step = 10
nepochs = config.epochs

# Generate synthetic dataset
*train_data, test_data = [fd_backend.to_ml_backend(x) for x in random_field(V, config.ntrain, seed=123)]


for epoch_num in trange(nepochs):
    print(f" Epoch num: {epoch_num}")

    model.train()

    for step_num, batch in tqdm(enumerate(train_data), total=config.ntrain):
        optimiser.zero_grad()

        # Forward pass
        k_P = model(batch)

        loss_uk = G(k_P)
        # loss_F = (F_k ** 2).sum()

        loss_k = ((k_exact_P - k_P)**2).sum()
        loss = loss_uk + config.alpha * loss_k

        # Backprop and perform Adam optimisation
        loss.backward()
        optimiser.step()

    if epoch_num % (nepochs/step) == 0 or epoch_num == nepochs-1:
        k_F = fd_backend.from_ml_backend(k_P, V)
        k_learned.append(k_F)
        print(f" Epoch: {epoch_num}  Loss: {loss.item()}")
        print(f" Loss_uk: {loss_uk.item()}  Loss_k: {loss_k.item()}")

    # Evaluation on the test random field
    error = evaluate(model, V, config=config, data=test_data, metric=config.evaluation_metric)
    print(f" Error ({config.evaluation_metric}): {error}")

# Plot
nn = 3
mm = 4
def plots(nn, mm, contour=False):
    p = tripcolor if not contour else tricontour
    _, axes = plt.subplots(nn, mm, figsize=(4*nn, 4*mm))

    for _ in axes:
        for ax in _:
            ax.set_axis_off()

    for i, ki in enumerate(k_learned):
        ax = axes[int(i/mm), i%mm]
        l = p(ki, axes=ax, cmap='jet')
        plt.colorbar(l)
        ax.set_title("$k^{%s}$" % ((i + 1) * step))

    ax = axes[nn-1, mm-1]
    l = p(k_exact, axes=ax, cmap='jet')
    plt.colorbar(l)
    ax.set_title("$k^{exact}$")

    plt.tight_layout()

plots(nn, mm)
plots(nn, mm, contour=True)
plt.show()
    
