import os
import argparse
import functools

import torch
import torch.optim as optim
import firedrake as fd


from tqdm.auto import tqdm, trange

from firedrake import *
from firedrake_adjoint import *

from dataset_processing.generate_data import random_field
from dataset_processing.load_data import load_dataset
from models.autoencoder import EncoderDecoder
from models.cnn import CNN
from training.utils import TrainingConfig
from evaluation.evaluate import evaluate


# Retrieve arguments
parser = argparse.ArgumentParser()
parser.add_argument("--resources_dir", default="../data", type=str, help="Resources directory")
parser.add_argument("--model", default="cnn", type=str, help="one of [encoder-decoder, cnn]")
parser.add_argument("--alpha", default=5e-1, type=float, help="Regularisation parameter")
parser.add_argument("--ntrain", default=100, type=int, help="Number of training samples")
parser.add_argument("--epochs", default=50, type=int, help="Epochs")
parser.add_argument("--learning_rate", default=1e-3, type=float, help="Learning rate")
parser.add_argument("--evaluation_metric", default="L2", type=str, help="Evaluation metric: one of [Lp, H1, Hdiv, Hcurl]")
parser.add_argument("--name_dir", default="poisson_data", type=str, help="Directory name used to access datasets and save trained models")

args = parser.parse_args()
config = TrainingConfig(**dict(args._get_kwargs()))


# Load dataset
dataset_dir = os.path.join(config.resources_dir, "datasets", config.name_dir)
if not os.path.exists(dataset_dir):
    raise ValueError(f"Dataset directory {os.path.abspath(dataset_dir)} does not exist")

mesh, train_data, test_data = load_dataset(config)

# Define function space and test function
V = FunctionSpace(mesh, "CG", 1)
v = TestFunction(V)
# Define right-hand side
x, y = SpatialCoordinate(mesh)
f = Function(V).interpolate(sin(pi * x) * sin(pi * y))
# Define Dirichlet boundary conditions
bcs = [DirichletBC(V, Constant(0.0), "on_boundary")]

# Define the Firedrake operations to be composed with PyTorch
def solve_poisson(k, u_exact, f, V, bcs):
    """Solve Poisson problem"""
    u = Function(V)
    v = TestFunction(V)
    F = (inner(exp(k) * grad(u), grad(v)) - inner(f, v)) * dx
    # Solve PDE (using LU factorisation)
    solve(F == 0, u, bcs=bcs, solver_parameters={'ksp_type': 'preonly', 'pc_type': 'lu'})
    # Assemble Firedrake L2-loss (and not l2-loss as in PyTorch)
    return assemble_L2_error(u, u_exact)

def assemble_L2_error(x, y):
    return assemble( 0.5 * (x - y) ** 2 * dx)

solve_poisson = functools.partial(solve_poisson, f= f, V=V, bcs=bcs)

# Get PyTorch backend from Firedrake (for mapping from Firedrake to PyTorch and vice versa)
fd_backend = fd.get_backend()

# Instantiate model
if config.model == "encoder-decoder":
    model = EncoderDecoder(V.dim())
elif config.model == "cnn":
    model = CNN(V.dim())
# Set double precision
model.double()

optimiser = optim.AdamW(model.parameters(), lr=config.learning_rate, eps=1e-8)

step = 10
best_error = 0.
k = Function(V)
for epoch_num in trange(config.epochs):
    print(f" Epoch num: {epoch_num}")

    model.train()

    for step_num, batch in tqdm(enumerate(train_data), total=len(train_data)):

        model.zero_grad()

        k_exact, u_exact, u_obs = batch

        # Define PyTorch operator for solving the PDE (for computing k -> 0.5 * ||u(k) - u_exact||^{2}_{L2})
        F = ReducedFunctional(solve_poisson(k, u_exact), Control(k))
        G = fd.torch_op(F)

        # Define PyTorch operator for computing the L2-loss (for computing k -> 0.5 * ||k - k_exact||^{2}_{L2})
        F = ReducedFunctional(assemble_L2_error(k, k_exact), Control(k))
        H = fd.torch_op(F)

        # Forward pass
        u_obs_P = fd_backend.to_ml_backend(u_obs)
        k_P = model(u_obs_P)

        # Solve PDE for k_P and assemble the L2-loss: 0.5 * ||u(k) - u_exact||^{2}_{L2}
        loss_uk = G(k_P)
        # Assemble L2-loss: 0.5 * ||k - k_exact||^{2}_{L2}
        loss_k = H(k_P)

        # Total loss
        loss = loss_uk + config.alpha * loss_k

        # Backprop and perform Adam optimisation
        loss.backward()
        optimiser.step()

    print(f" Total loss: {loss.item()}")
    print(f"\t Loss_uk: {loss_uk.item()}  Loss_k: {loss_k.item()}")

    # Evaluation on the test random field
    error = evaluate(model, config, test_data, V)
    print(f" Error ({config.evaluation_metric}): {error}")

    error_train = evaluate(model, config, train_data[:1], V)
    print(f"Debug Error ({config.evaluation_metric}): {error_train}")

    # Save best-performing model
    if error < best_error or epoch_num == 0:
        best_error = error
        # Create directory for trained models
        data_dir = os.path.join(config.resources_dir, "saved_models", config.name_dir)
        model_dir = os.path.join(data_dir, f"epoch-{epoch_num}-error_{best_error:.5f}")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        # Save model
        print(f"Saving model checkpoint to {model_dir}")
        model_to_save = (
                model.module if hasattr(model, "module") else model
            )  # Take care of distributed/parallel training
        torch.save(model_to_save.state_dict(), os.path.join(model_dir, "model.pt"))
        # Save training arguments together with the trained model
        config.to_file(os.path.join(model_dir, "training_args.json"))
