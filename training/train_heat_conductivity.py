import os
import argparse
import sys
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
from training.utils import TrainingConfig, get_logger
from evaluation.evaluate import evaluate


logger = get_logger("Training")


# Retrieve arguments
parser = argparse.ArgumentParser()
parser.add_argument("--resources_dir", default="../data", type=str, help="Resources directory")
parser.add_argument("--model", default="cnn", type=str, help="one of [encoder-decoder, cnn]")
parser.add_argument("--alpha", default=1e4, type=float, help="Regularisation parameter")
parser.add_argument("--epochs", default=50, type=int, help="Epochs")
parser.add_argument("--learning_rate", default=5e-5, type=float, help="Learning rate")
parser.add_argument("--evaluation_metric", default="L2", type=str, help="Evaluation metric: one of [Lp, H1, Hdiv, Hcurl]")
parser.add_argument("--max_eval_steps", default=5000, type=int, help="Maximum number of evaluation steps")
parser.add_argument("--dataset", default="poisson", type=str, help="Dataset name: one of [poisson]")
parser.add_argument("--model_dir", default="model", type=str, help="Directory name to save trained models")

args = parser.parse_args()
config = TrainingConfig(**dict(args._get_kwargs()))


# Load dataset
dataset_dir = os.path.join(config.resources_dir, "datasets", config.dataset)
if not os.path.exists(dataset_dir):
    raise ValueError(f"Dataset directory {os.path.abspath(dataset_dir)} does not exist")

mesh, train_data, test_data = load_dataset(config)

# Define function space and test function
V = FunctionSpace(mesh, "CG", 1)
v = TestFunction(V)
# Define right-hand side
x, y = SpatialCoordinate(mesh)
with stop_annotating():
    f = Function(V).interpolate(sin(pi * x) * sin(pi * y))
# Define Dirichlet boundary conditions
bcs = [DirichletBC(V, Constant(0.0), "on_boundary")]

# Define the Firedrake operations to be composed with PyTorch
def solve_poisson(κ, u_obs, f, V, bcs):
    """Solve Poisson problem"""
    u = Function(V)
    v = TestFunction(V)
    F = (inner(exp(κ) * grad(u), grad(v)) - inner(f, v)) * dx
    # Solve PDE (using LU factorisation)
    solve(F == 0, u, bcs=bcs, solver_parameters={'ksp_type': 'preonly', 'pc_type': 'lu'})
    # Assemble Firedrake L2-loss (and not l2-loss as in PyTorch)
    return assemble_L2_error(u, u_obs)

def assemble_L2_error(x, x_exact):
    """Assemble L2-loss"""
    return assemble(0.5 * (x - x_exact) ** 2 * dx)

solve_poisson = functools.partial(solve_poisson, f= f, V=V, bcs=bcs)

# Get PyTorch backend from Firedrake (for mapping from Firedrake to PyTorch and vice versa)
fd_backend = fd.get_backend()

# Instantiate model
if config.model == "encoder-decoder":
    n = V.dim()
    model = EncoderDecoder(n)
    config.add_input_shape(n)
elif config.model == "cnn":
    n = V.dim()
    model = CNN(V.dim())
    config.add_input_shape(n)
# Set double precision (default Firedrake type)
model.double()

optimiser = optim.AdamW(model.parameters(), lr=config.learning_rate, eps=1e-8)

max_grad_norm = 1.0
best_error = 0.

κ = Function(V)
u_obs = Function(V)
κ_exact = Function(V)

# Get working tape
tape = get_working_tape()

# Set tape locally to only record the operations relevant to G on the computational graph
with set_working_tape() as tape:
    # Define PyTorch operator for solving the PDE and compute the L2 error (for computing κ -> 0.5 * ||u(κ) - u_obs||^{2}_{L2})
    F = ReducedFunctional(solve_poisson(κ, u_obs), [Control(κ), Control(u_obs)])
    G = fd.torch_operator(F)

# Set tape locally to only record the operations relevant to H on the computational graph
with set_working_tape() as tape:
    # Define PyTorch operator for computing the L2-loss (for computing κ -> 0.5 * ||κ - κ_exact||^{2}_{L2})
    F = ReducedFunctional(assemble_L2_error(κ, κ_exact), [Control(κ), Control(κ_exact)])
    H = fd.torch_operator(F)

# Training loop
for epoch_num in trange(config.epochs, disable=True):
    logger.info(f"Epoch num: {epoch_num}")

    model.train()

    total_loss = 0.0
    total_loss_uκ = 0.0
    total_loss_κ = 0.0
    for step_num, batch in tqdm(enumerate(train_data), total=len(train_data)):

        model.zero_grad()

        # TODO: Add device to batch
        # Convert to PyTorch tensors
        κ_exact, u_obs = [fd_backend.to_ml_backend(x) for x in batch]

        # Forward pass
        κ = model(u_obs)

        # Solve PDE for κ_P and assemble the L2-loss: 0.5 * ||u(κ) - u_obs||^{2}_{L2}
        loss_uκ = G(κ, u_obs)
        total_loss_uκ += loss_uκ.item()
        # Assemble L2-loss: 0.5 * ||κ - κ_exact||^{2}_{L2}
        loss_κ = H(κ, κ_exact)
        total_loss_κ += loss_κ.item()

        # Total loss
        loss = loss_κ + config.alpha * loss_uκ
        total_loss += loss.item()

        # Backprop and perform Adam optimisation
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
        optimiser.step()

    logger.info(f"Total loss: {total_loss/len(train_data)}\
                  \n\t Loss uκ: {total_loss_uκ/len(train_data)}  Loss κ: {total_loss_κ/len(train_data)}")

    # Evaluation on the test random field
    error = evaluate(model, config, test_data, disable_tqdm=True)
    logger.info(f"Error ({config.evaluation_metric}): {error}")

    #error_train, *_ = evaluate(model, config, train_data[:50], disable_tqdm=True)
    #logger.info(f"Debug Error ({config.evaluation_metric}): {error_train}")

    # Save best-performing model
    if error < best_error or epoch_num == 0:
        best_error = error
        # Create directory for trained models
        name_dir = f"{config.dataset}-epoch-{epoch_num}-error_{best_error:.5f}"
        model_dir = os.path.join(config.resources_dir, "saved_models", config.model_dir, name_dir)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        # Save model
        logger.info(f"Saving model checkpoint to {model_dir}\n")
        model_to_save = (
                model.module if hasattr(model, "module") else model
            )  # Take care of distributed/parallel training
        torch.save(model_to_save.state_dict(), os.path.join(model_dir, "model.pt"))
        # Save training arguments together with the trained model
        config.to_file(os.path.join(model_dir, "training_args.json"))
