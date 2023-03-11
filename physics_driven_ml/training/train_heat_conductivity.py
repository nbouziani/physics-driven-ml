import os
import argparse
import functools

import torch
import torch.optim as optim
import torch.autograd as torch_ad

from tqdm.auto import tqdm, trange

from torch.utils.data import DataLoader

from firedrake import *
from firedrake_adjoint import *

from physics_driven_ml.dataset_processing import PDEDataset, BatchedElement
from physics_driven_ml.models import EncoderDecoder, CNN
from physics_driven_ml.utils import ModelConfig, get_logger
from physics_driven_ml.evaluation import evaluate


logger = get_logger("Training")


def train(model, config: ModelConfig,
          train_dl: DataLoader, dev_dl: DataLoader,
          G: torch_ad.Function, H: torch_ad.Function):
    """Train the model on a given dataset."""

    optimiser = optim.AdamW(model.parameters(), lr=config.learning_rate, eps=1e-8)

    max_grad_norm = 1.0
    best_error = 0.

    # Training loop
    for epoch_num in trange(config.epochs):
        logger.info(f"Epoch num: {epoch_num}")

        model.train()

        total_loss = 0.0
        total_loss_uκ = 0.0
        total_loss_κ = 0.0
        train_steps = len(train_dl)
        for step_num, batch in tqdm(enumerate(train_dl), total=train_steps):

            model.zero_grad()

            # Move batch to device
            batch = BatchedElement(*[x.to(config.device, non_blocking=True) if isinstance(x, torch.Tensor) else x for x in batch])
            κ_exact = batch.target
            u_obs = batch.u_obs

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

        logger.info(f"Total loss: {total_loss/train_steps}\
                    \n\t Loss uκ: {total_loss_uκ/train_steps}  Loss κ: {total_loss_κ/train_steps}")

        # Evaluation on the test random field
        error = evaluate(model, config, dev_dl, disable_tqdm=True)
        logger.info(f"Error ({config.evaluation_metric}): {error}")

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
            # Take care of distributed/parallel training
            model_to_save = (model.module if hasattr(model, "module") else model)
            torch.save(model_to_save.state_dict(), os.path.join(model_dir, "model.pt"))
            # Save training arguments together with the trained model
            config.to_file(os.path.join(model_dir, "training_args.json"))

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resources_dir", default="../data", type=str, help="Resources directory")
    parser.add_argument("--model", default="cnn", type=str, help="one of [encoder-decoder, cnn]")
    parser.add_argument("--alpha", default=1e4, type=float, help="Regularisation parameter")
    parser.add_argument("--epochs", default=50, type=int, help="Epochs")
    parser.add_argument("--batch_size", default=1, type=int, help="Batch size")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="Learning rate")
    parser.add_argument("--dropout", default=0.1, type=float, help="Dropout rate")
    parser.add_argument("--evaluation_metric", default="L2", type=str, help="Evaluation metric: one of [Lp, H1, Hdiv, Hcurl, , avg_rel]")
    parser.add_argument("--max_eval_steps", default=5000, type=int, help="Maximum number of evaluation steps")
    parser.add_argument("--dataset", default="heat_conductivity", type=str, help="Dataset name")
    parser.add_argument("--model_dir", default="model", type=str, help="Directory name to save trained models")
    parser.add_argument("--device", default="cpu", type=str, help="Device identifier (e.g. 'cuda:0' or 'cpu')")

    args = parser.parse_args()
    config = ModelConfig(**dict(args._get_kwargs()))

    # -- Load dataset -- #

    # Load train dataset
    train_dataset = PDEDataset(dataset=config.dataset, dataset_split="train", data_dir=config.resources_dir)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, collate_fn=train_dataset.collate, shuffle=False)
    # Load test dataset
    test_dataset = PDEDataset(dataset=config.dataset, dataset_split="test", data_dir=config.resources_dir)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, collate_fn=test_dataset.collate, shuffle=False)

    # -- Set PDE inputs (mesh, function space, boundary conditions, ...) -- #

    # Get mesh from dataset
    mesh = train_dataset.mesh
    # Define function space and test function
    V = FunctionSpace(mesh, "CG", 1)
    v = TestFunction(V)
    # Define right-hand side
    x, y = SpatialCoordinate(mesh)
    with stop_annotating():
        f = Function(V).interpolate(sin(pi * x) * sin(pi * y))
    # Define Dirichlet boundary conditions
    bcs = [DirichletBC(V, Constant(0.0), "on_boundary")]

    # -- Define the Firedrake operations to be composed with PyTorch -- #

    def solve_pde(κ, u_obs, f, V, bcs):
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

    solve_pde = functools.partial(solve_pde, f=f, V=V, bcs=bcs)

    # -- Construct the Firedrake torch operators -- #

    κ = Function(V)
    u_obs = Function(V)
    κ_exact = Function(V)

    # Set tape locally to only record the operations relevant to G on the computational graph
    with set_working_tape() as tape:
        # Define PyTorch operator for solving the PDE and compute the L2 error (for computing κ -> 0.5 * ||u(κ) - u_obs||^{2}_{L2})
        F = ReducedFunctional(solve_pde(κ, u_obs), [Control(κ), Control(u_obs)])
        G = torch_operator(F)

    # Set tape locally to only record the operations relevant to H on the computational graph
    with set_working_tape() as tape:
        # Define PyTorch operator for computing the L2-loss (for computing κ -> 0.5 * ||κ - κ_exact||^{2}_{L2})
        F = ReducedFunctional(assemble_L2_error(κ, κ_exact), [Control(κ), Control(κ_exact)])
        H = torch_operator(F)

    # -- Set the model -- #

    config.add_input_shape(V.dim())
    if config.model == "encoder-decoder":
        model = EncoderDecoder(config)
    elif config.model == "cnn":
        model = CNN(config)
    else:
        raise ValueError(f"Unknown model: {config.model}")

    # Set double precision (default Firedrake type)
    model.double()
    # Move model to device
    model.to(config.device)

    # -- Training -- #

    train(model, config=config, train_dl=train_dataloader, dev_dl=test_dataloader, G=G, H=H)
