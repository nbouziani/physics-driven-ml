import os
import argparse
import numpy as np
from typing import Union, Callable
from tqdm.auto import tqdm, trange
from numpy.random import default_rng

from firedrake import *

from physics_driven_ml.utils import get_logger


def random_field(V, N: int = 1, m: int = 5, σ: float = 0.6,
                 tqdm: bool = False, seed: int = 2023):
    """Generate N 2D random fields with m modes."""
    rng = default_rng(seed)
    x, y = SpatialCoordinate(V.ufl_domain())
    fields = []
    for _ in trange(N, disable=not tqdm):
        r = 0
        for _ in range(m):
            a, b = rng.standard_normal(2)
            k1, k2 = rng.normal(0, σ, 2)
            θ = 2 * pi * (k1 * x + k2 * y)
            r += Constant(a) * cos(θ) + Constant(b) * sin(θ)
        fields.append(interpolate(sqrt(1 / m) * r, V))
    return fields


def generate_data(V, dataset_dir: str, ntrain: int = 50, ntest: int = 10,
                  forward: Union[str, Callable] = "heat", noise: Union[str, Callable] = "normal",
                  scale_noise: float = 1., seed: int = 1234):
    """Generate train/test data for a given PDE-based forward problem and noise distribution.

    Parameters:
        - V: Firedrake function space
        - dataset_dir: directory to save the generated data
        - ntrain: number of training samples
        - ntest: number of test samples
        - forward: forward model (e.g "heat")
        - noise: noise distribution to form the observed data (e.g. "normal")
        - scale_noise: noise scaling factor
        - seed: random seed

    Custom forward problems:
        One can provide a custom forward problem by specifying a callable for the `forward` argument.
        This callable should take in a list of randomly generated inputs and the function space `V`, and
        it should return a list of Firedrake functions corresponding to the PDE solutions.

    Custom noise perturbations:
        Likewise, one can provide a custom noise perturbation by specifying a callable for the `noise` argument.
        This callable should take in a list of PDE solutions, and it should return a list of Firedrake functions
        corresponding to the observed data, i.e. the perturbed PDE solutions.
    """

    logger.info("\n Generate random fields")

    ks = random_field(V, N=ntrain+ntest, tqdm=True, seed=seed)

    logger.info("\n Generate corresponding PDE solutions")

    if forward == "heat":
        us = []
        v = TestFunction(V)
        x, y = SpatialCoordinate(V.ufl_domain())
        f = Function(V).interpolate(sin(pi * x) * sin(pi * y))
        bcs = [DirichletBC(V, Constant(0.0), "on_boundary")]
        for k in tqdm(ks):
            u = Function(V)
            F = (inner(exp(k) * grad(u), grad(v)) - inner(f, v)) * dx
            # Solve PDE using LU factorisation
            solve(F == 0, u, bcs=bcs, solver_parameters={'ksp_type': 'preonly', 'pc_type': 'lu'})
            us.append(u)
    elif callable(forward):
        us = forward(ks, V)
    else:
        raise NotImplementedError("Forward problem not implemented. Use 'heat' or provide a callable for your forward problem.")

    logger.info("\n Form noisy observations from PDE solutions")

    if noise == "normal":
        us_obs = []
        for u in tqdm(us):
            u_obs = Function(V).assign(u)
            noise = scale_noise * np.random.rand(V.dim())
            # Add noise to PDE solutions
            u_obs.dat.data[:] += noise
            us_obs.append(u_obs)
    elif callable(noise):
        us_obs = noise(us)
    else:
        raise NotImplementedError("Noise distribution not implemented. Use 'normal' or provide a callable for your noise distribution.")

    logger.info(f"\n Generated {ntrain} training samples and {ntest} test samples.")

    # Split into train/test
    ks_train, ks_test = ks[:ntrain], ks[ntrain:]
    us_train, us_test = us[:ntrain], us[ntrain:]
    us_obs_train, us_obs_test = us_obs[:ntrain], us_obs[ntrain:]

    logger.info(f"\n Saving train/test data to {os.path.abspath(dataset_dir)}.")

    # Save train data
    with CheckpointFile(os.path.join(dataset_dir, "train_data.h5"), "w") as afile:
        afile.h5pyfile["n"] = ntrain
        afile.save_mesh(mesh)
        for i, (k, u, u_obs) in enumerate(zip(ks_train, us_train, us_obs_train)):
            afile.save_function(k, idx=i, name="k")
            afile.save_function(u_obs, idx=i, name="u_obs")

    # Save test data
    with CheckpointFile(os.path.join(dataset_dir, "test_data.h5"), "w") as afile:
        afile.h5pyfile["n"] = ntest
        afile.save_mesh(mesh)
        for i, (k, u, u_obs) in enumerate(zip(ks_test, us_test, us_obs_test)):
            afile.save_function(k, idx=i, name="k")
            afile.save_function(u_obs, idx=i, name="u_obs")


if __name__ == "__main__":
    logger = get_logger("Data generation")

    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", default=50, type=int, help="Number of training samples")
    parser.add_argument("--ntest", default=10, type=int, help="Number of test samples")
    parser.add_argument("--forward", default="heat", type=str, help="Forward problem (e.g. 'heat')")
    parser.add_argument("--noise", default="normal", type=str, help="Noise distribution (e.g. 'normal')")
    parser.add_argument("--scale_noise", default=5e-3, type=float, help="Noise scaling")
    parser.add_argument("--nx", default=50, type=int, help="Number of cells in x-direction")
    parser.add_argument("--ny", default=50, type=int, help="Number of cells in y-direction")
    parser.add_argument("--Lx", default=1., type=float, help="Length of the domain")
    parser.add_argument("--Ly", default=1., type=float, help="Width of the domain")
    parser.add_argument("--degree", default=1, type=int, help="Degree of the finite element CG space")
    parser.add_argument("--data_dir", default=os.environ["DATA_DIR"], type=str, help="Data directory")
    parser.add_argument("--dataset_name", default="heat_conductivity", type=str, help="Dataset name")

    args = parser.parse_args()

    # Set up mesh and finite element space
    mesh = RectangleMesh(args.nx, args.ny, args.Lx, args.Ly, name="mesh")
    V = FunctionSpace(mesh, "CG", args.degree)
    # Set up data directory
    dataset_dir = os.path.join(args.data_dir, "datasets", args.dataset_name)
    # Make data directory while dealing with parallelism
    try:
        os.makedirs(dataset_dir)
    except FileExistsError:
        # Another process created the directory
        pass
    # Generate data
    generate_data(V, dataset_dir=dataset_dir, ntrain=args.ntrain,
                  ntest=args.ntest, forward=args.forward,
                  noise=args.noise, scale_noise=args.scale_noise)
