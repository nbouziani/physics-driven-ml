import argparse
import numpy as np
from mpi4py import MPI
from numpy.random import default_rng

from firedrake import *


comm = MPI.COMM_WORLD


def random_field(V, N=1, m=15, σ=1.4, seed=2023):
    # Generate 2D random field with m modes
    rng = default_rng(seed)
    x, y = SpatialCoordinate(V.ufl_domain())
    fields = []
    for _ in range(N):
        r = 0
        for _ in range(m):
            a, b = rng.standard_normal(2)
            k1, k2 = rng.normal(0, σ, 2)
            θ = 2 * pi * (k1 * x + k2 * y)
            r += Constant(a) * cos(θ) + Constant(b) * sin(θ)
        fields.append(interpolate(sqrt(1 / m) * r, V))
    return fields


def generate_data(V, ntrain, forward='poisson', noise='normal', scale_noise=1., seed=1234):
    """Generate train/test data for

        forward: or a callable
    """
    ks = random_field(V, ntrain=ntrain+1, seed=seed)
    if forward == 'poisson':
        us = []
        v = TestFunction(V)
        x, y = SpatialCoordinate(V.ufl_domain())
        f = Function(V).interpolate(sin(pi * x) * sin(pi * y))
        bcs = [DirichletBC(V, Constant(0.0), "on_boundary")]
        for k in ks:
            u = Function(V)
            F = (inner(exp(k) * grad(u), grad(v)) - inner(f, v)) * dx
            # Solve PDE using LU solver
            solve(F == 0, u, bcs=bcs, solver_parameters={'ksp_type': 'preonly', 'pc_type': 'lu'})
            us.append(u)
    elif callable(forward):
        us = forward(ks, V)
    else:
        raise NotImplementedError('Forward problem not implemented. Use "poisson" or provide a callable for your forward problem.')

    # Add noise to PDE solutions
    if noise == 'normal':
        us_obs = []
        for u in us:
            noise = scale_noise * np.random.rand(V.dim())
            u.dat.data[:] += noise
            us_obs.append(u)
    elif callable(noise):
        us_obs = noise(us)
    else:
        raise NotImplementedError('Noise distribution not implemented. Use "normal" or provide a callable for your noise distribution.')

    # Gather data

    # Split into train/test
    #*ks_train, k_test = ks
    #*us_train, u_test = us
    #*us_obs_train, u_obs_test = us_obs

    #return k_exact, u_exact, u_obs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", default=50, type=int, help="Number of training samples")
    parser.add_argument("--forward", default="poisson", type=str, help="Forward problem (e.g. 'poisson')")
    parser.add_argument("--noise", default="normal", type=str, help="Noise distribution (e.g. 'normal')")
    parser.add_argument("--scale_noise", default=5e-3, type=float, help="Noise scaling")
    parser.add_argument("--nx", default=50, type=int, help="Number of cells in x-direction")
    parser.add_argument("--ny", default=50, type=int, help="Number of cells in y-direction")
    parser.add_argument("--Lx", default=1., type=float, help="Length of the domain")
    parser.add_argument("--Ly", default=1., type=float, help="Width of the domain")
    parser.add_argument("--degree", default=1, type=int, help="Degree of the finite element CG space")

    args = parser.parse_args()

    # Set up mesh and finite element space
    mesh = RectangleMesh(args.nx, args.ny, args.Lx, args.Ly)
    V = FunctionSpace(mesh, "CG", args.degree)
    # Generate data
    generate_data(V, args.ntrain, args.forward, args.noise, args.scale_noise)
