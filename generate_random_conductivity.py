from firedrake import *
from numpy.random import default_rng


def random_field(V, N=20, σ1=1.5, σ2=1.5):
    # Generate 2D random field with N modes
    rng = default_rng(2023)
    x, y = SpatialCoordinate(V.ufl_domain())
    r = 0
    for _ in range(N):
        a, b = rng.normal(0, σ1, 2)
        k1, k2 = rng.normal(0, σ2, 2)
        θ = 2 * pi * (k1 * x + k2 * y)
        r += Constant(a) * cos(θ) + Constant(b) * sin(θ)
    return interpolate(sqrt(1 / N) * r, V)
