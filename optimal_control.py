import numpy as np
import matplotlib.pyplot as plt

import torch.optim as optim
import firedrake as fd

from firedrake import *
from firedrake_adjoint import *

from generate_random_conductivity import random_field
from models.autoencoder import EncoderDecoder
from models.cnn import CNN


def solve_poisson(k, f, V, u_exact):
    """Solve Poisson problem"""
    u = Function(V)
    v = TestFunction(V)
    F = (inner(exp(k) * grad(u), grad(v)) - inner(f, v)) * dx
    bcs = [DirichletBC(V, Constant(1.0), "on_boundary")]
    # Solve PDE
    solve(F == 0, u, bcs=bcs, solver_parameters={'ksp_type': 'preonly', 'pc_type': 'lu'})
    # Assemble Firedrake L2-loss
    return assemble( 0.5 * (u - u_exact) ** 2 * dx)


def residual(k, f, V, u_exact):
    """Solve Poisson problem"""
    u = Function(V)
    v = TestFunction(V)
    F = (inner(k * grad(u_exact), grad(v)) - inner(f, v)) * dx
    return assemble(F)


mesh = UnitSquareMesh(50, 50)
V = FunctionSpace(mesh, "CG", 1)
v = TestFunction(V)

x, y = SpatialCoordinate(mesh)
f = Function(V).interpolate(sin(pi * x) * sin(pi * y))

conductivity = "circle"
alpha = 1e-3
Lx = 1.0
Ly = 1.0
with stop_annotating():
    if conductivity == "circle":
        k_exact = Function(V).interpolate(conditional((x - Lx / 2)**2 + (y - Ly / 2)**2 < 0.1, 2, 1))
    elif conductivity == "random":
        k_exact = random_field(V)
    u_exact = Function(V)
    F = (inner(exp(k_exact) * grad(u_exact), grad(v)) - inner(f, v)) * dx
    bcs = [DirichletBC(V, Constant(0.0), "on_boundary")]
    # Solve PDE
    solve(F == 0, u_exact, bcs=bcs)
    u_obs = Function(V).assign(u_exact)
    scale_noise = 5e-3
    noise = scale_noise * np.random.rand(V.dim())
    u_obs.dat.data[:] += noise

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
collection = tripcolor(k_exact, axes=axes[0], alpha=1)
fig.colorbar(collection);
collection = tripcolor(u_exact, axes=axes[1], alpha=1)
fig.colorbar(collection);
collection = tripcolor(u_obs, axes=axes[2], alpha=1)
fig.colorbar(collection);

plt.show()

fd_backend = fd.get_backend()

model = 'encoder'
if model == 'encoder':
    model = EncoderDecoder(V.dim())
elif model == 'cnn':
    model = CNN(V.dim())
# Set double precision
model.double()

print('Start Define ReducedFunctional')

k = Function(V).assign(1)
f̂ = ReducedFunctional(solve_poisson(k, f, V, u_exact), Control(k))
# f̂ = ReducedFunctional(residual(k, f, V, u_exact), Control(k))
G = fd.torch_op(f̂)

u_obs_P = fd_backend.to_ml_backend(u_obs)
k_exact_P = fd_backend.to_ml_backend(k_exact)

optimizer = optim.Adam(model.parameters(), lr=0.001)


k_learned = []
n_epochs = 1000
step = 10
for epoch in range(n_epochs):

    optimizer.zero_grad()

    # Forward pass
    k_P = model(u_obs_P)

    loss_uk = G(k_P)
    # loss_F = (F_k ** 2).sum()

    loss_k = ((k_exact_P - k_P)**2).sum()
    loss = loss_uk + alpha * loss_k

    # Backprop and perform Adam optimisation
    loss.backward()
    optimizer.step()

    if epoch % (n_epochs/step) == 0 or epoch == n_epochs-1:
        k_F = fd_backend.from_ml_backend(k_P, V)
        k_learned.append(k_F)
        print(f" Epoch: {epoch}  Loss: {loss.item()}")
        print(f" Loss_uk: {loss_uk.item()}  Loss_k: {loss_k.item()}")

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
        l = p(ki, axes=ax)
        plt.colorbar(l)
        ax.set_title("$k^{%s}$" % ((i + 1) * step))

    ax = axes[nn-1, mm-1]
    l = p(k_exact, axes=ax)
    plt.colorbar(l)
    ax.set_title("$k^{exact}$")

    plt.tight_layout()

plots(nn, mm)
plots(nn, mm, contour=True)
plt.show()
    