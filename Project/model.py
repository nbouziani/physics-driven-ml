from firedrake import *
import numpy as np


def linear_elastic_forward_model(E, nu, strain):
    mu = E / (2 * (1 + nu))
    _lambda = E * nu / ((1 + nu) * (1 - 2 * nu))
    sigma = 2 * mu * strain + _lambda * tr(strain) * Identity(2)
    return sigma


def epsilon(u):
    return 0.5 * (nabla_grad(u) + nabla_grad(u).T)


def sigma(u, E, nu):
    return 2.0 * E * nu / (1.0 - nu) * epsilon(u) + E / (1.0 + nu) * tr(epsilon(u)) * Identity(len(u))


def linear_elastic_constitutive_model(i1, i2):
    mesh = UnitSquareMesh(10, 10)
    V = VectorFunctionSpace(mesh, "Lagrange", 1)
    E = Constant(1.0)  # Young's modulus
    nu = Constant(0.3)  # Poisson's ratio
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Constant((0.0, 0.0))  # Body force

    a = inner(sigma(u, E, nu), epsilon(v)) * dx
    L = inner(f, v) * dx
    bc = DirichletBC(V, Constant((i1, i2)), "on_boundary")
    u = Function(V)  # Displacement field
    solve(a == L, u, bc)
    W = TensorFunctionSpace(mesh, "Lagrange", 1)
    epsilon_values = project(epsilon(u), W)
    return epsilon_values.dat.data[0, :, :].reshape(-1)


def generate_dataset_by_linear_elastic_forward_model(num_samples, E, nu):
    mesh = UnitSquareMesh(10, 10)
    V = TensorFunctionSpace(mesh, "CG", 1)
    Vc = TensorFunctionSpace(mesh, "CG", 1)
    X, y = [], []
    for _ in range(num_samples):
        epsilon = Function(V)
        strain = np.random.rand(2, 2)
        epsilon.interpolate(as_tensor(strain))
        sigma = linear_elastic_forward_model(E, nu, epsilon)
        sigma_proj = project(sigma, Vc)
        X.append(strain.reshape(-1))
        y.append(sigma_proj.dat.data[0, :, :].reshape(-1))
    X = np.array(X)
    y = np.array(y)
    return X, y


if __name__ == '__main__':
    # Generate a dataset
    num_samples = 5
    E = Constant(2.1e11)  # Young's modulus in Pa
    nu = Constant(0.3)  # Poisson's ratio
    X, y = generate_dataset_by_linear_elastic_forward_model(num_samples, E, nu)
    print(X)
