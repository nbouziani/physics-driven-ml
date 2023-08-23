
import matplotlib.pyplot as plt
from firedrake import *
import pandas as pd


def three_point_bending(E, nu, w_max):
    # Create mesh and function space
    length = Constant(1.0)  # Length of the beam
    b = Constant(0.05)  # Thickness of the beam
    h = Constant(0.1)  # Height of the beam
    n = Constant(100)  # Number of elements

    mesh = IntervalMesh(n, length)
    V = FunctionSpace(mesh, "CG", 1)
    # Define trial and test functions
    u = TrialFunction(V)
    v = TestFunction(V)
    # Define Dirichlet boundary conditions
    left_bc = DirichletBC(V, 0.0, 1)
    right_bc = DirichletBC(V, 0.0, 2)
    bc = [left_bc, right_bc]
    I = b * h ** 3 / 12
    return 48 * w_max * E * I / length ** 3


def get_dataset(num_samples, E, nu):
    mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, "CG", 1)
    Vc = FunctionSpace(mesh, "CG", 1)
    X, y = [], []
    for _ in range(num_samples):
        w = np.random.rand()
        w_max = Function(V).interpolate(Constant(w))
        force = three_point_bending(E, nu, w_max)
        force_proj = interpolate(force, Vc)
        X.append(w_max.vector().get_local()[0])
        y.append(force_proj.vector().get_local()[0])
        # Store data in a CSV file
        data = pd.DataFrame({"w_max": X, "force": y})
        data.to_csv("data.csv", index=False)
    return X, y


if __name__ == '__main__':
    # Generate a dataset
    num_samples = 500
    E = Constant(2.1e3)  # Young's modulus in GPa
    nu = Constant(0.3)  # Poisson's ratio
    X, y = get_dataset(num_samples, E, nu)

    # Plot the force-deflection curve
    plt.plot(X, y, 'ro')
    plt.xlabel("Deflection")
    plt.ylabel("Force")
    plt.title("Three-Point Bending Test")
    plt.show()