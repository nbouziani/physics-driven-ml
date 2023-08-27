
import matplotlib.pyplot as plt
from firedrake import *
import pandas as pd


def three_point_bending(E, w_max):
    # Create mesh and function space
    length = Constant(1.0)  # Length of the beam
    b = Constant(0.05)  # Thickness of the beam
    h = Constant(0.1)  # Height of the beam
    n = 100  # Number of elements
    nx, ny = Constant(10), Constant(5)

    mesh = RectangleMesh(nx, ny, length, h, quadrilateral=True)
    V = FunctionSpace(mesh, "CG", 1)

    # Define trial and test functions
    u = TrialFunction(V)
    v = TestFunction(V)

    x, y = SpatialCoordinate(mesh)
    epsilon = 0.1
    loading_region = And(x >= length / 2 - epsilon, x <= length / 2 + epsilon)
    w = conditional(loading_region, w_max, 0)

    # Define Dirichlet boundary conditions
    # fixed left and right boundary
    # set w_max in the midddle of the beam
    left_bc = DirichletBC(V, 0.0, 1)
    right_bc = DirichletBC(V, 0.0, 2)
    bottom_bc = DirichletBC(V, w, 3)
    top_bc = DirichletBC(V, w, 4)
    bc = [left_bc, right_bc, bottom_bc, top_bc]

    I = b * h ** 3 / 12

    f = Constant(0.0)
    a = E * I * inner(grad(u), grad(v)) * dx
    L = f * v * dx
    u = Function(V)
    solve(a == L, u, bcs=bc)

    force = 48 * w * E * I / length ** 3
    s = assemble(.5 * force * ds(4))
    return s


def enhanced_three_point_bending_model(model, E, nu, deflection_max):
    # Calculate relevant parameters
    L = 1.0  # Length of the beam
    b = 1.0  # Width of the beam
    h = 0.1  # Height of the beam

    I = (b * h ** 3) / 12  # Moment of inertia
    A = b * h  # Cross-sectional area

    # Calculate the maximum moment and maximum stress
    M_max = (3 * E * I * deflection_max) / (L ** 2)

    # Calculate the force at various deflection values
    num_points = 100  # Number of points along the deflection range
    deflection_range = np.linspace(0, deflection_max, num_points)

    force_values = []
    stress_values = []
    for deflection in deflection_range:
        strain_tensor = np.array([[deflection / L, 0], [0, 0]])
        print(strain_tensor)
        output_tensor = model.predict(strain_tensor.reshape(1, 4))
        output_tensor = output_tensor.reshape(-1)

        force = output_tensor[0]
        stress = output_tensor[2]
        #
        force_values.append(force)
        stress_values.append(stress)

    return deflection_range, force_values, stress_values

def get_dataset(num_samples, E):
    mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, "CG", 1)
    Vc = FunctionSpace(mesh, "CG", 1)
    X, y = [], []
    for _ in range(num_samples):
        w = np.random.rand()
        w_max = Function(V).interpolate(Constant(w))
        force = three_point_bending(E, w_max)
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
    X, y = get_dataset(num_samples)

    # Plot the force-deflection curve
    plt.plot(X, y)
    plt.xlabel("Deflection")
    plt.ylabel("Force")
    plt.grid(True)
    plt.title("Three-Point Bending Test")
    plt.show()
