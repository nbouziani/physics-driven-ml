
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
    f = Constant(-1.0)
    F = E*I*u.dx(0).dx(0)*v*dx - f*v*dx

    # Solve the problem
    w = Function(V)
    solve(lhs(F) == rhs(F), w, bcs=bc)

    return 48 * w_max * E * I / length ** 3


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
