import joblib
import numpy as np
import matplotlib.pyplot as plt
from firedrake import *


def three_point_bending_model(E, nu, deflection_max):
    # Calculate relevant parameters
    L = Constant(1.0) # Length of the beam
    b = Constant(1.0)  # Width of the beam
    h = Constant(0.1)  # Height of the beam

    I = (b * h ** 3) / 12  # Moment of inertia
    A = b * h  # Cross-sectional area

    # Calculate the maximum moment and maximum stress
    M_max = (3 * E * I * deflection_max) / (L ** 2)
    stress_max = (M_max * h / 2) / I

    # Calculate the force at various deflection values
    num_points = 100  # Number of points along the deflection range
    deflection_range = np.linspace(0, deflection_max, num_points)
    force_values = (E * A * deflection_range) / L

    # Calculate the corresponding stress values
    stress_values = (force_values * h / 2) / I

    return deflection_range, force_values, stress_values


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

mesh = IntervalMesh(100, 1.0)  # 1D mesh with 100 intervals
V = FunctionSpace(mesh, "CG", 1)  # Continuous Galerkin function space of degree 1


E = Constant(2e9)  # Young's modulus in Pa
nu = Constant(0.3)  # Poisson's ratio
deflection_max = 0.01  # Maximum applied deflection in meters

model = joblib.load("random_forest.pkl")
# Generate the input strain tensor based on the deflection range
deflection_range, force_values, stress_values = enhanced_three_point_bending_model(model, E, nu, deflection_max)

# Plot the force-deflection curve
plt.plot(deflection_range, force_values)
plt.xlabel('Deflection (m)')
plt.ylabel('Force (N)')
plt.title('Force-Deflection Curve')
plt.grid(True)
plt.show()

