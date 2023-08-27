from Project.linear_elasticity import linear_elasticity_model
from three_bending_model import three_point_bending
from firedrake import *
import matplotlib.pyplot as plt
import numpy as np


def get_dataset(num_samples, E):
    n = 5  # Number of elements
    mesh = IntervalMesh(n, length)
    V = FunctionSpace(mesh, "CG", 1)
    X, y = [], []
    for _ in range(num_samples):
        w = np.random.rand()
        force = three_point_bending(E)
        X.append(w)
        y.append(force)
    return X, y


num_samples = 50
length = 1
E = Constant(2.1e3)  # Young's modulus in GPa
X, y = get_dataset(num_samples, E)
print(X)
# Plot the force-deflection curve
plt.plot(X, y, 'ro')
plt.xlabel("Deflection")
plt.ylabel("Force")
plt.grid(True)
plt.title("Three-Point Bending Test")
plt.show()
