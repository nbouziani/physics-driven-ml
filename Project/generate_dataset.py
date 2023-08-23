from firedrake import *
import numpy as np
from tqdm.auto import tqdm, trange


def forward_model_np(E, nu, strain_tensor):
    lmbda = E * nu / (1 + nu) / (1 - 2 * nu)
    mu = E / 2 / (1 + nu)
    # lmbda*tr(eps(v))*Identity(d) + 2*mu*eps(v), 0.1 is the trace of eps(v)
    s = lmbda * np.trace(strain_tensor) * np.eye(2) + 2 * mu * strain_tensor
    return s


def generate_dataset_by_linear_elasticity(size=500):
    X, y = [], []
    for i in tqdm(range(size)):
        # Randomly generate E and nu within given ranges
        E = np.random.uniform(30e3, 90e3)  # Young's modulus in Pa
        nu = np.random.uniform(0.1, 0.3)  # Poisson's ratio
        mesh = UnitSquareMesh(1, 1)
        V_tensor = TensorFunctionSpace(mesh, "CG", 1)
        # Generate diagonal elements
        a11, a12, a22 = [np.random.uniform(-0.1, 0.1) for _ in range(3)]
        # Construct the 2x2 matrix
        strain = np.array([[a11, a12], [a12, a22]])

        # stress = linear_elastic_forward_model(E, nu, strain)
        stress = forward_model_np(E, nu, strain)

        # Flatten and concatenate [E, nu] and strain
        input_data = np.hstack([E, nu, a11, a22, a12])

        X.append(input_data)
        y.append([stress[0, 0], stress[1, 1], stress[0, 1]])
        # y.append([stress.dat.data[0,0,0], stress.dat.data[0,1,1], stress.dat.data[0,0,1]])
    # print(y)
    # Convert lists to numpy arrays

    np.save("../data/datasets/linear_elasticity/X.npy", X)
    np.save("../data/datasets/linear_elasticity/y.npy", y)


if __name__ == '__main__':
    generate_dataset_by_linear_elasticity(1000)
