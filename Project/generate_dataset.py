from firedrake import *
import numpy as np
from tqdm.auto import tqdm, trange
from linear_elasticity.linear_elasticity_model import linear_elastic_forward_model

def forward_model_np(E, nu, strain_tensor):
    lmbda = E * nu / (1 + nu) / (1 - 2 * nu)
    mu = E / 2 / (1 + nu)
    # lmbda*tr(eps(v))*Identity(d) + 2*mu*eps(v), 0.1 is the trace of eps(v)
    s = lmbda * np.trace(strain_tensor) * np.eye(2) + 2 * mu * strain_tensor
    return s


def generate_dataset_by_linear_elasticity(E, nu, size=500):
    X, y = [], []
    for _ in tqdm(range(size)):
        a11, a12, a22 = [np.random.uniform(-0.2, 0.2) for _ in range(3)]
        strain = np.array([[a11, a12], [a12, a22]])
        stress = linear_elastic_forward_model(E, nu, strain)
        X.append(np.hstack([a11, a22, a12]))
        y.append([stress[0, 0], stress[1, 1], stress[0, 1]])

    np.save("../data/datasets/linear_elasticity/X.npy", X)
    np.save("../data/datasets/linear_elasticity/y.npy", y)


if __name__ == '__main__':
    E = Constant(210e9)  # Young's modulus in Pa
    nu = Constant(0.3)  # Poisson's ratio
    generate_dataset_by_linear_elasticity(E, nu, 1000)
