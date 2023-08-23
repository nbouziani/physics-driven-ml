from firedrake import *
import numpy as np


def linear_elastic_forward_model(E, nu, strain):
    mesh = RectangleMesh(20, 20, 1, 1)
    x, y = SpatialCoordinate(mesh)
    V = VectorFunctionSpace(mesh, 'CG', 1)
    v, u_ = TestFunction(V), TrialFunction(V)
    u = Function(V, name="Displacement")
    mu = E / (2 * (1 + nu))
    _lambda = E * nu / ((1 + nu) * (1 - 2 * nu))

    f = Constant((0.0, 0.0))
    exx, eyy, exy = strain[0, 0], strain[1, 1], strain[0, 1]
    uLx = exx * x + exy * y
    uLy = exy * x
    uRx = exx * x + exy * y
    uRy = exy * x

    uBx = exy * y
    uBy = eyy * y + exy * x
    uTx = exy * y
    uTy = eyy * y + exy * x

    bcL = DirichletBC(V, [uLx, uLy], 1)
    bcR = DirichletBC(V, [uRx, uRy], 2)
    bcB = DirichletBC(V, [uBx, uBy], 3)
    bcT = DirichletBC(V, [uTx, uTy], 4)

    a = inner(sigma(u_, mu, _lambda), epsilon(v)) * dx
    L = inner(f, v) * dx
    # Solve PDE
    solve(a == L, u, bcs=[bcL, bcB, bcR, bcT], solver_parameters={'ksp_type': 'preonly', 'pc_type': 'lu'})
    sig = sigma(u, mu, _lambda)
    sxx = assemble(sig[0, 0] * dx)
    syy = assemble(sig[1, 1] * dx)
    sxy = assemble(sig[0, 1] * dx)
    syx = assemble(sig[1, 0] * dx)

    stress_tensor = np.array([[sxx, sxy],
                              [syx, syy]])
    displaced_coordinates = interpolate(SpatialCoordinate(mesh) + u, V)
    displaced_mesh = Mesh(displaced_coordinates)

    return stress_tensor


def epsilon(u):
    return 0.5 * (nabla_grad(u) + nabla_grad(u).T)


def sigma(v, mu, _lambda):
    d = 2
    return _lambda * tr(epsilon(v)) * Identity(d) + 2 * mu * epsilon(v)


# check
def check_forward(E, nu, strain_tensor):
    lmbda = E * nu / (1 + nu) / (1 - 2 * nu)
    mu = E / 2 / (1 + nu)
    # lmbda*tr(eps(v))*Identity(d) + 2*mu*eps(v), 0.1 is the trace of eps(v)
    s = lmbda * np.trace(strain_tensor) * np.eye(2) + 2 * mu * strain_tensor
    return s


if __name__ == '__main__':
    E = 10
    nu = 0.1
    strain_tensor = np.array([[0.1, 0],
                              [0, 0]])
    print(linear_elastic_forward_model(E, nu, strain_tensor))
    print(check_forward(E, nu, strain_tensor))
