import numpy as np
import scipy as sp
import cvxpy as cp
import matplotlib.pyplot as plt


# State-space model
class MPCControllerup:
    def __init__(self):
        self.A = np.array(
            [
                [-0.3324, -1.0684, 0, 0, 0, 0],
                [1.0000, 0, 0, 0, 0, 0],
                [0, 0, -0.0474, -0.1713, 0, 0],
                [0, 0, 0.2500, 0, 0, 0],
                [0, 0, 0, 0, -0.0014, -0.1980],
                [0, 0, 0, 0, 0.2500, 0],
            ]
        )

        self.B = np.array(
            [
                [0.0625, 0, 0],
                [0, 0, 0],
                [0, 0.5000, 0],
                [0, 0, 0],
                [0, 0, 0.0312],
                [0, 0, 0],
            ]
        )

        self.C = np.array([[-0.0401, -0.0308, -0.3769, 0.2964, 0.0192, 0.0181]])

        self.D = np.array([[0, 0, 0]])

        self.dt = 0.1
        self.nx = self.A.shape[0]
        self.nu = self.B.shape[1]
        self.Ad = sp.linalg.expm(self.A * self.dt)
        self.Bd = np.linalg.pinv(self.A) @ (self.Ad - np.eye(self.nx)) @ self.B
        self.Cd = self.C
        self.Dd = self.D
        self.N = 1
        self.M = 1
        self.lam = 1e-5

    def controlup(self, x0, ref):
        if isinstance(ref, (float, int)):
            ref = np.ones(self.N) * ref
        elif isinstance(ref, np.ndarray) and ref.shape == ():
            ref = np.ones(self.N) * ref.item()

        x = cp.Variable((self.nx, self.N + 1))
        u = cp.Variable((self.nu, self.M))

        objective = 0
        constraints = [x[:, 0] == x0]

        for k in range(self.N):
            objective += (self.Cd @ x[:, k + 1] - ref[k]) ** 2
            if k < self.M:
                objective += self.lam * cp.quad_form(u[:, k], np.eye(self.nu))
                constraints += [x[:, k + 1] == self.Ad @ x[:, k] + self.Bd @ u[:, k]]
                constraints += [u[:, k] >= -10, u[:, k] <= 0]
            else:
                constraints += [x[:, k + 1] == self.Ad @ x[:, k]]

        prob = cp.Problem(cp.Minimize(objective), constraints)
        prob.solve()

        return u[:, 0].value


class MPCControllerdown:
    def __init__(self):
        self.A = np.array(
            [
                [-0.3324, -1.0684, 0, 0, 0, 0],
                [1.0000, 0, 0, 0, 0, 0],
                [0, 0, -0.0474, -0.1713, 0, 0],
                [0, 0, 0.2500, 0, 0, 0],
                [0, 0, 0, 0, -0.0014, -0.1980],
                [0, 0, 0, 0, 0.2500, 0],
            ]
        )

        self.B = np.array(
            [
                [0.0625, 0, 0],
                [0, 0, 0],
                [0, 0.5000, 0],
                [0, 0, 0],
                [0, 0, 0.0312],
                [0, 0, 0],
            ]
        )

        self.C = np.array([[-0.0401, -0.0308, -0.3769, 0.2964, 0.0192, 0.0181]])

        self.D = np.array([[0, 0, 0]])

        self.dt = 0.1
        self.nx = self.A.shape[0]
        self.nu = self.B.shape[1]
        self.Ad = sp.linalg.expm(self.A * self.dt)
        self.Bd = np.linalg.pinv(self.A) @ (self.Ad - np.eye(self.nx)) @ self.B
        self.Cd = self.C
        self.Dd = self.D
        self.N = 1
        self.M = 1
        self.lam = 1e-5

    def controldown(self, x0, ref):
        if isinstance(ref, (float, int)):
            ref = np.ones(self.N) * ref
        elif isinstance(ref, np.ndarray) and ref.shape == ():
            ref = np.ones(self.N) * ref.item()

        x = cp.Variable((self.nx, self.N + 1))
        u = cp.Variable((self.nu, self.M))

        objective = 0
        constraints = [x[:, 0] == x0]

        for k in range(self.N):
            objective += (self.Cd @ x[:, k + 1] - ref[k]) ** 2
            if k < self.M:
                objective += self.lam * cp.quad_form(u[:, k], np.eye(self.nu))
                constraints += [x[:, k + 1] == self.Ad @ x[:, k] + self.Bd @ u[:, k]]
                constraints += [u[:, k] >= 0, u[:, k] <= 1]
            else:
                constraints += [x[:, k + 1] == self.Ad @ x[:, k]]

        prob = cp.Problem(cp.Minimize(objective), constraints)
        prob.solve()

        return u[:, 0].value


class MPCControllerupp:
    def __init__(self):
        self.A = np.array(
            [
                [-0.3324, -1.0684, 0, 0, 0, 0],
                [1.0000, 0, 0, 0, 0, 0],
                [0, 0, -0.0474, -0.1713, 0, 0],
                [0, 0, 0.2500, 0, 0, 0],
                [0, 0, 0, 0, -0.0014, -0.1980],
                [0, 0, 0, 0, 0.2500, 0],
            ]
        )

        self.B = np.array(
            [
                [0.0625, 0, 0],
                [0, 0, 0],
                [0, 0.5000, 0],
                [0, 0, 0],
                [0, 0, 0.0312],
                [0, 0, 0],
            ]
        )

        self.C = np.array([[-0.0401, -0.0308, -0.3769, 0.2964, 0.0192, 0.0181]])

        self.D = np.array([[0, 0, 0]])

        self.dt = 0.1
        self.nx = self.A.shape[0]
        self.nu = self.B.shape[1]
        self.Ad = sp.linalg.expm(self.A * self.dt)
        self.Bd = np.linalg.pinv(self.A) @ (self.Ad - np.eye(self.nx)) @ self.B
        self.Cd = self.C
        self.Dd = self.D
        self.N = 1
        self.M = 1
        self.lam = 1e-5

    def controlup(self, x0, ref):
        if isinstance(ref, (float, int)):
            ref = np.ones(self.N) * ref
        elif isinstance(ref, np.ndarray) and ref.shape == ():
            ref = np.ones(self.N) * ref.item()

        x = cp.Variable((self.nx, self.N + 1))
        u = cp.Variable((self.nu, self.M))

        objective = 0
        constraints = [x[:, 0] == x0]

        for k in range(self.N):
            objective += (self.Cd @ x[:, k + 1] - ref[k]) ** 2
            if k < self.M:
                objective += self.lam * cp.quad_form(u[:, k], np.eye(self.nu))
                constraints += [x[:, k + 1] == self.Ad @ x[:, k] - self.Bd @ u[:, k]]
                constraints += [u[:, k] >= 0, u[:, k] <= 250]
            else:
                constraints += [x[:, k + 1] == -self.Ad @ x[:, k]]

        prob = cp.Problem(cp.Minimize(objective), constraints)
        prob.solve()

        return u[:, 0].value
