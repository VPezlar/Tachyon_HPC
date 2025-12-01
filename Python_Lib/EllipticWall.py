import numpy as np
from scipy.optimize import fsolve


def EllipticWall(a, b, F, I, D, z):

    # Setting up Non-Linear system of equations


    def func(x):

        # Function definitions
        # F1 -> tanh
        F1    = np.sqrt((1 - (I - a - F) ** 2 / a ** 2) * b ** 2) - D
        F1_P  = -b * (I - a - F) / (a ** 2 * np.sqrt(1 - ((I - a - F) ** 2 / a ** 2)))
        F1_PP = -b / (a ** 2 * (1 - (I - a - F) ** 2 / (a ** 2)) ** (3 / 2))

        # F2 -> Ellipse
        F2    = (np.tanh(x[0] * (I - x[2])) / x[1]) + (1 / x[1]) - D
        F2_P  = (x[0] * (1 / np.cosh(x[0] * (I - x[2]))) ** 2) / x[1]
        F2_PP = -(2 * x[0] ** 2 * (1 / np.cosh(x[0] * (I - x[2]))) ** 2 * np.tanh(x[0] * (I - x[2]))) / x[1]

        return [F1 - F2,
                F1_P - F2_P,
                F1_PP - F2_PP]

    # Solving System
    root = fsolve(func, [3.7, 0.1, 2])
    print(root)

    # Forming Wall
    f = np.zeros(len(z))
    for i in range(len(z)):
        if z[i] <= I:
            f[i] = (np.tanh(root[0] * (z[i] - root[2])) / root[1]) + (1 / root[1]) - D
        else:
            f[i] = np.sqrt((1 - ((z[i] - a - F) ** 2 / a ** 2)) * b ** 2) - D

    return f
