import numpy as np


################################# Chebyshev Collocation Derivative Matrix
def chebyshev(N, M):
    # C-matrix ((ci/cj) * (-1)^(i+j))
    C = np.zeros((N + 1, N + 1))
    D = np.identity(N + 1)
    for i in range(N + 1):
        for j in range(N + 1):
            C[i, j] = ((-1) ** (i + j))
    C[0, :] = C[0, :] * 2
    C[-1, :] = C[-1, :] * 2
    C[:, 0] = C[:, 0] * 0.5
    C[:, -1] = C[:, -1] * 0.5
    #################################
    Z = np.zeros((N + 1, N + 1))
    if (N + 1) % 2 == 0:
        rng = int((N + 1) / 2)
        nrng = rng
    else:
        rng = int((N + 2) / 2)
        nrng = rng - 1
    for i in range(rng):
        for j in range(N + 1):
            Z[i, j] = -2 * (np.sin((np.pi / (2 * N)) * (i + j)) * np.sin((np.pi / (2 * N)) * (i - j)))
    # Flipping
    for i in range(nrng):
        for j in range(N + 1):
            Z[N - i, N - j] = -Z[i, j]
    for i in range(N + 1):
        Z[i, i] = 1
    Z = 1 / Z
    for i in range(N + 1):
        Z[i, i] = 0
    ################################
    D_GLA = np.zeros((N + 1, N + 1, M))
    for i in range(M):
        D = ((i + 1) * Z) * (C * np.tile(np.reshape(np.diag(D), (N + 1, 1)), (1, N + 1)) - D)
        for j in range(N + 1):
            D[j, j] = -sum(D[j, :])
        D_GLA[:, :, i] = D
    return D_GLA


############################################# Gauss-Lobatto Grid
def GLgrid(NumberOfNodes):
    GL_Order = NumberOfNodes - 1
    # Gauss-Lobatto grid
    # Memory allocation for Gauss-Lobatto grid
    grid = np.zeros(NumberOfNodes)
    # Creating Gauss-Lobatto Grid points [-1, 1]
    for i in range(NumberOfNodes):
        grid[i] = np.cos((i * np.pi) / float(GL_Order))
    return grid


############################################# Finite Mapping for Chebyshev
def GLfinMappingMetric(min, max, grid):
    eta_max = max
    eta_min = min
    B1 = (eta_max + eta_min) / 2
    A1 = B1 - eta_min
    metric = 1 / A1
    mapped_grid = A1 * grid + B1
    return [metric, mapped_grid]


def Map1D(min_val, max_val, grid, condition):
    Lx = max_val - min_val
    Lxi = grid[-1] - grid[0]
    xscale = Lx / Lxi
    x_1D = xscale * (grid - grid[0]) + min_val

    if condition == "False":
        return x_1D
    else:
        return [1/xscale, x_1D]

