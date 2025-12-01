import numpy as np


################################ Fourier Grid
def FURgrid(NumberOfNodes):
    return


################################ Fourier Collocation Matrix
def DFourier(N):
    # Memory allocation for Collocation matrix
    D_F1 = np.double(np.zeros((N, N)))
    D_F2 = np.double(np.zeros((N, N)))

    # Memory allocation for Fourier Grid
    grid = np.zeros(N)

    # Creating Fourier Grid points
    for i in range(N):
        grid[i] = (2 * np.pi * i) / float(N)

    # Matrix entries
    for i in range(N):
        for j in range(N):
            if i != j:
                D_F1[i, j] = 0.5 * ((-1) ** (i + j)) * (1 / np.tan(((i - j) * np.pi) / float(N)))
                D_F2[i, j] = -0.5 * ((-1) ** (i + j)) * (1 / np.sin(((i - j) * np.pi) / float(N))) ** 2
            else:
                D_F1[i, j] = 0
                D_F2[i, j] = -((N ** 2) / 12) - (1 / 6)

    return [D_F1, D_F2, grid]


################################ Finite Mapping
def FurMappingMetric(min, max, grid):
    Lx = max - min
    metric = 2 * np.pi / Lx
    grid = (1 / metric) * grid

    return [metric, grid]
