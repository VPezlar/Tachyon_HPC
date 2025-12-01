import numpy as np


def Map2D(x, y, Dxi, Deta):
    
    Nx = len(x[0, :])
    Ny = len(y[:, 0])
    
    # Making vectors
    x_vec = np.reshape(x, (Nx * Ny), order="C")
    y_vec = np.reshape(y, (Nx * Ny), order="C")

    dxdxi = Dxi @ x_vec
    dxdxi = np.reshape(dxdxi, (Ny, Nx))

    dxdeta = Deta @ x_vec
    dxdeta = np.reshape(dxdeta, (Ny, Nx))

    dydxi = Dxi @ y_vec
    dydxi = np.reshape(dydxi, (Ny, Nx))

    dydeta = Deta @ y_vec
    dydeta = np.reshape(dydeta, (Ny, Nx))

    dxidx = np.zeros((Ny, Nx))
    for i in range(Ny):
        for j in range(Nx):
            dxidx[i, j] = dydeta[i, j] / (dxdxi[i, j] * dydeta[i, j] - dxdeta[i, j] * dydxi[i, j])

    dxidy = np.zeros((Ny, Nx))
    for i in range(Ny):
        for j in range(Nx):
            dxidy[i, j] = -dxdeta[i, j] / (dxdxi[i, j] * dydeta[i, j] - dxdeta[i, j] * dydxi[i, j])

    detadx = np.zeros((Ny, Nx))
    for i in range(Ny):
        for j in range(Nx):
            detadx[i, j] = -dydxi[i, j] / (dxdxi[i, j] * dydeta[i, j] - dxdeta[i, j] * dydxi[i, j])

    detady = np.zeros((Ny, Nx))
    for i in range(Ny):
        for j in range(Nx):
            detady[i, j] = dxdxi[i, j] / (dxdxi[i, j] * dydeta[i, j] - dxdeta[i, j] * dydxi[i, j])

    return [dxidx, dxidy, detadx, detady, dxdxi, dxdeta, dydxi, dydeta]
