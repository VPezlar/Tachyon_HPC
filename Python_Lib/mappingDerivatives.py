import numpy as np


def mapMet(xi, xi0, ximax, eta, eta0, eta_max, eta0_global, eta_max_global, x0, xmax, ymax, ymax_global, y0, y0_global, s, n, Dxi, Deta):
    # Allocating arrays
    y = np.zeros((len(eta), len(xi)))
    x = np.zeros((len(eta), len(xi)))

    Lx = xmax - x0
    Lxi = ximax - xi0
    xscale = Lx / Lxi
    x_1D = xscale * (xi - xi0) + x0

    Ly = ymax - y0
    Leta = eta_max - eta0
    yscale = Ly / Leta
    ybase = yscale * (eta - eta0) + y0

    fs = ((eta_max_global - eta) / (eta_max_global - eta0_global)) ** n

    offset = ((xmax - x0) / 2) + x0

    # Function Values
    # Squished and Smoothed
    # for i in range(len(eta)):
    #     for j in range(len(xi)):
    #         y[i, j] = (1 + np.tanh(s * fs[i] * (x[j] - offset))) / (4 / (ymax_global - y0_global)) * fs[i] + ybase[i]

    # Squished
    for i in range(len(eta)):
        for j in range(len(xi)):
            x[i, j] = x_1D[j]

    for i in range(len(eta)):
        for j in range(len(xi)):
            y[i, j] = (1 + np.tanh(s * (x[0, j] - offset))) / (4 / (ymax_global - y0_global)) * fs[i] + ybase[i]

    # Making vectors
    x_vec = np.reshape(x, (len(xi) * len(eta)), order="C")
    y_vec = np.reshape(y, (len(xi) * len(eta)), order="C")

    dxdxi = Dxi @ x_vec
    dxdxi = np.reshape(dxdxi, (len(eta), len(xi)))

    dxdeta = Deta @ x_vec
    dxdeta = np.reshape(dxdeta, (len(eta), len(xi)))

    dydxi = Dxi @ y_vec
    dydxi = np.reshape(dydxi, (len(eta), len(xi)))

    dydeta = Deta @ y_vec
    dydeta = np.reshape(dydeta, (len(eta), len(xi)))

    dxidx = np.zeros((len(eta), len(xi)))
    for i in range(len(eta)):
        for j in range(len(xi)):
            dxidx[i, j] = dydeta[i, j] / (dxdxi[i, j] * dydeta[i, j] - dxdeta[i, j] * dydxi[i, j])

    dxidy = np.zeros((len(eta), len(xi)))
    for i in range(len(eta)):
        for j in range(len(xi)):
            dxidy[i, j] = -dxdeta[i, j] / (dxdxi[i, j] * dydeta[i, j] - dxdeta[i, j] * dydxi[i, j])

    detadx = np.zeros((len(eta), len(xi)))
    for i in range(len(eta)):
        for j in range(len(xi)):
            detadx[i, j] = -dydxi[i, j] / (dxdxi[i, j] * dydeta[i, j] - dxdeta[i, j] * dydxi[i, j])

    detady = np.zeros((len(eta), len(xi)))
    for i in range(len(eta)):
        for j in range(len(xi)):
            detady[i, j] = dxdxi[i, j] / (dxdxi[i, j] * dydeta[i, j] - dxdeta[i, j] * dydxi[i, j])

    return [x[0, :], y, dxidx, dxidy, detadx, detady, dxdxi, dxdeta, dydxi, dydeta]
