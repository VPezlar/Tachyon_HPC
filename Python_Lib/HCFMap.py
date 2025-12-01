import numpy as np
from fontTools.unicodedata import block

import spectral as sp
from matplotlib import pyplot as plt


def RotexMap(xi, eta, Dxi, Deta, Theta, Theta_top, xmin, xmax, ymin, ymax, Ceil, shift):
    # Settings
    k = np.tan(np.deg2rad(Theta))
    n = 1

    # Base 1D Mapped Grids
    xb   = sp.Map1D(xmin, xmax, xi, "False")
    yb   = k * xb + shift
    ytop = np.tan(np.deg2rad(Theta_top)) * xb + (Ceil - xmax * np.tan(np.deg2rad(Theta_top)))

    y = np.zeros((len(eta), len(xi)))
    x = np.zeros((len(eta), len(xi)))

    for i in range(len(eta)):
        # Smoothing Function
        fs_d = ((eta[-1] - eta[i]) / (eta[-1] - eta[0])) ** n
        fs_u = ((eta[0] - eta[i]) / (eta[0] - eta[-1])) ** n

        # Smoothed function
        y[i, :] = fs_d * yb + fs_u * ytop

    for i in range(len(eta)):
        x[i, :] = xb

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

    return [x, y, dxidx, dxidy, detadx, detady, dxdxi, dxdeta, dydxi, dydeta]
