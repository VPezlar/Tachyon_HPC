import spectral as sp
import scipy
import numpy as np


def mapMet(X, eta, eta_min, eta_max, ymin, ymax, fwall, d_xi, d_eta):
    
    # -----------------------------------------------#
    #                 Transformation                 #
    # -----------------------------------------------#
    
    eta2 = sp.GLfinMappingMetric(eta_min, eta_max, eta) 
    eta2 = eta2[1]

    h = sp.GLfinMappingMetric(ymin, ymax, eta)
    h = h[1]
    
    # Allocating arrays
    y = np.zeros((len(eta), len(X)))
    x = np.zeros((len(eta), len(X)))

    Dxi  = scipy.sparse.kron(scipy.sparse.identity(len(eta), format="lil"), d_xi, format="lil")
    Deta = scipy.sparse.kron(d_eta, scipy.sparse.identity(len(X), format="lil"), format="lil")

    # Smoothing function
    eta_Min_Global = -1
    eta_Max_Global = 1
    n = 1
    
    fs = ((eta_Max_Global - eta2) / (eta_Max_Global - eta_Min_Global)) ** n

    for i in range(len(eta)):
        for j in range(len(X)):
            y[i, j] = fwall[j] * fs[i] + h[i]
            x[i, j] = X[j]

    # Forming 2D Shape Vectors
    x_vec = np.reshape(x, (len(X) * len(eta)), order="C")
    y_vec = np.reshape(y, (len(X) * len(eta)), order="C")


    # -----------------------------------------------#
    #                   Metrics                      #
    # -----------------------------------------------#

    # Obtaining Metrics
    dxdxi = Dxi @ x_vec
    dxdxi = np.reshape(dxdxi, (len(eta), len(X)))

    dxdeta = Deta @ x_vec
    dxdeta = np.reshape(dxdeta, (len(eta), len(X)))

    dydxi = Dxi @ y_vec
    dydxi = np.reshape(dydxi, (len(eta), len(X)))

    dydeta = Deta @ y_vec
    dydeta = np.reshape(dydeta, (len(eta), len(X)))

    dxidx = np.zeros((len(eta), len(X)))
    for i in range(len(eta)):
        for j in range(len(X)):
            dxidx[i, j] = dydeta[i, j] / (dxdxi[i, j] * dydeta[i, j] - dxdeta[i, j] * dydxi[i, j])

    dxidy = np.zeros((len(eta), len(X)))
    for i in range(len(eta)):
        for j in range(len(X)):
            dxidy[i, j] = -dxdeta[i, j] / (dxdxi[i, j] * dydeta[i, j] - dxdeta[i, j] * dydxi[i, j])

    detadx = np.zeros((len(eta), len(X)))
    for i in range(len(eta)):
        for j in range(len(X)):
            detadx[i, j] = -dydxi[i, j] / (dxdxi[i, j] * dydeta[i, j] - dxdeta[i, j] * dydxi[i, j])

    detady = np.zeros((len(eta), len(X)))
    for i in range(len(eta)):
        for j in range(len(X)):
            detady[i, j] = dxdxi[i, j] / (dxdxi[i, j] * dydeta[i, j] - dxdeta[i, j] * dydxi[i, j])

    dxdxi = np.reshape(dxdxi, (len(eta) * len(X)), order="C")
    dxdeta = np.reshape(dxdeta, (len(eta) * len(X)), order="C")
    dydxi = np.reshape(dydxi, (len(eta) * len(X)), order="C")
    dydeta = np.reshape(dydeta, (len(eta) * len(X)), order="C")
    dxidx = np.reshape(dxidx, (len(eta) * len(X)), order="C")
    dxidy = np.reshape(dxidy, (len(eta) * len(X)), order="C")
    detadx = np.reshape(detadx, (len(eta) * len(X)), order="C")
    detady = np.reshape(detady, (len(eta) * len(X)), order="C")

    return [x, y, dxidx, dxidy, detadx, detady, dxdxi, dxdeta, dydxi, dydeta]
