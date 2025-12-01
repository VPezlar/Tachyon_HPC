import numpy as np
from scipy.sparse import bmat
from scipy.sparse import lil_matrix
from scipy.sparse import diags as ssd


def FormPBC(Rho, T, Dn):
    # Forming a 1D vector from 2D array
    D_shape = np.array(Dn.shape)
    T    = np.reshape(T, (D_shape[0]), order="C")
    Rho  = np.reshape(Rho, (D_shape[0]), order="C")

    # Getting Baseflow Derivative
    Tn   = Dn @ T

    # Forming Sub-matrices
    BC11 = ssd(Tn) + ssd(T) @ Dn
    BC15 = ssd(Rho) @ Dn
    Z0   = lil_matrix(np.zeros((D_shape[0], D_shape[1])))

    # Forming Block Matrix
    PBC  = bmat([[BC11, Z0, Z0, Z0, BC15],
                 [Z0, Z0, Z0, Z0, Z0],
                 [Z0, Z0, Z0, Z0, Z0],
                 [Z0, Z0, Z0, Z0, Z0],
                 [Z0, Z0, Z0, Z0, Z0]], format="lil")

    return PBC
