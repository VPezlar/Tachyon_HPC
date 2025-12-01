import copy

import numpy as np
import scipy.sparse as sp
from scipy.sparse import bmat
from scipy.sparse import diags as ssd


def BiOp(U, V, W, T, Dy, Dyy, Dz, Dzz, Re, M, alpha):
    # Reshaping 2D arrays to 1D vectors
    order = "C"
    U = np.reshape(U, (len(U[0, :]) * len(U[:, 0])), order=order)
    V = np.reshape(V, (len(V[0, :]) * len(V[:, 0])), order=order)
    W = np.reshape(W, (len(W[0, :]) * len(W[:, 0])), order=order)
    T = np.reshape(T, (len(T[0, :]) * len(T[:, 0])), order=order)

    # Cross Derivative
    Dyz = Dz @ Dy

    # Baseflow - U
    Uy = Dy @ U
    Uyy = Dyy @ U
    Uz = Dz @ U
    Uzz = Dzz @ U

    # Baseflow - V
    Vy = Dy @ V
    Vyy = Dyy @ V
    Vz = Dz @ V
    Vzz = Dzz @ V
    Vyz = Dyz @ V

    # Baseflow - W
    Wy = Dy @ W
    Wyy = Dyy @ W
    Wz = Dz @ W
    Wzz = Dzz @ W
    Wyz = Dyz @ W

    # Baseflow - T
    Ty = Dy @ T
    Tyy = Dyy @ T
    Tz = Dz @ T
    Tzz = Dzz @ T

    # Constants
    Pr = 0.72
    gamma = 1.4
    T_0 = 273.15
    Su = 110.4 / T_0
    MU_0 = 1  # 1.716 * (10 ** (-5))

    # Sutherland's Formula and its derivatives
    MU = MU_0 * (T ** (3 / 2)) * (1 + Su) / (T + Su)
    MU_T = MU_0 * (1 + Su) * (np.sqrt(T) * (T + 3 * Su)) / (2 * (T + Su) ** 2)
    MU_TT = MU_0 * (1 + Su) * ((-T ** 2) - 6 * Su * T + 3 * Su ** 2) / (4 * np.sqrt(T) * (T + Su) ** 3)

    # Baseflow Mu
    MU_y = Dy @ MU
    MU_z = Dz @ MU

    # Density (Chapman Rubesin)
    RHO = 1 / T
    RHO_y = Dy @ RHO
    RHO_z = Dz @ RHO

    # Temperature Conduction Coefficient = Mu
    K = MU
    K_T = MU_T
    K_TT = MU_TT

    # BiGlobal Linear Operator Values
    A11 = np.diag(V) @ Dy + np.diag(W) @ Dz + np.diag(Vy + Wz + 1j * alpha * U)
    A12 = np.diag(1j * alpha * RHO)
    A13 = np.diag(RHO) @ Dy + np.diag(RHO_y)
    A14 = np.diag(RHO) @ Dz + np.diag(RHO_z)
    A15 = np.zeros((len(U), len(U)))

    A21 = np.diag(V * Uy + W * Uz + 1j * alpha * T / (gamma * (M ** 2)))
    A22 = -np.diag(MU / Re) @ (Dyy + Dzz) + np.diag(RHO * V - MU_y / Re) @ Dy + np.diag(RHO * W - MU_z / Re) @ Dz + \
           np.diag(1j * alpha * RHO * U + 4 * MU * (alpha ** 2) / (3 * Re))
    A23 = -np.diag(1j * alpha * MU / (3 * Re)) @ Dy + np.diag(RHO * Uy - 1j * alpha * MU_y / Re)
    A24 = -np.diag(1j * alpha * MU / (3 * Re)) @ Dz + np.diag(RHO * Uz - 1j * alpha * MU_z / Re)
    A25 = -(np.diag(Uy * MU_T / Re) @ Dy + np.diag(Uz * MU_T / Re) @ Dz) + \
          np.diag(1j * alpha * RHO / (gamma * (M ** 2)) - (MU_T / Re) *
                  (Uyy + Uzz - alpha * (2j / 3) * (Vy + Wz)) - (MU_TT / Re) * (Uy * Ty + Uz * Tz))

    A31 = np.diag(T / (gamma * (M ** 2))) @ Dy + np.diag(V * Vy + W * Vz + Ty / (gamma * (M ** 2)))
    A32 = -np.diag(1j * alpha * MU / (3 * Re)) @ Dy + np.diag(2j * alpha * MU_y / (3 * Re))
    A33 = -np.diag(MU / Re) @ ((4 / 3) * Dyy + Dzz) + np.diag(RHO * V - ((4 * MU_y) / (3 * Re))) @ Dy + \
          np.diag(RHO * W - (MU_z / Re)) @ Dz + np.diag(RHO * (1j * alpha * U + Vy) + ((alpha ** 2) * MU / Re))
    A34 = -np.diag(MU / (3 * Re)) @ Dyz - np.diag(MU_z / Re) @ Dy + np.diag(2 * MU_y / (3 * Re)) @ Dz + np.diag(RHO * Vz)
    A35 = np.diag((RHO / (gamma * (M ** 2))) - (2 * MU_T * (2 * Vy - Wz)) / (3 * Re)) @ Dy - np.diag((MU_T * (Vz + Wy)) / Re) @ Dz + \
          np.diag(RHO_y / (gamma * (M ** 2))) - np.diag((MU_TT / Re) * ((2 / 3) * (2 * Vy - Wz) * Ty + (Vz + Wy) * Tz)) - \
          np.diag((MU_T / Re) * ((4 / 3) * Vyy + Vzz + (Wyz / 3) + 1j * alpha * Uy))

    A41 = np.diag(T / (gamma * (M ** 2))) @ Dz + np.diag(V * Wy + W * Wz + Tz / (gamma * (M ** 2)))
    A42 = -np.diag(1j * alpha * MU / (3 * Re)) @ Dz + np.diag(2j * alpha * MU_z / (3 * Re))
    A43 = -np.diag(MU / (3 * Re)) @ Dyz + np.diag(2 * MU_z / (3 * Re)) @ Dy - np.diag(MU_y / Re) @ Dz + np.diag(RHO * Wy)
    A44 = -np.diag(MU / Re) @ (Dyy + (4 / 3) * Dzz) + np.diag(RHO * V - (MU_y / Re)) @ Dy + \
          np.diag(RHO * W - (4 * MU_z) / (3 * Re)) @ Dz + np.diag(RHO * (1j * alpha * U + Wz) + (MU * (alpha ** 2)) / Re)
    A45 = -np.diag((MU_T / Re) * (Wy + Vz)) @ Dy + np.diag((RHO / (gamma * (M ** 2))) - (2 * MU_T * (2 * Wz - Vy)) / (3 * Re)) @ Dz + \
          np.diag(RHO_z / (gamma * (M ** 2)) - (MU_TT / Re) * ((Wy + Vz) * Ty + (2 / 3) * (2 * Wz - Vy) * Tz) -
                  (MU_T / Re) * ((4 / 3) * Wzz + Wyy + (Vyz / 3) + 1j * alpha * Uz))

    A51 = -np.diag((gamma - 1) * T) @ (np.diag(V) @ Dy + np.diag(W) @ Dz) + \
          np.diag((Ty * V + Tz * W) - (gamma - 1) * 1j * alpha * T * U)

    A52 = -np.diag((2 * gamma * (gamma - 1) * (M ** 2) * MU) / Re) @ \
          (np.diag(Uy) @ Dy + np.diag(Uz) @ Dz - np.diag((2 / 3) * 1j * alpha * (Vy + Wz)))

    A53 = -np.diag((2 * gamma * (gamma - 1) * (M ** 2) * MU) / Re) @ \
          (np.diag((2 / 3) * (2 * Vy - Wz)) @ Dy + np.diag(Vz + Wy) @ Dz + np.diag(1j * alpha * Uy)) - \
          np.diag((gamma - 1) * (Ty * RHO + T * RHO_y) + gamma * RHO * Ty)

    A54 = -np.diag((2 * gamma * (gamma - 1) * (M ** 2) * MU) / Re) @ \
          (np.diag(Vz + Wy) @ Dy + np.diag((2 / 3) * (2 * Wz - Vy)) @ Dz + np.diag(1j * alpha * Uz)) - \
          np.diag((gamma - 1) * (Tz * RHO + T * RHO_z) + gamma * RHO * Tz)

    A55 = -np.diag((gamma / (Re * Pr)) * K) @ (Dyy + Dzz) + np.diag(RHO * V - (2 * gamma * K_T * Ty) / (Re * Pr)) @ Dy + \
          np.diag(RHO * W - (2 * gamma * K_T * Tz) / (Re * Pr)) @ Dz - np.diag((gamma - 1) * (RHO_y * V + RHO_z * W)) - \
          np.diag((gamma / (Re * Pr)) * (-K * (alpha ** 2) + K_T * (Tyy + Tzz) + K_TT * ((Ty ** 2) + (Tz ** 2)))) - \
          np.diag(((gamma * (gamma - 1) * (M ** 2) * MU_T) / Re) * ((4 / 3) * ((Vy ** 2) + (Wz ** 2) - Vy * Wz) +
                                                                    (Uy ** 2) + (Uz ** 2) + (Vz ** 2) + (Wy ** 2) + 2 * Vz * Wy) + RHO * U * 1j * alpha)

    # Forming Operator
    A = np.block([[A11, A12, A13, A14, A15],
                  [A21, A22, A23, A24, A25],
                  [A31, A32, A33, A34, A35],
                  [A41, A42, A43, A44, A45],
                  [A51, A52, A53, A54, A55]])

    # B Operator Values
    ZERO = np.zeros((len(U), len(U)))
    B11 = np.diag(np.ones(len(U)) * 1j)
    B22 = np.diag(RHO * 1j)
    B33 = np.diag(RHO * 1j)
    B44 = np.diag(RHO * 1j)
    B51 = np.diag((1 - gamma) * T * 1j)
    B55 = np.diag(RHO * 1j)

    # Forming Operator
    B = np.block([[B11, ZERO, ZERO, ZERO, ZERO],
                  [ZERO, B22, ZERO, ZERO, ZERO],
                  [ZERO, ZERO, B33, ZERO, ZERO],
                  [ZERO, ZERO, ZERO, B44, ZERO],
                  [B51, ZERO, ZERO, ZERO, B55]])

    return [A, B]


########################################   Diagonal Version   #####################################
def BiOpDiag(RHO, U, V, W, T, Dy, Dyy, Dz, Dzz, Re, M, alpha, T_ref):
    # Reshaping 2D arrays to 1D vectors
    order = "C"
    RHO = np.reshape(RHO, (len(RHO[0, :]) * len(RHO[:, 0])), order=order)
    U   = np.reshape(U, (len(U[0, :]) * len(U[:, 0])), order=order)
    V   = np.reshape(V, (len(V[0, :]) * len(V[:, 0])), order=order)
    W   = np.reshape(W, (len(W[0, :]) * len(W[:, 0])), order=order)
    T   = np.reshape(T, (len(T[0, :]) * len(T[:, 0])), order=order)

    # Cross Derivative
    Dyz = Dy @ Dz

    # Baseflow - U
    Uy = Dy @ U
    Uyy = Dyy @ U
    Uz = Dz @ U
    Uzz = Dzz @ U

    # Baseflow - V
    Vy = Dy @ V
    Vyy = Dyy @ V
    Vz = Dz @ V
    Vzz = Dzz @ V
    Vyz = Dyz @ V

    # Baseflow - W
    Wy = Dy @ W
    Wyy = Dyy @ W
    Wz = Dz @ W
    Wzz = Dzz @ W
    Wyz = Dyz @ W

    # Baseflow - T
    Ty = Dy @ T
    Tyy = Dyy @ T
    Tz = Dz @ T
    Tzz = Dzz @ T

    # Constants
    Pr = 0.72
    gamma = 1.4
    CS = 110.4  # Air
    # CS = 107  # N2

    # Sutherland's Formula and its derivatives (Non-dimensional)
    CSa = CS / T_ref
    MU = (T ** (3 / 2)) * (1 + CSa) / (T + CSa)
    MU_T = (1 + CSa) * (3 * np.sqrt(T) / (2 * (T + CSa)) - (T ** (3 / 2)) / ((T + CSa) ** 2))
    MU_TT = (1 + CSa) * (3 / (4 * np.sqrt(T) * (T + CSa)) - 3 * np.sqrt(T) / ((T + CSa) ** 2) + 2 * (T ** (3 / 2)) / ((T + CSa) ** 3))

    # Baseflow Mu
    MU_y = MU_T * Ty
    MU_z = MU_T * Tz

    # Density
    RHO_y = Dy @ RHO
    RHO_z = Dz @ RHO

    # Temperature Conduction Coefficient = Mu
    K = MU
    K_T = MU_T
    K_TT = MU_TT

    # BiGlobal Linear Operator Values
    # RHO terms
    f11 = Vy + Wz + 1j * alpha * U
    f11_y = V
    f11_z = W
    A11 = ssd(f11) + ssd(f11_y) @ Dy + ssd(f11_z) @ Dz
    A11 = ssd(1 / (1j * np.ones(len(RHO)))) @ A11

    f12 = RHO * 1j * alpha
    A12 = ssd(f12)
    A12 = ssd(1 / (1j * np.ones(len(RHO)))) @ A12

    f13 = RHO_y
    f13_y = RHO
    A13 = ssd(f13) + ssd(f13_y) @ Dy
    A13 = ssd(1 / (1j * np.ones(len(RHO)))) @ A13

    f14 = RHO_z
    f14_z = RHO
    A14 = ssd(f14) + ssd(f14_z) @ Dz
    A14 = ssd(1 / (1j * np.ones(len(RHO)))) @ A14

    A15 = ssd(1j * np.ones(len(RHO))) * 0

    # U - velocity terms
    f21 = V * Uy + W * Uz + T * 1j * alpha / (gamma * (M ** 2))
    A21 = ssd(f21)
    A21 = ssd(1 / (1j * RHO)) @ A21

    f22 = RHO * U * 1j * alpha + (4 / 3) * (MU * (alpha ** 2) / Re)
    f22_z = (RHO * W) - (MU_z / Re)
    f22_y = (RHO * V) - (MU_y / Re)
    f22_zz = -(MU / Re)
    f22_yy = -(MU / Re)
    A22 = ssd(f22) + ssd(f22_z) @ Dz + ssd(f22_y) @ Dy + ssd(f22_zz) @ Dzz + ssd(f22_yy) @ Dyy
    A22 = ssd(1 / (1j * RHO)) @ A22

    f23 = RHO * Uy - (MU_y * 1j * alpha / Re)
    f23_y = -(MU * 1j * alpha / (3 * Re))
    A23 = ssd(f23) + ssd(f23_y) @ Dy
    A23 = ssd(1 / (1j * RHO)) @ A23

    f24 = RHO * Uz - (MU_z * 1j * alpha / Re)
    f24_z = -(MU * 1j * alpha / (3 * Re))
    A24 = 0*ssd(f24) + ssd(f24_z) @ Dz
    A24 = ssd(1 / (1j * RHO)) @ A24

    f25 = RHO * 1j * alpha / (gamma * (M ** 2)) - (MU_T / Re) * (Uyy + Uzz - (2 / 3) * (Vy + Wz) * 1j * alpha) - (MU_TT / Re) * (Uy * Ty + Uz * Tz)
    f25_y = -(MU_T * Uy / Re)
    f25_z = -(MU_T * Uz / Re)
    A25 = ssd(f25) + ssd(f25_y) @ Dy + ssd(f25_z) @ Dz
    A25 = ssd(1 / (1j * RHO)) @ A25

    # V - velocity terms
    f31 = V * Vy + W * Vz + Ty / (gamma * (M ** 2))
    f31_y = T / (gamma * (M ** 2))
    A31 = ssd(f31) + ssd(f31_y) @ Dy
    A31 = ssd(1 / (1j * RHO)) @ A31

    f32 = (2 / 3) * (MU_y * 1j * alpha / Re)
    f32_y = -MU * 1j * alpha / (3 * Re)
    A32 = ssd(f32) + ssd(f32_y) @ Dy
    A32 = ssd(1 / (1j * RHO)) @ A32

    f33 = RHO * (U * 1j * alpha + Vy) + MU * (alpha ** 2) / Re
    f33_y = RHO * V - (4 / 3) * (MU_y / Re)
    f33_z = RHO * W - (MU_z / Re)
    f33_yy = -(MU / Re) * (4 / 3)
    f33_zz = -(MU / Re)
    A33 = ssd(f33) + ssd(f33_z) @ Dz + ssd(f33_y) @ Dy + ssd(f33_zz) @ Dzz + ssd(f33_yy) @ Dyy
    A33 = ssd(1 / (1j * RHO)) @ A33

    f34 = RHO * Vz
    f34_y = -MU_z / Re
    f34_z = (2 / 3) * (MU_y / Re)
    f34_yz = -MU / (3 * Re)
    A34 = ssd(f34) + ssd(f34_y) @ Dy + ssd(f34_z) @ Dz + ssd(f34_yz) @ Dyz
    A34 = ssd(1 / (1j * RHO)) @ A34

    f35 = RHO_y / (gamma * (M ** 2)) - (MU_TT / Re) * ((2 / 3) * (2 * Vy - Wz) * Ty + (Vz + Wy) * Tz) - (MU_T / Re) * ((4 / 3) * Vyy + Vzz + Wyz / 3 + Uy * 1j * alpha)
    f35_y = RHO / (gamma * (M ** 2)) - (MU_T / Re) * (2 / 3) * (2 * Vy - Wz)
    f35_z = -(MU_T / Re) * (Vz + Wy)
    A35 = ssd(f35) + ssd(f35_y) @ Dy + ssd(f35_z) @ Dz
    A35 = ssd(1 / (1j * RHO)) @ A35

    # W - velocity terms
    f41 = V * Wy + W * Wz + Tz / (gamma * (M ** 2))
    f41_z = T / (gamma * (M ** 2))
    A41 = ssd(f41) + ssd(f41_z) @ Dz
    A41 = ssd(1 / (1j * RHO)) @ A41

    f42 = (2 / 3) * (MU_z / Re) * 1j * alpha
    f42_z = -(MU / (3 * Re)) * 1j * alpha
    A42 = ssd(f42) + ssd(f42_z) @ Dz
    A42 = ssd(1 / (1j * RHO)) @ A42

    f43 = RHO * Wy
    f43_y = (2 / 3) * (MU_z / Re)
    f43_z = -(MU_y / Re)
    f43_yz = -(MU / (3 * Re))
    A43 = ssd(f43) + ssd(f43_y) @ Dy + ssd(f43_z) @ Dz + ssd(f43_yz) @ Dyz
    A43 = ssd(1 / (1j * RHO)) @ A43

    f44 = RHO * (U * 1j * alpha + Wz) + (MU * (alpha ** 2) / Re)
    f44_y = RHO * V - (MU_y / Re)
    f44_z = RHO * W - (4 / 3) * (MU_z / Re)
    f44_yy = -(MU / Re)
    f44_zz = -(MU / Re) * (4 / 3)
    A44 = ssd(f44) + ssd(f44_y) @ Dy + ssd(f44_z) @ Dz + ssd(f44_yy) @ Dyy + ssd(f44_zz) @ Dzz
    A44 = ssd(1 / (1j * RHO)) @ A44

    f45 = RHO_z / (gamma * (M ** 2)) - (MU_TT / Re) * ((Wy + Vz) * Ty + (2 / 3) * (2 * Wz - Vy) * Tz) - (MU_T / Re) * ((4 / 3) * Wzz + Wyy + Vyz / 3 + Uz * 1j * alpha)
    f45_y = -(MU_T / Re) * (Wy + Vz)
    f45_z = RHO / (gamma * (M ** 2)) - (MU_T / Re) * (2 / 3) * (2 * Wz - Vy)
    A45 = ssd(f45) + ssd(f45_y) @ Dy + ssd(f45_z) @ Dz
    A45 = ssd(1 / (1j * RHO)) @ A45

    # T - terms
    f51 = (gamma - 1) * (Vy * T + Wz * T) + (V * Ty + W * Tz)
    A51 = ssd(f51)
    A51 = ssd(1 / (1j * RHO)) @ A51

    f52 = -(2 * gamma * (gamma - 1) * (M ** 2) * MU / Re) * (-(2 / 3) * (Vy + Wz) * 1j * alpha) + (gamma - 1) * RHO * T * 1j * alpha
    f52_y = -(2 * gamma * (gamma - 1) * (M ** 2) * MU / Re) * Uy
    f52_z = -(2 * gamma * (gamma - 1) * (M ** 2) * MU / Re) * Uz
    A52 = ssd(f52) + ssd(f52_y) @ Dy + ssd(f52_z) @ Dz
    A52 = ssd(1 / (1j * RHO)) @ A52

    f53 = -(2 * gamma * (gamma - 1) * (M ** 2) * MU / Re) * Uy * 1j * alpha + RHO * Ty
    f53_y = -(2 * gamma * (gamma - 1) * (M ** 2) * MU / Re) * (2 / 3) * (2 * Vy - Wz) + (gamma - 1) * RHO * T
    f53_z = -(2 * gamma * (gamma - 1) * (M ** 2) * MU / Re) * (Vz + Wy)
    A53 = ssd(f53) + ssd(f53_y) @ Dy + ssd(f53_z) @ Dz
    A53 = ssd(1 / (1j * RHO)) @ A53

    f54 = -(2 * gamma * (gamma - 1) * (M ** 2) * MU / Re) * Uz * 1j * alpha + RHO * Tz
    f54_y = -(2 * gamma * (gamma - 1) * (M ** 2) * MU / Re) * (Vz + Wy)
    f54_z = -(2 * gamma * (gamma - 1) * (M ** 2) * MU / Re) * (2 / 3) * (2 * Wz - Vy) + (gamma - 1) * RHO * T
    A54 = ssd(f54) + ssd(f54_y) @ Dy + ssd(f54_z) @ Dz
    A54 = ssd(1 / (1j * RHO)) @ A54

    f55 = (gamma - 1) * (RHO * Vy + RHO * Wz) - (gamma / (Re * Pr)) * (-K * (alpha ** 2) + K_T * (Tyy + Tzz) + K_TT * ((Ty ** 2) + (Tz ** 2))) - \
          (gamma * (gamma - 1) * (M ** 2) * MU_T / Re) * ((4 / 3) * ((Vy ** 2) + (Wz ** 2) - Vy * Wz) + (Uy ** 2) + (Uz ** 2) + (Vz ** 2) + (Wy ** 2) + 2 * Vz * Wy) + RHO * U * 1j * alpha
    f55_y = RHO * V - gamma * 2 * K_T * Ty / (Re * Pr)
    f55_z = RHO * W - gamma * 2 * K_T * Tz / (Re * Pr)
    f55_yy = -gamma * K / (Re * Pr)
    f55_zz = -gamma * K / (Re * Pr)
    A55 = ssd(f55) + ssd(f55_z) @ Dz + ssd(f55_y) @ Dy + ssd(f55_zz) @ Dzz + ssd(f55_yy) @ Dyy
    A55 = ssd(1 / (1j * RHO)) @ A55

    # Forming Operator
    A = bmat([[A11, A12, A13, A14, A15],
              [A21, A22, A23, A24, A25],
              [A31, A32, A33, A34, A35],
              [A41, A42, A43, A44, A45],
              [A51, A52, A53, A54, A55]], format="lil")

    # # ZE = A15
    # tst = 1
    # A = bmat([[A11, A14, A13, A12, A15],
    #           [A41, A44, A43, A42, A45],
    #           [A31, A34, A33, A32, A35],
    #           [A21, A24, A23, A22, A25],
    #           [A51, A54, A53, A52, A55]], format="lil")

    return A


########################################   Diagonal Version   #####################################
def BiOpDiagDense(U, V, W, T, Dy, Dyy, Dz, Dzz, Re, M, alpha):
    # Reshaping 2D arrays to 1D vectors
    order = "C"
    U = np.reshape(U, (len(U[0, :]) * len(U[:, 0])), order=order)
    V = np.reshape(V, (len(V[0, :]) * len(V[:, 0])), order=order)
    W = np.reshape(W, (len(W[0, :]) * len(W[:, 0])), order=order)
    T = np.reshape(T, (len(T[0, :]) * len(T[:, 0])), order=order)

    # Cross Derivative
    Dyz = Dy @ Dz

    # Baseflow - U
    Uy = Dy @ U
    Uyy = Dyy @ U
    Uz = Dz @ U
    Uzz = Dzz @ U

    # Baseflow - V
    Vy = Dy @ V
    Vyy = Dyy @ V
    Vz = Dz @ V
    Vzz = Dzz @ V
    Vyz = Dyz @ V

    # Baseflow - W
    Wy = Dy @ W
    Wyy = Dyy @ W
    Wz = Dz @ W
    Wzz = Dzz @ W
    Wyz = Dyz @ W

    # Baseflow - T
    Ty = Dy @ T
    Tyy = Dyy @ T
    Tz = Dz @ T
    Tzz = Dzz @ T

    # Constants
    Pr = 0.72
    gamma = 1.4
    T_ref = 273.15
    CS = 110.4

    # Sutherland's Formula and its derivatives
    CSa = CS / T_ref
    MU = (T ** (3 / 2)) * (1 + CSa) / (T + CSa)
    MU_T = (1 + CSa) * (3 * np.sqrt(T) / (2 * (T + CSa)) - (T ** (3 / 2)) / ((T + CSa) ** 2))
    MU_TT = (1 + CSa) * (3 / (4 * np.sqrt(T) * (T + CSa)) - 3 * np.sqrt(T) / ((T + CSa) ** 2) + 2 * (T ** (3 / 2)) / ((T + CSa) ** 3))

    # Baseflow Mu
    MU_y = MU_T * Ty
    MU_z = MU_T * Tz

    # Density (Chapman Rubesin)
    RHO = 1 / T
    RHO_y = Dy @ RHO
    RHO_z = Dz @ RHO

    # Temperature Conduction Coefficient = Mu
    K = MU
    K_T = MU_T
    K_TT = MU_TT

    # Baseflow Mu
    K_y = K_T * Ty
    K_z = K_T * Tz

    # BiGlobal Linear Operator Values
    # RHO terms
    f11 = Vy + Wz + 1j * alpha * U
    f11_y = V
    f11_z = W
    A11 = np.diag(f11) + np.diag(f11_y) @ Dy + np.diag(f11_z) @ Dz
    A11 = np.diag(1 / (1j * np.ones(len(RHO)))) @ A11

    f12 = RHO * 1j * alpha
    A12 = np.diag(f12)
    A12 = np.diag(1 / (1j * np.ones(len(RHO)))) @ A12

    f13 = RHO_y
    f13_y = RHO
    A13 = np.diag(f13) + np.diag(f13_y) @ Dy
    A13 = np.diag(1 / (1j * np.ones(len(RHO)))) @ A13

    f14 = RHO_z
    f14_z = RHO
    A14 = np.diag(f14) + np.diag(f14_z) @ Dz
    A14 = np.diag(1 / (1j * np.ones(len(RHO)))) @ A14

    A15 = np.zeros((len(U), len(U)))

    # U - velocity terms
    f21 = V * Uy + W * Uz + T * 1j * alpha / (gamma * (M ** 2))
    A21 = np.diag(f21)
    A21 = np.diag(1 / (1j * RHO)) @ A21

    f22 = RHO * U * 1j * alpha + (4 / 3) * (MU * (alpha ** 2) / Re)
    f22_y = RHO * V - (MU_y / Re)
    f22_z = RHO * W - (MU_z / Re)
    f22_yy = -(MU / Re)
    f22_zz = -(MU / Re)
    A22 = np.diag(f22) + np.diag(f22_y) @ Dy + np.diag(f22_z) @ Dz + np.diag(f22_yy) @ Dyy + np.diag(f22_zz) @ Dzz
    A22 = np.diag(1 / (1j * RHO)) @ A22

    f23 = RHO * Uy - (MU_y * 1j * alpha / Re)
    f23_y = -(MU * 1j * alpha / (3 * Re))
    A23 = np.diag(f23) + np.diag(f23_y) @ Dy
    A23 = np.diag(1 / (1j * RHO)) @ A23

    f24 = RHO * Uz - (MU_z * 1j * alpha / Re)
    f24_z = -(MU * 1j * alpha / (3 * Re))
    A24 = np.diag(f24) + np.diag(f24_z) @ Dz
    A24 = np.diag(1 / (1j * RHO)) @ A24

    f25 = RHO * 1j * alpha / (gamma * (M ** 2)) - (MU_T / Re) * (Uyy + Uzz - (2 / 3) * (Vy + Wz) * 1j * alpha) - (MU_TT / Re) * (Uy * Ty + Uz * Tz)
    f25_y = -(MU_T * Uy / Re)
    f25_z = -(MU_T * Uz / Re)
    A25 = np.diag(f25) + np.diag(f25_y) @ Dy + np.diag(f25_z) @ Dz
    A25 = np.diag(1 / (1j * RHO)) @ A25

    # V - velocity terms
    f31 = V * Vy + W * Vz + Ty / (gamma * (M ** 2))
    f31_y = T / (gamma * (M ** 2))
    A31 = np.diag(f31) + np.diag(f31_y) @ Dy
    A31 = np.diag(1 / (1j * RHO)) @ A31

    f32 = (2 / 3) * MU_y * 1j * alpha / Re
    f32_y = -MU * 1j * alpha / (3 * Re)
    A32 = np.diag(f32) + np.diag(f32_y) @ Dy
    A32 = np.diag(1 / (1j * RHO)) @ A32

    f33 = RHO * (U * 1j * alpha + Vy) + MU * (alpha ** 2) / Re
    f33_y = RHO * V - (4 / 3) * (MU_y / Re)
    f33_z = RHO * W - (MU_z / Re)
    f33_yy = -(MU / Re) * (4 / 3)
    f33_zz = -(MU / Re)
    A33 = np.diag(f33) + np.diag(f33_y) @ Dy + np.diag(f33_z) @ Dz + np.diag(f33_yy) @ Dyy + np.diag(f33_zz) @ Dzz
    A33 = np.diag(1 / (1j * RHO)) @ A33

    f34 = RHO * Vz
    f34_y = -MU_z / Re
    f34_z = (2 / 3) * (MU_y / Re)
    f34_yz = -MU / (3 * Re)
    A34 = np.diag(f34) + np.diag(f34_y) @ Dy + np.diag(f34_z) @ Dz + np.diag(f34_yz) @ Dyz
    A34 = np.diag(1 / (1j * RHO)) @ A34

    f35 = RHO_y / (gamma * (M ** 2)) - (MU_TT / Re) * ((2 / 3) * (2 * Vy - Wz) * Ty + (Vz + Wy) * Tz) - (MU_T / Re) * ((4 / 3) * Vyy + Vzz + Wyz / 3 + Uy * 1j * alpha)
    f35_y = RHO / (gamma * (M ** 2)) - (MU_T / Re) * (2 / 3) * (2 * Vy - Wz)
    f35_z = -(MU_T / Re) * (Vz + Wy)
    A35 = np.diag(f35) + np.diag(f35_y) @ Dy + np.diag(f35_z) @ Dz
    A35 = np.diag(1 / (1j * RHO)) @ A35

    # W - velocity terms
    f41 = V * Wy + W * Wz + Tz / (gamma * (M ** 2))
    f41_z = T / (gamma * (M ** 2))
    A41 = np.diag(f41) + np.diag(f41_z) @ Dz
    A41 = np.diag(1 / (1j * RHO)) @ A41

    f42 = (2 / 3) * (MU_z / Re) * 1j * alpha
    f42_z = -(MU / (3 * Re)) * 1j * alpha
    A42 = np.diag(f42) + np.diag(f42_z) @ Dz
    A42 = np.diag(1 / (1j * RHO)) @ A42

    f43 = RHO * Wy
    f43_y = (2 / 3) * (MU_z / Re)
    f43_z = -(MU_y / Re)
    f43_yz = -(MU / (3 * Re))
    A43 = np.diag(f43) + np.diag(f43_y) @ Dy + np.diag(f43_z) @ Dz + np.diag(f43_yz) @ Dyz
    A43 = np.diag(1 / (1j * RHO)) @ A43

    f44 = RHO * (U * 1j * alpha + Wz) + (MU * (alpha ** 2) / Re)
    f44_y = RHO * V - (MU_y / Re)
    f44_z = RHO * W - (4 / 3) * (MU_z / Re)
    f44_yy = -(MU / Re)
    f44_zz = -(MU / Re) * (4 / 3)
    A44 = np.diag(f44) + np.diag(f44_y) @ Dy + np.diag(f44_z) @ Dz + np.diag(f44_yy) @ Dyy + np.diag(f44_zz) @ Dzz
    A44 = np.diag(1 / (1j * RHO)) @ A44

    f45 = RHO_z / (gamma * (M ** 2)) - (MU_TT / Re) * ((Wy + Vz) * Ty + (2 / 3) * (2 * Wz - Vy) * Tz) - (MU_T / Re) * ((4 / 3) * Wzz + Wyy + Vyz / 3 + Uz * 1j * alpha)
    f45_y = -(MU_T / Re) * (Wy + Vz)
    f45_z = RHO / (gamma * (M ** 2)) - (MU_T / Re) * (2 / 3) * (2 * Wz - Vy)
    A45 = np.diag(f45) + np.diag(f45_y) @ Dy + np.diag(f45_z) @ Dz
    A45 = np.diag(1 / (1j * RHO)) @ A45

    # T - terms
    f51 = (gamma - 1) * (Vy * T + Wz * T) + (V * Ty + W * Tz)
    A51 = np.diag(f51)
    A51 = np.diag(1 / (1j * RHO)) @ A51

    f52 = -(2 * gamma * (gamma - 1) * (M ** 2) * MU / Re) * (-(2 / 3) * (Vy + Wz) * 1j * alpha) + (gamma - 1) * RHO * T * 1j * alpha
    f52_y = -(2 * gamma * (gamma - 1) * (M ** 2) * MU / Re) * Uy
    f52_z = -(2 * gamma * (gamma - 1) * (M ** 2) * MU / Re) * Uz
    A52 = np.diag(f52) + np.diag(f52_y) @ Dy + np.diag(f52_z) @ Dz
    A52 = np.diag(1 / (1j * RHO)) @ A52

    f53 = -(2 * gamma * (gamma - 1) * (M ** 2) * MU / Re) * Uy * 1j * alpha + RHO * Ty
    f53_y = -(2 * gamma * (gamma - 1) * (M ** 2) * MU / Re) * (2 / 3) * (2 * Vy - Wz) + (gamma - 1) * RHO * T
    f53_z = -(2 * gamma * (gamma - 1) * (M ** 2) * MU / Re) * (Vz + Wy)
    A53 = np.diag(f53) + np.diag(f53_y) @ Dy + np.diag(f53_z) @ Dz
    A53 = np.diag(1 / (1j * RHO)) @ A53

    f54 = -(2 * gamma * (gamma - 1) * (M ** 2) * MU / Re) * Uz * 1j * alpha + RHO * Tz
    f54_y = -(2 * gamma * (gamma - 1) * (M ** 2) * MU / Re) * (Vz + Wy)
    f54_z = -(2 * gamma * (gamma - 1) * (M ** 2) * MU / Re) * (2 / 3) * (2 * Wz - Vy) + (gamma - 1) * RHO * T
    A54 = np.diag(f54) + np.diag(f54_y) @ Dy + np.diag(f54_z) @ Dz
    A54 = np.diag(1 / (1j * RHO)) @ A54

    f55 = (gamma - 1) * (RHO * Vy + RHO * Wz) - (gamma / (Re * Pr)) * (-K * (alpha ** 2) + K_T * (Tyy + Tzz) + K_TT * ((Ty ** 2) + (Tz ** 2))) - \
          (gamma * (gamma - 1) * (M ** 2) * MU_T / Re) * ((4 / 3) * ((Vy ** 2) + (Wz ** 2) - Vy * Wz) + (Uy ** 2) + (Uz ** 2) + (Vz ** 2) + (Wy ** 2) + 2 * Vz * Wy) + RHO * U * 1j * alpha
    f55_y = RHO * V - gamma * 2 * K_T * Ty / (Re * Pr)
    f55_z = RHO * W - gamma * 2 * K_T * Tz / (Re * Pr)
    f55_yy = -gamma * K / (Re * Pr)
    f55_zz = -gamma * K / (Re * Pr)
    A55 = np.diag(f55) + np.diag(f55_y) @ Dy + np.diag(f55_z) @ Dz + np.diag(f55_yy) @ Dyy + np.diag(f55_zz) @ Dzz
    A55 = np.diag(1 / (1j * RHO)) @ A55

    # Forming Operator
    sc = 0
    A = np.block([[A11, A12, A13, A14, A15],
              [A21, A22, A23, A24, A25],
              [A31, A32, A33, A34, A35],
              [A41, A42, A43, A44, A45],
              [A51 * sc, A52 * sc, A53 * sc, A54 * sc, A55 * sc]])

    return A

def BiOpDiagGen(RHO, U, V, W, T, Dy, Dyy, Dx, Dxx, Re, Ma, m, T_ref, axis, ETA, Y):

    eta = copy.copy(ETA)
    y   = copy.copy(Y)
    y_wall = y[0,:]
    eta2D  = RHO * 0

    if axis=="Planar":
        h3 = RHO * 0 + 1
    else:
        for i in range(len(Y[:,0])):
            y[i, :] = y_wall

        for i in range(len(Y[0,:])):
            eta2D[:,i] = eta

        h3 = y + eta2D


    # Reshaping 2D arrays to 1D vectors
    order  = "C"
    h3     = np.reshape(h3, (len(h3[0, :]) * len(h3[:, 0])), order=order)
    RHO    = np.reshape(RHO, (len(RHO[0, :]) * len(RHO[:, 0])), order=order)
    U      = np.reshape(U, (len(U[0, :]) * len(U[:, 0])), order=order)
    V      = np.reshape(V, (len(V[0, :]) * len(V[:, 0])), order=order)
    W      = np.reshape(W, (len(W[0, :]) * len(W[:, 0])), order=order)
    T      = np.reshape(T, (len(T[0, :]) * len(T[:, 0])), order=order)

    # Cross Derivative
    Dyx    = Dy @ Dx

    # Curvature Term h3
    if axis == "Planar":
        h3x    = h3*0
        h3xx   = h3*0
        h3y    = h3*0
        h3yy   = h3*0
        h3xy   = h3*0
    else:
        h3x    = Dx @ h3
        h3xx   = Dxx @ h3
        h3y    = Dy @ h3
        h3yy   = Dyy @ h3
        h3xy   = Dyx @ h3


    # Baseflow - U
    Uy     = Dy @ U
    Uyy    = Dyy @ U
    Ux     = Dx @ U
    Uxx    = Dxx @ U
    Uyx    = Dyx @ U

    # Baseflow - V
    Vy     = Dy @ V
    Vyy    = Dyy @ V
    Vx     = Dx @ V
    Vxx    = Dxx @ V
    Vyx    = Dyx @ V

    # Baseflow - W
    Wy     = Dy @ W
    Wyy    = Dyy @ W
    Wx     = Dx @ W
    Wxx    = Dxx @ W

    # Baseflow - T
    Tx     = Dx @ T
    Ty     = Dy @ T
    Txx    = Dxx @ T
    Tyy    = Dyy @ T

    # Constants
    Pr     = 0.72
    gamma  = 1.4
    CS     = 110.4  # Air
    # CS    = 107  # N2

    # Sutherland's Formula and its derivatives (Non-dimensional)
    CSa    = CS / T_ref
    MU     = (T ** (3 / 2)) * (1 + CSa) / (T + CSa)
    MU_T   = (1 + CSa) * (3 * np.sqrt(T) / (2 * (T + CSa)) - (T ** (3 / 2)) / ((T + CSa) ** 2))
    MU_TT  = (1 + CSa) * (3 / (4 * np.sqrt(T) * (T + CSa)) - 3 * np.sqrt(T) / ((T + CSa) ** 2) + 2 * (T ** (3 / 2)) / (
                (T + CSa) ** 3))

    # Baseflow Mu
    MU_y   = MU_T * Ty
    MU_x   = MU_T * Tx

    # Density
    RHO_y  = Dy @ RHO
    RHO_x  = Dx @ RHO

    # Temperature Conduction Coefficient = Mu
    K = MU
    K_T = MU_T
    K_TT = MU_TT

    # Baseflow Mu
    K_y = MU_T * Ty
    K_x = MU_T * Tx

    # BiGlobal Linear Operator Values
    # RHO terms
    f11    = Ux+Vy+(1j*m*W/h3)+(U*h3x/h3)+(V*h3y/h3)
    f11_x  = U
    f11_y  = V
    A11    = ssd(f11) + ssd(f11_y) @ Dy + ssd(f11_x) @ Dx
    A11    = ssd(1 / (1j * np.ones(len(RHO)))) @ A11

    f12    = RHO_x+(RHO*h3x/h3)
    f12_x  = RHO
    A12    = ssd(f12) + ssd(f12_x) @ Dx
    A12    = ssd(1 / (1j * np.ones(len(RHO)))) @ A12

    f13    = RHO_y+(RHO*h3y/h3)
    f13_y  = RHO
    A13    = ssd(f13) + ssd(f13_y) @ Dy
    A13    = ssd(1 / (1j * np.ones(len(RHO)))) @ A13

    f14    = (1j*m*RHO/h3)
    A14    = ssd(f14)
    A14    = ssd(1 / (1j * np.ones(len(RHO)))) @ A14

    A15    = ssd(1j * np.ones(len(RHO))) * 0

    # U - velocity terms
    f21    = V * Uy + U * Ux + Tx / (gamma * (Ma ** 2)) - W**2*(h3x/h3)
    f21_x  = T / (gamma * (Ma ** 2))
    A21    = ssd(f21) + ssd(f21_x) @ Dx
    A21    = ssd(1 / (1j * RHO)) @ A21

    f22    = ((2/3)*(h3x*MU_x/(h3*Re)) + (RHO*Ux) + (m**2*MU/(h3**2*Re)) + (1j*m*RHO*W/h3) - (2/3)*MU*h3x**2/(h3**2*Re) + (2/3)*MU*h3xx/(h3*Re))
    f22_x  = -(4/3)*(MU_x/Re)-(4/3)*MU*h3x/(h3*Re)+(RHO*U)
    f22_y  = -(MU_y/Re)-(MU*h3y/(h3 * Re))+(RHO*V)
    f22_xx = -(4/3)*(MU/Re)
    f22_yy = -(MU/Re)
    A22    = ssd(f22) + ssd(f22_y) @ Dy + ssd(f22_x) @ Dx + ssd(f22_yy) @ Dyy + ssd(f22_xx) @ Dxx
    A22    = ssd(1 / (1j * RHO)) @ A22

    f23    = (2/3)*(h3y*MU_x/(h3*Re))+(RHO*Uy)-(2/3)*(MU*h3x*h3y/(h3**2*Re))+(2/3)*MU*h3xy/(h3*Re)
    f23_x  = -(MU_y/Re)-(MU*h3y/(3*h3*Re))
    f23_y  = (2/3)*(MU_x/Re)
    f23_xy = -(MU/(3*Re))
    A23    = ssd(f23) + ssd(f23_y) @ Dy + ssd(f23_x) @ Dx + ssd(f23_xy) @ Dyx
    A23    = ssd(1 / (1j * RHO)) @ A23

    f24    = (2 / 3) * (MU_x /(Re*h3)) * 1j * m -(2/3)*(1j*m*MU*h3x/(h3**2*Re))-(2*RHO*W*h3x/h3)
    f24_x  = -(MU / (3 * Re * h3)) * 1j * m
    A24    = ssd(f24) + ssd(f24_x) @ Dx
    A24    = ssd(1 / (1j * RHO)) @ A24

    f25 = (RHO_x / (gamma * (Ma ** 2)) - (MU_TT / Re) * ((Uy + Vx) * Ty + (2 / 3) * (2 * Ux - Vy) * Tx) - (MU_T / Re) * ((4 / 3) * Uxx + Uyy + Vyx / 3 + Wx * 1j * m / h3)
           + (2/3)*(MU_TT*Tx/(Re*h3))*(U*h3x+V*h3y) + (2/3)*(U*MU_T/Re)*(-h3x**2/(h3**2)+h3xx/h3) + (2/3)*(V*MU_T/Re)*(-h3x*h3y/(h3**2)+h3xy/h3)
           -(4/3)*(MU_T*Ux/Re)*(h3x/h3)-(MU_T*Uy/Re)*(h3y/h3)-(MU_T*Vx/(3*Re))*(h3y/h3))
    f25_x = RHO / (gamma * (Ma ** 2)) - (MU_T / Re) * (2 / 3) * (2 * Ux - Vy) + (2/3)*(MU_T/Re)*(U*(h3x/h3) + V*(h3y/h3))
    f25_y = -(MU_T / Re) * (Uy + Vx)

    A25    = ssd(f25) + ssd(f25_y) @ Dy + ssd(f25_x) @ Dx
    A25    = ssd(1 / (1j * RHO)) @ A25

    # V - velocity terms
    f31    = V * Vy + U * Vx + Ty / (gamma * (Ma ** 2)) - W**2*(h3y/h3)
    f31_y  = T / (gamma * (Ma ** 2))
    A31    = ssd(f31) + ssd(f31_y) @ Dy
    A31    = ssd(1 / (1j * RHO)) @ A31

    f32    = (2/3)*(h3x*MU_y/(h3*Re))+(RHO*Vx)-(2/3)*(MU*h3x*h3y/(h3**2*Re))
    f32_x  = (2/3)*(MU_y/Re)
    f32_y  = -(MU_x/Re)-(MU*h3x/(h3*Re))
    f32_xy = -(MU/(3*Re))
    A32    = ssd(f32) + ssd(f32_y) @ Dy + ssd(f32_x) @ Dx + ssd(f32_xy) @ Dyx
    A32    = ssd(1 / (1j * RHO)) @ A32

    f33    = (2/3)*(h3y*MU_y/(h3*Re)) + (RHO*Vy) + (m**2*MU/(h3**2*Re)) + (1j*m*RHO*W/h3) - (2/3)*(MU*h3y**2/(h3**2*Re)) + (2/3)*(MU*h3yy/(h3*Re))
    f33_x  = -(MU_x/Re) - (MU*h3x/(h3*Re)) + (RHO*U)
    f33_y  = -(4/3)*(MU_y/Re)-(4/3)*(MU*h3y/(h3*Re))+(RHO*V)
    f33_xx = -(MU/Re)
    f33_yy = -(4/3)*(MU/Re)
    A33    = ssd(f33) + ssd(f33_x) @ Dx + ssd(f33_y) @ Dy + ssd(f33_xx) @ Dxx + ssd(f33_yy) @ Dyy
    A33    = ssd(1 / (1j * RHO)) @ A33

    f34    = (2 / 3) * (MU_y * 1j * m / (Re * h3)) -(2/3)*(1j*m*MU/Re)*(h3y/(h3**2))-(2*RHO*W*h3y/h3)
    f34_y  = -MU * 1j * m / (3 * Re * h3)
    A34    = ssd(f34) + ssd(f34_y) @ Dy
    A34    = ssd(1 / (1j * RHO)) @ A34

    f35    = (RHO_y / (gamma * (Ma ** 2)) - (MU_TT / Re) * ((2 / 3) * (2 * Vy - Ux) * Ty + (Vx + Uy) * Tx) - (MU_T / Re) * ((4 / 3) * Vyy + Vxx + Uyx / 3 + Wy * 1j * m)
              +(2/3)*(MU_TT/Re)*Ty*(U*(h3x/h3)+V*(h3y/h3))-(MU_T/Re)*Uy*(h3x/(3*h3))-(MU_T/Re)*Vx*(h3x/h3)
              -(4/3)*h3y*MU_T*Vy/(Re*h3) +(2/3)*(MU_T/Re)*U*(-h3x*h3y/(h3**2)+h3xy/h3)
              + (2/3)*(MU_T/Re)*V*(-h3y**2/(h3**2)+h3yy/h3))
    f35_x  = -(MU_T / Re) * (Vx + Uy)
    f35_y  = RHO / (gamma * (Ma ** 2)) - (MU_T / Re) * (2 / 3) * (2 * Vy - Ux) + (2/3)*(MU_T/Re)*(U*(h3x/h3)+V*(h3y/h3))
    A35    = ssd(f35) + ssd(f35_y) @ Dy + ssd(f35_x) @ Dx
    A35    = ssd(1 / (1j * RHO)) @ A35

    # W - velocity terms
    f41    = (U*Wx)+(V*Wy)+(1j*m*T/(gamma*h3*Ma**2))+(U*W*h3x/h3)+(V*W*h3y/h3)
    A41    = ssd(f41)
    A41    = ssd(1 / (1j * RHO)) @ A41

    f42    = -(1j*m*MU_x/(h3*Re))+(RHO*Wx)+(2/3)*(1j*m*MU*h3x/(h3**2*Re))+(RHO*W*h3x/h3)
    f42_x  = -(1j*m*MU/(3*h3*Re))
    A42    = 0*ssd(f42) + ssd(f42_x) @ Dx
    A42    = ssd(1 / (1j * RHO)) @ A42

    f43    = -(1j*m*MU_y/(h3*Re))+(RHO*Wy)+(2/3)*(1j*m*MU*h3y/(h3**2*Re))+(RHO*W*h3y/h3)
    f43_y  = -(1j*m*MU/(3*h3*Re))
    A43    = ssd(f43) + ssd(f43_y) @ Dy
    A43    = ssd(1 / (1j * RHO)) @ A43

    f44    = (4/3)*(m**2*MU/(h3**2*Re)) + (1j*m*RHO*W/h3) + (RHO*U*h3x/h3) + (RHO*V*h3y/h3)
    f44_x  = -(MU_x / Re) - (MU*h3x / (h3*Re)) + (RHO*U)
    f44_y  = -(MU_y / Re) - (MU*h3y / (h3*Re)) + (RHO*V)
    f44_xx = -(MU / Re)
    f44_yy = -(MU / Re)
    A44    = ssd(f44) + ssd(f44_x) @ Dx + ssd(f44_y) @ Dy + ssd(f44_xx) @ Dxx + ssd(f44_yy) @ Dyy
    A44    = ssd(1 / (1j * RHO)) @ A44

    f45    = ((2/3)*(1j*m*MU_T*Ux/(h3*Re))+(2/3)*(1j*m*MU_T*Vy/(h3*Re))-(h3y*MU_T*Wy/(h3*Re))-(h3x*MU_T*Wx/(h3*Re))
              -(MU_TT*Ty*Wy/Re)-(MU_TT*Tx*Wx/Re)-(MU_T*Wxx/Re)-(MU_T*Wyy/Re)+(1j*m*RHO/(gamma*h3*Ma**2))
              +(2/3)*(1j*m*U*h3x*MU_T/(h3**2*Re))+(2/3)*(1j*m*V*h3y*MU_T/(h3**2*Re)))
    f45_x  = -(MU_T*Wx/Re)
    f45_y  = -(MU_T*Wy/Re)
    A45    = ssd(f45) + ssd(f45_y) @ Dy + ssd(f45_x) @ Dx
    A45    = ssd(1 / (1j * RHO)) @ A45

    # Temperature terms
    f51   = (gamma - 1) * (Vy * T + Ux * T) + (V * Ty + U * Tx) + (gamma*T*U*h3x/h3) - (T*U*h3x/h3) + (gamma*T*V*h3y/h3) - (T*V*h3y/h3)
    A51   = ssd(f51)
    A51   = ssd(1 / (1j * RHO)) @ A51

    f52   = ((-(2 * gamma * (gamma - 1) * (Ma ** 2) * MU / Re) * (Wx/h3) * 1j * m + RHO * Tx
             + (gamma*(gamma-1)*Ma**2*(4/3)*MU/Re)*(Ux*(h3x/h3) + Vy*(h3x/h3)))
             + (gamma*(gamma-1)*Ma**2*(4/3)*MU/Re)*(U*((h3x**2)/(h3**2))+V*((h3x*h3y)/(h3**2)))
             + (gamma-1)*RHO*T*(h3x/h3))
    f52_x = (-(2 * gamma * (gamma - 1) * (Ma ** 2) * MU / Re) * (2 / 3) * (2 * Ux - Vy) + (gamma - 1) * RHO * T
             + (4/3)*(gamma**2*Ma**2*MU*U*h3x/(h3*Re))- (4/3)*(gamma*Ma**2*MU*U*h3x/(h3*Re))
             + (4/3)*(gamma**2*Ma**2*MU*V*h3y/(h3*Re))- (4/3)*(gamma*Ma**2*MU*V*h3y/(h3*Re)))
    f52_y = -(2 * gamma * (gamma - 1) * (Ma ** 2) * MU / Re) * (Vx + Uy)
    A52   = ssd(f52) + ssd(f52_y) @ Dy + ssd(f52_x) @ Dx
    A52   = ssd(1 / (1j * RHO)) @ A52

    f53   = (-(2*1j*gamma**2*m*Ma**2*MU*Wy/(h3*Re))+(2*1j*gamma*m*Ma**2*MU*Wy/(h3*Re))
            +(4/3)*(gamma**2*Ma**2*MU*h3y*Ux/(h3*Re))-(4/3)*(gamma*Ma**2*MU*h3y*Ux/(h3*Re))
            +(4/3)*(gamma**2*Ma**2*MU*h3y*Vy/(h3*Re))-(4/3)*(gamma*Ma**2*MU*h3y*Vy/(h3*Re))+(RHO*Ty)
            +(4/3)*(gamma**2*Ma**2*MU*U*h3x*h3y/(h3**2*Re))-(4/3)*(gamma*Ma**2*MU*U*h3x*h3y/(h3**2*Re))
            +(4/3)*(gamma**2*Ma**2*MU*V*h3y**2/(h3**2*Re))-(4/3)*(gamma*Ma**2*MU*V*h3y**2/(h3**2*Re))
            +(gamma*RHO*T*h3y/h3)-(RHO*T*h3y/h3))

    f53_x = -(2 * gamma * (gamma - 1) * (Ma ** 2) * MU / Re) * (Vx + Uy)

    f53_y = (-(2 * gamma * (gamma - 1) * (Ma ** 2) * MU / Re) * (2 / 3) * (2 * Vy - Ux) + (gamma - 1) * RHO * T
             +(gamma*(gamma-1)*Ma**2*(4/3)*MU/Re)*(U*(h3x/h3)+V*(h3y/h3)))



    A53   = ssd(f53) + ssd(f53_y) @ Dy + ssd(f53_x) @ Dx
    A53   = ssd(1 / (1j * RHO)) @ A53

    f54   = (-(2 * gamma * (gamma - 1) * (Ma ** 2) * MU / Re) * (-(2 / 3) * (Vy/h3 + Ux/h3) * 1j * m) + (gamma - 1) * RHO * T * 1j * m / h3
             + (gamma*(gamma-1)*Ma**2*(4/3)*(1j*m)*MU/Re)*(U*(h3x/(h3**2))+V*(h3y/(h3**2))))
    f54_x = -(2 * gamma * (gamma - 1) * (Ma ** 2) * MU / Re) * Wx
    f54_y = -(2 * gamma * (gamma - 1) * (Ma ** 2) * MU / Re) * Wy
    A54   = ssd(f54) + ssd(f54_x) @ Dx + ssd(f54_y) @ Dy
    A54   = ssd(1 / (1j * RHO)) @ A54

    f55 = ((gamma - 1) * (RHO * Vy + RHO * Ux)

           - (gamma / (Re * Pr)) * (-K * (m ** 2)/h3**2 + K_T * (Tyy + Txx) + K_TT * ((Ty ** 2) + (Tx ** 2)))
           - (gamma * (gamma - 1) * (Ma ** 2) * MU_T / Re) * ((4 / 3) * ((Vy ** 2) + (Ux ** 2) - Vy * Ux) + (Wy ** 2) + (Wx ** 2) + (Vx ** 2) + (Uy ** 2) + 2 * Vx * Uy)
           + RHO * W * 1j * m / h3
           + (gamma-1)*(RHO*U*(h3x/h3)+RHO*V*(h3y/h3))+(gamma*(gamma-1)*Ma**2*(4/3)*MU_T/Re)*(U*V*(h3y*h3x/(h3**2))+U*Vy*(h3x/h3)+V*Vy*(h3y/h3)+U*Ux*(h3x/h3)+V*Ux*(h3y/h3))
           +(gamma*(gamma-1)*Ma**2*(2/3)*MU_T/Re)*(U**2*(h3x**2/(h3**2))+V**2*(h3y**2/(h3**2)))-(gamma/(Pr*Re))*(K_y*(h3y/h3)+K_x*(h3x/h3)))

    f55_x = RHO * U - gamma * 2 * K_T * Tx / (Re * Pr) - (gamma*K/(Pr*Re))*(h3x/h3)
    f55_y = RHO * V - gamma * 2 * K_T * Ty / (Re * Pr) - (gamma*K/(Pr*Re))*(h3y/h3)
    f55_yy = -gamma * K / (Re * Pr)
    f55_xx = -gamma * K / (Re * Pr)
    A55   = ssd(f55) + ssd(f55_x) @ Dx + ssd(f55_y) @ Dy + ssd(f55_xx) @ Dxx + ssd(f55_yy) @ Dyy
    A55   = ssd(1 / (1j * RHO)) @ A55

    # Forming Operator

    # tst = 1
    #
    # A = bmat([[A11*tst, A12*tst, A13*tst, A14*tst, A15*tst],
    #           [A21*tst, A22*tst, A23*tst, A24*tst, A25*tst],
    #           [A31*tst, A32*tst, A33*tst, A34*tst, A35*1],
    #           [A41*tst, A42*tst, A43*tst, A44*tst, A45*tst],
    #           [A51*tst, A52*tst, A53*tst, A54*tst, A55*tst]], format="lil")

    A = bmat([[A11, A12, A13, A14, A15],
              [A21, A22, A23, A24, A25],
              [A31, A32, A33, A34, A35],
              [A41, A42, A43, A44, A45],
              [A51, A52, A53, A54, A55]], format="lil")



    return A





