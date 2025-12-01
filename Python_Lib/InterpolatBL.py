import numpy as np
import scipy


def IntBL(FLOW, y):
    U = np.zeros(len(y))
    V = np.zeros(len(y))
    W = np.zeros(len(y))
    T = np.zeros(len(y))

    STATE = [U, V, W, T]
    for i in range(4):
        # y is longer than length in FLOW -> find closest value of y to the largest coordinate in FLOW
        idx = (np.abs(y - np.max(FLOW[:, 0]))).argmin()
        f = scipy.interpolate.CubicSpline(FLOW[:, 0], FLOW[:, i + 1], bc_type='natural')
        Var_new = f(y[0:(idx + 1)])
        if i == 1:  # i.e. V velocity component
            Straight_Line = np.zeros(y.argmax() - idx)
            for j in range(y.argmax() - idx):
                K = ((Var_new[-1] - Var_new[-2]) / (y[idx] - y[idx - 1]))
                B = Var_new[-1] - (K * y[idx])
                Straight_Line[j] = K * y[idx + j + 1] + B
            STATE[i] = np.block([Var_new, Straight_Line])
        else:
            STATE[i] = np.block([Var_new, np.ones(y.argmax() - idx) * Var_new[-1]])

    return STATE
