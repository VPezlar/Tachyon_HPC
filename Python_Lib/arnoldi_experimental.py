import numpy as np
import scipy


def arnoldi(A, M, sigma, orig_size, size):
    N = orig_size
    n = size  # Hessenberg size

    Q = np.cdouble(np.zeros((N, n)))  # Basis
    H = np.cdouble(np.zeros((N, n)))  # Hessenberg Matrix

    x0 = np.cdouble(np.ones(N))
    x0 = x0 / np.linalg.norm(x0)

    # First Input
    Q[:, 0] = x0

    for i in range(n):
        M_qi = M @ Q[:, i]
        u = scipy.sparse.linalg.spsolve(A - sigma * M, M_qi)
        # Orthogonalization
        for j in range(i + 1):
            H[j, i] = Q[:, j] @ u
            u = u - H[j, i] * Q[:, j]
        if i + 1 < n:
            H[i + 1, i] = np.linalg.norm(u)
            Q[:, i + 1] = u / H[i + 1, i]

    eig_val = np.linalg.eig(H[0:n, :])[0]
    eig_val = (1 / eig_val) + sigma

    return eig_val, H[0:n, :]

