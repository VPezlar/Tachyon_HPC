import numpy as np
import FDq
import copy


def InterOp(USED, UNUSED):
    INTERPOLATION_OPERATOR = np.zeros((UNUSED + 1, USED + 1))
    for i in range(UNUSED + 1):
        for j in range(USED + 1):
            INTERPOLATOR = 0
            if j == 0 or j == USED:
                dj = 2
            else:
                dj = 1
            for k in range(USED + 1):
                if k == 0 or k == USED:
                    dk = 2
                else:
                    dk = 1
                INTERPOLATOR = INTERPOLATOR + (1 / dk) * np.cos(np.pi * k * j / USED) * np.cos(np.pi * k * i / UNUSED)
            INTERPOLATION_OPERATOR[i, j] = (2 / (dj * USED)) * INTERPOLATOR

    return INTERPOLATION_OPERATOR


########################################## Dirichlet BC ###########################################
def Dir_BC_hor_min(N_eq, Disc, Domains, Dimensions, Domain_numbers, Skip):
    Dim = copy.copy(Dimensions)
    if Disc == "Spectral":
        Dim = Dimensions + 1
    elif Disc == "XSpectral":
        for i in range(len(Dimensions[:, 0])):
            Dim[i, 1] = Dimensions[i, 1] + 1
    elif Disc == "YSpectral":
        for i in range(len(Dimensions[:, 0])):
            Dim[i, 0] = Dimensions[i, 0] + 1

    for i in Domain_numbers:
        Arr = Domains[i - 1]
        for k in range(N_eq):
            if k not in Skip[i-1]:
                for l in range(Dim[i - 1, 1]):
                    Arr[l + (Dim[i - 1, 0] * Dim[i - 1, 1]) * k, :] = 0
                    Arr[l + (Dim[i - 1, 0] * Dim[i - 1, 1]) * k, l + (Dim[i - 1, 0] * Dim[i - 1, 1]) * k] = 1
        Domains[i - 1] = Arr
    return Domains


def Dir_BC_hor_max(N_eq, Disc, Domains, Dimensions, Domain_numbers, Skip):
    Dim = copy.copy(Dimensions)
    if Disc == "Spectral":
        Dim = Dimensions + 1
    elif Disc == "XSpectral":
        for i in range(len(Dimensions[:, 0])):
            Dim[i, 1] = Dimensions[i, 1] + 1
    elif Disc == "YSpectral":
        for i in range(len(Dimensions[:, 0])):
            Dim[i, 0] = Dimensions[i, 0] + 1

    for i in Domain_numbers:
        Arr = Domains[i - 1]
        for k in range(N_eq):
            if k not in Skip[i-1]:
                for l in range(Dim[i - 1, 1]):
                    Arr[- 1 - l + (Dim[i - 1, 0] * Dim[i - 1, 1]) * (k + 1), :] = 0
                    Arr[- 1 - l + (Dim[i - 1, 0] * Dim[i - 1, 1]) * (k + 1), -1 - l + (Dim[i - 1, 0] * Dim[i - 1, 1]) * (k + 1)] = 1
        Domains[i - 1] = Arr
    return Domains


def Dir_BC_ver_min(N_eq, Disc, Domains, Dimensions, Domain_numbers, Skip):
    Dim = copy.copy(Dimensions)
    if Disc == "Spectral":
        Dim = Dimensions + 1
    elif Disc == "XSpectral":
        for i in range(len(Dimensions[:, 0])):
            Dim[i, 1] = Dimensions[i, 1] + 1
    elif Disc == "YSpectral":
        for i in range(len(Dimensions[:, 0])):
            Dim[i, 0] = Dimensions[i, 0] + 1

    for i in Domain_numbers:
        Arr = Domains[i - 1]
        for k in range(N_eq):
            if k not in Skip[i-1]:
                for l in range(Dim[i - 1, 0]):
                    Arr[(Dim[i - 1, 0] * Dim[i - 1, 1]) * k + Dim[i - 1, 1] * l, :] = 0
                    Arr[(Dim[i - 1, 0] * Dim[i - 1, 1]) * k + Dim[i - 1, 1] * l,
                        (Dim[i - 1, 0] * Dim[i - 1, 1]) * k + Dim[i - 1, 1] * l] = 1
        Domains[i - 1] = Arr
    return Domains


def Dir_BC_ver_max(N_eq, Disc, Domains, Dimensions, Domain_numbers, Skip):
    Dim = copy.copy(Dimensions)
    if Disc == "Spectral":
        Dim = Dimensions + 1
    elif Disc == "XSpectral":
        for i in range(len(Dimensions[:, 0])):
            Dim[i, 1] = Dimensions[i, 1] + 1
    elif Disc == "YSpectral":
        for i in range(len(Dimensions[:, 0])):
            Dim[i, 0] = Dimensions[i, 0] + 1

    for i in Domain_numbers:
        Arr = Domains[i - 1]
        for k in range(N_eq):
            if k not in Skip[i - 1]:
                for l in range(Dim[i - 1, 0]):
                    Arr[(Dim[i - 1, 0] * Dim[i - 1, 1]) * k + Dim[i - 1, 1] + Dim[i - 1, 1] * l - 1, :] = 0
                    Arr[(Dim[i - 1, 0] * Dim[i - 1, 1]) * k + Dim[i - 1, 1] + Dim[i - 1, 1] * l - 1,
                        (Dim[i - 1, 0] * Dim[i - 1, 1]) * k + Dim[i - 1, 1] + Dim[i - 1, 1] * l - 1] = 1
        Domains[i - 1] = Arr
    return Domains


########################################### Neumann BC ############################################
def Neu_BC_ver_min(A, N_eq, Disc, Dimensions, Domain_numbers, D, Skip):
    # Neumann BC Vertical (Experimental)
    Dim = copy.copy(Dimensions)
    if Disc == "Spectral":
        Dim = Dimensions + 1
    elif Disc == "XSpectral":
        for i in range(len(Dimensions[:, 0])):
            Dim[i, 1] = Dimensions[i, 1] + 1
    elif Disc == "YSpectral":
        for i in range(len(Dimensions[:, 0])):
            Dim[i, 0] = Dimensions[i, 0] + 1

    for i in Domain_numbers:
        DX = D[i - 1]

        offset = 0
        for j in range(N_eq):
            for k in range(i - 1):
                if i > 1:
                    offset = offset + (Dim[k, 0] * Dim[k, 1])

        for j in range(N_eq):
            if j not in Skip[i - 1]:
                for k in range(Dim[i - 1, 0]):
                    start = Dim[i - 1, 1] * Dim[i - 1, 0] * j
                    end   = Dim[i - 1, 1] * Dim[i - 1, 0] * (j + 1)
                    ymove = Dim[i - 1, 1] * k
                    A[offset + start + ymove, :] = 0
                    A[offset + start + ymove, (offset + start):(offset + end)] = DX[ymove, :]

    return A


def Neu_BC_ver_max(A, N_eq, Disc, Dimensions, Domain_numbers, D, Skip):
    # Neumann BC Vertical (Experimental)
    Dim = copy.copy(Dimensions)
    if Disc == "Spectral":
        Dim = Dimensions + 1
    elif Disc == "XSpectral":
        for i in range(len(Dimensions[:, 0])):
            Dim[i, 1] = Dimensions[i, 1] + 1
    elif Disc == "YSpectral":
        for i in range(len(Dimensions[:, 0])):
            Dim[i, 0] = Dimensions[i, 0] + 1

    for i in Domain_numbers:
        DX = D[i - 1]

        offset = 0
        for j in range(N_eq):
            for k in range(i - 1):
                if i > 1:
                    offset = offset + (Dim[k, 0] * Dim[k, 1])

        for j in range(N_eq):
            if j not in Skip[i - 1]:
                for k in range(Dim[i - 1, 0]):
                    start = Dim[i - 1, 1] * Dim[i - 1, 0] * j
                    end   = Dim[i - 1, 1] * Dim[i - 1, 0] * (j + 1)
                    move  = Dim[i - 1, 1] * k + (Dim[i - 1, 1] - 1)

                    A[offset + start + move, :] = 0
                    A[offset + start + move, (offset + start):(offset + end)] = DX[move, :]

    return A


def Neu_BC_hor_min(A, N_eq, Disc, Dimensions, Domain_numbers, D, Skip):
    # Neumann BC Vertical (Experimental)
    Dim = copy.copy(Dimensions)
    if Disc == "Spectral":
        Dim = Dimensions + 1
    elif Disc == "XSpectral":
        for i in range(len(Dimensions[:, 0])):
            Dim[i, 1] = Dimensions[i, 1] + 1
    elif Disc == "YSpectral":
        for i in range(len(Dimensions[:, 0])):
            Dim[i, 0] = Dimensions[i, 0] + 1

    for i in Domain_numbers:
        DY = D[i - 1]

        offset = 0
        for j in range(N_eq):
            for k in range(i - 1):
                if i > 1:
                    offset = offset + (Dim[k, 0] * Dim[k, 1])

        for j in range(N_eq):
            if j not in Skip[i - 1]:
                for k in range(Dim[i - 1, 1]):
                    start = Dim[i - 1, 1] * Dim[i - 1, 0] * j
                    end   = Dim[i - 1, 1] * Dim[i - 1, 0] * (j + 1)
                    ymove = k
                    A[offset + start + ymove, :] = 0
                    A[offset + start + ymove, (offset + start):(offset + end)] = DY[ymove, :]

    return A


def Neu_BC_hor_max(A, N_eq, Disc, Dimensions, Domain_numbers, D, Skip):
    # Neumann BC Vertical (Experimental)
    Dim = copy.copy(Dimensions)
    if Disc == "Spectral":
        Dim = Dimensions + 1
    elif Disc == "XSpectral":
        for i in range(len(Dimensions[:, 0])):
            Dim[i, 1] = Dimensions[i, 1] + 1
    elif Disc == "YSpectral":
        for i in range(len(Dimensions[:, 0])):
            Dim[i, 0] = Dimensions[i, 0] + 1

    for i in Domain_numbers:
        DY = D[i - 1]

        offset = 0
        for j in range(N_eq):
            for k in range(i - 1):
                if i > 1:
                    offset = offset + (Dim[k, 0] * Dim[k, 1])

        for j in range(N_eq):
            if j not in Skip[i - 1]:
                for k in range(Dim[i - 1, 1]):
                    start = Dim[i - 1, 1] * Dim[i - 1, 0] * j
                    end   = Dim[i - 1, 1] * Dim[i - 1, 0] * (j + 1)
                    ymove = Dim[i - 1, 1] * (Dim[i - 1, 0] - 1) + k
                    A[offset + start + ymove, :] = 0
                    A[offset + start + ymove, (offset + start):(offset + end)] = DY[ymove, :]

    return A


#######################################################################################
def Vert_Interface_C0(A, N_eq, Discretization, Dimensions, Ver_interfaces_C0, grids, q):
    Dim = copy.copy(Dimensions)
    if Discretization == "Spectral":
        Dim = Dimensions + 1
    elif Discretization == "XSpectral":
        for i in range(len(Dimensions[:, 0])):
            Dim[i, 1] = Dimensions[i, 1] + 1
    elif Discretization == "YSpectral":
        for i in range(len(Dimensions[:, 0])):
            Dim[i, 0] = Dimensions[i, 0] + 1

    # Domain with lower number side
    for i in Ver_interfaces_C0[:, 0]:
        offset = 0
        for z in range(N_eq):
            for j in range(i - 1):
                if i > 1:
                    offset = offset + (Dim[j, 0] * Dim[j, 1])

        for k in range(N_eq):
            for l in range(Dim[i - 1, 0]):
                A[(Dim[i - 1, 0] * Dim[i - 1, 1]) * k + offset + Dim[i - 1, 1] + Dim[i - 1, 1] * l - 1, :] = 0
                A[(Dim[i - 1, 0] * Dim[i - 1, 1]) * k + offset + Dim[i - 1, 1] + Dim[i - 1, 1] * l - 1,
                  (Dim[i - 1, 0] * Dim[i - 1, 1]) * k + offset + Dim[i - 1, 1] + Dim[i - 1, 1] * l - 1] = -1

    # Domain with higher number side
    for i in range(len(Ver_interfaces_C0[:, 0])):
        offset1 = 0
        offset2 = 0

        for z in range(N_eq):
            for j in range(Ver_interfaces_C0[i, 0] - 1):
                if Ver_interfaces_C0[i, 0] > 1:
                    offset1 = offset1 + (Dim[j, 0] * Dim[j, 1])

        for z in range(N_eq):
            for j in range(Ver_interfaces_C0[i, 1] - 1):
                if Ver_interfaces_C0[i, 1] > 1:
                    offset2 = offset2 + (Dim[j, 0] * Dim[j, 1])

        INTERP_Y_C0 = FDq.FDq_Int(grids[Ver_interfaces_C0[i, 1] - 1], grids[Ver_interfaces_C0[i, 0] - 1], q)

        for k in range(N_eq):
            for j in range(Dim[Ver_interfaces_C0[i, 0] - 1, 0]):
                for l in range(Dim[Ver_interfaces_C0[i, 1] - 1, 0]):
                    A[(Dim[Ver_interfaces_C0[i, 0] - 1, 0] * Dim[Ver_interfaces_C0[i, 0] - 1, 1]) * k + offset1 + Dim[
                        Ver_interfaces_C0[i, 0] - 1, 1] + Dim[
                          Ver_interfaces_C0[i, 0] - 1, 1] * j - 1,
                      (Dim[Ver_interfaces_C0[i, 1] - 1, 0] * Dim[Ver_interfaces_C0[i, 1] - 1, 1]) * k + offset2 + Dim[
                          Ver_interfaces_C0[i, 1] - 1, 1] * l] \
                        = INTERP_Y_C0[j, l]
    return A


def Vert_Interface_C1(A, N_eq, Discretization, Dimensions, Ver_Interfaces_C1, DGL_Z, grids, q):
    Dim = copy.copy(Dimensions)
    if Discretization == "Spectral":
        Dim = Dimensions + 1
    elif Discretization == "XSpectral":
        for i in range(len(Dimensions[:, 0])):
            Dim[i, 1] = Dimensions[i, 1] + 1
    elif Discretization == "YSpectral":
        for i in range(len(Dimensions[:, 0])):
            Dim[i, 0] = Dimensions[i, 0] + 1

    for i in range(len(Ver_Interfaces_C1[:, 1])):
        offsetV = 0
        offsetH = 0

        for z in range(N_eq):
            for j in range(Ver_Interfaces_C1[i, 1] - 1):
                if Ver_Interfaces_C1[i, 1] > 1:
                    offsetV = offsetV + (Dim[j, 0] * Dim[j, 1])

        for z in range(N_eq):
            for j in range(Ver_Interfaces_C1[i, 0] - 1):
                if Ver_Interfaces_C1[i, 0] > 1:
                    offsetH = offsetH + (Dim[j, 0] * Dim[j, 1])

        INTERP_Y_C1 = FDq.FDq_Int(grids[Ver_Interfaces_C1[i, 0] - 1], grids[Ver_Interfaces_C1[i, 1] - 1], q)
        D1 = DGL_Z[Ver_Interfaces_C1[i, 0] - 1]
        D2 = DGL_Z[Ver_Interfaces_C1[i, 1] - 1]

        for k in range(N_eq):
            for j in range(Dim[Ver_Interfaces_C1[i, 1] - 1, 0]):
                A[(Dim[Ver_Interfaces_C1[i, 1] - 1, 0] * Dim[Ver_Interfaces_C1[i, 1] - 1, 1]) * k + offsetV + Dim[
                    Ver_Interfaces_C1[i, 1] - 1, 1] * j, :] = 0

        # For when Computational Derivative is Interfaced
        # ------------------------------------------------
        # Lower Number Side
        # for k in range(N_eq):
        #     for l in range(Dim[Ver_Interfaces_C1[i, 1] - 1, 0]):
        #         for m in range(Dim[Ver_Interfaces_C1[i, 0] - 1, 1]):
        #             for n in range(Dim[Ver_Interfaces_C1[i, 0] - 1, 0]):
        #                 A[(Dim[Ver_Interfaces_C1[i, 1] - 1, 0] * Dim[Ver_Interfaces_C1[i, 1] - 1, 1]) * k + offsetV + Dim[Ver_Interfaces_C1[i, 1] - 1, 1] * l,
        #                   (Dim[Ver_Interfaces_C1[i, 0] - 1, 0] * Dim[Ver_Interfaces_C1[i, 0] - 1, 1]) * k + offsetH + Dim[Ver_Interfaces_C1[i, 0] - 1, 1] * n + m] = \
        #                     - INTERP_Y_C1[l, n] * D1[-1, m]
        #
        # for k in range(N_eq):
        #     for l in range(Dim[Ver_Interfaces_C1[i, 1] - 1, 0]):
        #         for m in range(Dim[Ver_Interfaces_C1[i, 1] - 1, 1]):
        #             A[(Dim[Ver_Interfaces_C1[i, 1] - 1, 0] * Dim[Ver_Interfaces_C1[i, 1] - 1, 1]) * k + offsetV + Dim[Ver_Interfaces_C1[i, 1] - 1, 1] * l,
        #               (Dim[Ver_Interfaces_C1[i, 1] - 1, 0] * Dim[Ver_Interfaces_C1[i, 1] - 1, 1]) * k + offsetV + Dim[Ver_Interfaces_C1[i, 1] - 1, 1] * l + m] = D2[0, m]

        # Lower Number Side
        for k in range(N_eq):
            for l in range(Dim[Ver_Interfaces_C1[i, 1] - 1, 0]):
                verLoc   = offsetV + (Dim[Ver_Interfaces_C1[i, 1] - 1, 0] * Dim[Ver_Interfaces_C1[i, 1] - 1, 1]) * k
                verMove  = Dim[Ver_Interfaces_C1[i, 1] - 1, 1] * l
                horStart = offsetH + (Dim[Ver_Interfaces_C1[i, 0]-1, 0] * Dim[Ver_Interfaces_C1[i, 0]-1, 1]) * k
                horEnd   = offsetH + (Dim[Ver_Interfaces_C1[i, 0]-1, 0] * Dim[Ver_Interfaces_C1[i, 0]-1, 1]) * (k + 1)

                A[verLoc + verMove, horStart:horEnd] = - D1[Dim[Ver_Interfaces_C1[i, 0] - 1, 1] * (l + 1) - 1, :]

        for k in range(N_eq):
            for l in range(Dim[Ver_Interfaces_C1[i, 1] - 1, 0]):
                verLoc   = offsetV + (Dim[Ver_Interfaces_C1[i, 1] - 1, 0] * Dim[Ver_Interfaces_C1[i, 1] - 1, 1]) * k
                verMove  = Dim[Ver_Interfaces_C1[i, 1] - 1, 1] * l
                horStart = offsetV + (Dim[Ver_Interfaces_C1[i, 1]-1, 0] * Dim[Ver_Interfaces_C1[i, 1]-1, 1]) * k
                horEnd   = offsetV + (Dim[Ver_Interfaces_C1[i, 1]-1, 0] * Dim[Ver_Interfaces_C1[i, 1]-1, 1]) * (k + 1)

                A[verLoc + verMove, horStart:horEnd] = D2[verMove, :]

    return A


def Horz_Interface_C0(A, N_eq, Discretization, Dimensions, Hor_Interfaces_C0, grids, q):
    Dim = copy.copy(Dimensions)
    if Discretization == "Spectral":
        Dim = Dimensions + 1
    elif Discretization == "XSpectral":
        for i in range(len(Dimensions[:, 0])):
            Dim[i, 1] = Dimensions[i, 1] + 1
    elif Discretization == "YSpectral":
        for i in range(len(Dimensions[:, 0])):
            Dim[i, 0] = Dimensions[i, 0] + 1

    # Domain with lower number side
    for i in Hor_Interfaces_C0[:, 0]:
        offset = 0
        for z in range(N_eq):
            for j in range(i - 1):
                if i > 1:
                    offset = offset + (Dim[j, 0] * Dim[j, 1])

        for k in range(N_eq):
            for l in range(Dim[i - 1, 1]):
                A[(Dim[i - 1, 0] * Dim[i - 1, 1]) * k + offset + ((Dim[i - 1, 0] - 1) * Dim[i - 1, 1]) + l, :] = 0
                A[(Dim[i - 1, 0] * Dim[i - 1, 1]) * k + offset + ((Dim[i - 1, 0] - 1) * Dim[i - 1, 1]) + l,
                  (Dim[i - 1, 0] * Dim[i - 1, 1]) * k + offset + ((Dim[i - 1, 0] - 1) * Dim[i - 1, 1]) + l] = -1

    # Domain with higher number side
    for i in range(len(Hor_Interfaces_C0[:, 0])):
        offsetV = 0
        offsetH = 0

        for z in range(N_eq):
            for j in range(Hor_Interfaces_C0[i, 0] - 1):
                if Hor_Interfaces_C0[i, 0] > 1:
                    offsetV = offsetV + (Dim[j, 0] * Dim[j, 1])

        for z in range(N_eq):
            for j in range(Hor_Interfaces_C0[i, 1] - 1):
                if Hor_Interfaces_C0[i, 1] > 1:
                    offsetH = offsetH + (Dim[j, 0] * Dim[j, 1])

        INTERP_Z_C0 = FDq.FDq_Int(grids[Hor_Interfaces_C0[i, 1] - 1], grids[Hor_Interfaces_C0[i, 0] - 1], q)

        for k in range(N_eq):
            for j in range(Dim[Hor_Interfaces_C0[i, 0] - 1, 1]):
                for l in range(Dim[Hor_Interfaces_C0[i, 1] - 1, 1]):
                    A[(Dim[Hor_Interfaces_C0[i, 0] - 1, 0] * Dim[Hor_Interfaces_C0[i, 0] - 1, 1]) * k + offsetV + (
                                (Dim[Hor_Interfaces_C0[i, 0] - 1, 0] - 1) * Dim[Hor_Interfaces_C0[i, 0] - 1, 1]) + j,
                      (Dim[Hor_Interfaces_C0[i, 1] - 1, 0] * Dim[Hor_Interfaces_C0[i, 1] - 1, 1]) * k + offsetH + l] = INTERP_Z_C0[j, l]

    return A


def Horz_Interface_C1(A, N_eq, Discretization, Dimensions, Hor_Interfaces_C1, DGL_Y, grids, q):
    Dim = copy.copy(Dimensions)
    if Discretization == "Spectral":
        Dim = Dimensions + 1
    elif Discretization == "XSpectral":
        for i in range(len(Dimensions[:, 0])):
            Dim[i, 1] = Dimensions[i, 1] + 1
    elif Discretization == "YSpectral":
        for i in range(len(Dimensions[:, 0])):
            Dim[i, 0] = Dimensions[i, 0] + 1

    for i in range(len(Hor_Interfaces_C1[:, 1])):
        offsetV = 0
        offsetH = 0

        for z in range(N_eq):
            for j in range(Hor_Interfaces_C1[i, 1] - 1):
                if Hor_Interfaces_C1[i, 1] > 1:
                    offsetV = offsetV + (Dim[j, 0] * Dim[j, 1])

        for z in range(N_eq):
            for j in range(Hor_Interfaces_C1[i, 0] - 1):
                if Hor_Interfaces_C1[i, 0] > 1:
                    offsetH = offsetH + (Dim[j, 0] * Dim[j, 1])

        # INTERP_Z_C1 = InterOp(Dimensions[Hor_Interfaces_C1[i, 0] - 1, 1], Dimensions[Hor_Interfaces_C1[i, 1] - 1, 1])
        INTERP_Z_C1 = FDq.FDq_Int(grids[Hor_Interfaces_C1[i, 0] - 1], grids[Hor_Interfaces_C1[i, 1] - 1], q)

        D1 = DGL_Y[Hor_Interfaces_C1[i, 0] - 1]
        D2 = DGL_Y[Hor_Interfaces_C1[i, 1] - 1]

        for k in range(N_eq):
            for j in range(Dim[Hor_Interfaces_C1[i, 1] - 1, 1]):
                A[(Dim[Hor_Interfaces_C1[i, 1] - 1, 0] * Dim[Hor_Interfaces_C1[i, 1] - 1, 1]) * k + offsetV + j, :] = 0

        # # For when Computational Derivative is Interfaced
        # #------------------------------------------------
        # # Lower Number Side
        # for k in range(N_eq):
        #     for l in range(Dim[Hor_Interfaces_C1[i, 1] - 1, 1]):
        #         for m in range(Dim[Hor_Interfaces_C1[i, 0] - 1, 0]):
        #             for n in range(Dim[Hor_Interfaces_C1[i, 0] - 1, 1]):
        #                 A[(Dim[Hor_Interfaces_C1[i, 1] - 1, 0] * Dim[Hor_Interfaces_C1[i, 1] - 1, 1]) * k + offsetV + l,
        #                   (Dim[Hor_Interfaces_C1[i, 0] - 1, 0] * Dim[Hor_Interfaces_C1[i, 0] - 1, 1]) * k + offsetH + Dim[
        #                       Hor_Interfaces_C1[i, 0] - 1, 1] * m + n] \
        #                     = INTERP_Z_C1[l, n] * D1[-1, m]
        #
        # for k in range(N_eq):
        #     for l in range(Dim[Hor_Interfaces_C1[i, 1] - 1, 1]):
        #         for m in range(Dim[Hor_Interfaces_C1[i, 1] - 1, 0]):
        #             A[(Dim[Hor_Interfaces_C1[i, 1] - 1, 0] * Dim[Hor_Interfaces_C1[i, 1] - 1, 1]) * k + offsetV + l,
        #               (Dim[Hor_Interfaces_C1[i, 1] - 1, 0] * Dim[Hor_Interfaces_C1[i, 1] - 1, 1]) * k + offsetV + Dim[
        #                   Hor_Interfaces_C1[i, 1] - 1, 1] * m + l] = -D2[0, m]

        # Lower Number Domain
        for k in range(N_eq):
            for l in range(Dim[Hor_Interfaces_C1[i, 1] - 1, 1]):
                verLoc   = offsetV + (Dim[Hor_Interfaces_C1[i, 1] - 1, 0] * Dim[Hor_Interfaces_C1[i, 1] - 1, 1]) * k
                verMove  = l
                horStart = offsetH + (Dim[Hor_Interfaces_C1[i, 0]-1, 0] * Dim[Hor_Interfaces_C1[i, 0]-1, 1]) * k
                horEnd   = offsetH + (Dim[Hor_Interfaces_C1[i, 0]-1, 0] * Dim[Hor_Interfaces_C1[i, 0]-1, 1]) * (k + 1)

                A[verLoc + verMove, horStart:horEnd] = - D1[((Dim[Hor_Interfaces_C1[i, 0]-1, 0] - 1) * Dim[Hor_Interfaces_C1[i, 0]-1, 1]) + l, :]

        for k in range(N_eq):
            for l in range(Dim[Hor_Interfaces_C1[i, 1] - 1, 1]):
                verLoc   = offsetV + (Dim[Hor_Interfaces_C1[i, 1] - 1, 0] * Dim[Hor_Interfaces_C1[i, 1] - 1, 1]) * k
                verMove  = l
                horStart = offsetV + (Dim[Hor_Interfaces_C1[i, 1]-1, 0] * Dim[Hor_Interfaces_C1[i, 1]-1, 1]) * k
                horEnd   = offsetV + (Dim[Hor_Interfaces_C1[i, 1]-1, 0] * Dim[Hor_Interfaces_C1[i, 1]-1, 1]) * (k + 1)

                A[verLoc + verMove, horStart:horEnd] = D2[l, :]

    return A
