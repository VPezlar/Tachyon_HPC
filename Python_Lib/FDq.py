import numpy as np
import subprocess


def FDq_Mat(N, q):
    # Size and order of FDq matrix
    input_size = str(N + 1) + "\n" + str(q)

    # Opening input text file
    f = open('../FDq/inputs/input_size.dat', 'w')
    f.write(input_size)
    f.close()

    # Calling executable to create FDq matrices
    subprocess.call("../FDq/bin/FDq")

    # Reading output of the executable
    D1_FDq = np.loadtxt("../FDq/output/d1.dat")
    D2_FDq = np.loadtxt("../FDq/output/d2.dat")
    grid = np.loadtxt("../FDq/output/eta.dat")
    return [D1_FDq, D2_FDq, grid]


def FDq_Int(grid1, grid2, q):
    # Lengths of respective grids
    N1 = len(grid1)
    N2 = len(grid2)

    # Creating an input file
    f = open('../INTERP/inputs/INTERP.dat', 'w')

    # First line -> lengths of grids and order of int polynomial
    f.write(str(N1) + "," + str(N2) + "," + str(q))

    # Grid1
    for i in range(N1):
        f.write("\n" + str(grid1[i]))

    # Grid2
    for i in range(N2):
        f.write("\n" + str(grid2[i]))
    f.close()

    # Calling Executable
    subprocess.call("../INTERP/bin/INTER")
    Inter_Pol = np.loadtxt("../INTERP/output/DD0.dat")
    return Inter_Pol
