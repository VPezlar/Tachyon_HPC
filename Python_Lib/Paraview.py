import numpy as np


def Export_VTK(Vec, z, y, Dim, Var, N_Sub):

    # Ny_1 = Dim[0, 0]
    # Nz_1 = Dim[0, 1]
    #
    # Vec1 = Vec[0]
    # z1 = z[0]
    # y1 = y[0]

    # Writing each sub-domain into separate .vtk file

    # Subdomain 1
    File1 = Var + "1.vtk"
    VTK1  = open(File1, 'w')
    VTK1.write("# vtk DataFile Version 3.0")
    VTK1.write("\nvtk output mesh")
    VTK1.write("\nASCII")
    VTK1.write("\nDATASET STRUCTURED_GRID")
    VTK1.write("\nDIMENSIONS" + " " + str(Nz_1 + 1) + " " + str(Ny_1 + 1) + " " + "1")
    VTK1.write("\nPOINTS" + " " + str((Nz_1 + 1) * (Ny_1 + 1)) + " " + "float")

    for i in range(Ny_1 + 1):
        for j in range(Nz_1 + 1):
            VTK1.write("\n" + "% .15f" % z1[i,j] + " " + "% .15f" % y1[i,j] + " " + "0")

    VTK1.write("\nPOINT_DATA " + str((Ny_1 + 1) * (Nz_1 + 1)))
    VTK1.write("\nSCALARS Multi double")
    VTK1.write("\nLOOKUP_TABLE default")
    for i in range(Ny_1 + 1):
        for j in range(Nz_1 + 1):
            VTK1.write("\n" + "% .15f" % Vec1[i, j])

    VTK1.close()


