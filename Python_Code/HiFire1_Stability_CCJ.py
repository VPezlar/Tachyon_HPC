# MPI Control
import os
import sys

CWD = os.getcwd()

# Add Python_Lib directory to path
sys.path.insert(0, os.path.join(os.path.dirname(CWD), 'Python_Lib'))

# 3rd Party libraries
import numpy as np  # General array manipulation
import scipy  # Sparse format and sparse eigensolver
import scipy.sparse
import scipy.sparse.linalg
import time  # RunTime monitoring
import copy  # Occasional dealing with numpy pointer implementation
import pandas as pd  # Baseflow read and manipulation

# Config Settings
import configparser # module to load config file
config = configparser.ConfigParser()
config.read('../Input/Config/Config.ini')

# SubRoutines
import spectral as sp  # 1D Linear mapping
import multidomain as md  # Main multi-domain library. (BCs, Interfacing, Conformal, Non-Conformal)
import FDq  # Compact finite-difference derivative matrices
import PBCO
import CLNSEBiOp  # Compressible Linearized NS BiGlobal Operator (Dense, Sparse, Diagonalized)
import Rotex_Mapping as RoMap
import Paraview
import Paraview2
import blasius_mapping as bl


# -----------------------------------------------#
#                   Settings                     #
# -----------------------------------------------#
StartTime = time.time()
RUN = config["SETTINGS"]["RUN"]
MODE = config["SETTINGS"]["MODE"]

# Discretization - x
Nx_1 = int(config["SETTINGS"]["Nx_1"])
Nx_2 = int(config["SETTINGS"]["Nx_2"])

# Discretization - y
Ny_1 = int(config["SETTINGS"]["Ny_1"])
Ny_2 = int(config["SETTINGS"]["Ny_2"])

q = int(config["SETTINGS"]["q"])

# Resolution statement
print("Res. = " + "[" + str(Nx_1 + Nx_2) + " x " + str(max([Ny_1, Ny_2])) + "]")

Dimensions = np.array([[Ny_1, Nx_1],
                       [Ny_2, Nx_2]
                       ])

# -----------------------------------------------#
#             Un-Mapped FDq Matrices             #
# -----------------------------------------------#

# Z Grid and Derivative Matrices
dFDq_xi1, d2FDq_xi1, xi1 = FDq.FDq_Mat(Nx_1, q)
dFDq_xi2, d2FDq_xi2, xi2 = FDq.FDq_Mat(Nx_2, q)

# Y Grid and Derivative Matrices
dFDq_eta1, d2FDq_eta1, eta1 = FDq.FDq_Mat(Ny_1, q)
dFDq_eta2, d2FDq_eta2, eta2 = FDq.FDq_Mat(Ny_2, q)

# Computational Domain Grids
grids_xi = [xi1, xi2]
grids_eta = [eta1, eta2]

# -----------------------------------------------#
#                   Mapping                      #
# -----------------------------------------------#

# ----------------------------------------------------------------------------------------------#
#                   GILADUSH THIS IS THE PART THAT YOU ARE ALLOWED TO TOUCH                     #
# ----------------------------------------------------------------------------------------------#

# Dimensions
Theta_c = 7  # [Deg]
Theta_f = 33
L_sc    = 1000 * 0.4066 / 0.3703771179

# BL percentage
xi1_half  = np.double(config['SETTINGS']['xi1_half'])
xi2_half  = np.double(config['SETTINGS']['xi2_half'])
eta1_half = np.double(config['SETTINGS']['eta1_half'])
eta2_half = np.double(config['SETTINGS']['eta2_half'])

# Domain bounds
# Cone
x1_min = 1000 / L_sc
x1_max = 1

# Cylinder
x2_min = 1
x2_max = (1101.25 + (1601.3 - 1101.25)/(3/2)) / L_sc # Extent of the domain on the cylinder is "hopefully" set by the fraction -> 2/1 = "till middle of the cylinder"

y1_max = 0.124939
y1_min = (y1_max - (x1_max-x1_min) * np.tan(np.deg2rad(Theta_c)))

y2_min = 0.124939
y2_max = 0.124939

Theta_ceil = 8
H_ceil     = 0.165 * 1.2

# --------------------------------------------------------------------------------------------------------#
#                   GILADUSH THIS IS THE PART THAT YOU ARE NO LONGER ALLOWED TO TOUCH                     #
# --------------------------------------------------------------------------------------------------------#

##### Domain 1
# BL Pre-Mapping (X-Direction)
xi1_bl, Jx1_bl, Jxx1_bl = bl.BL_Map(xi1, 1, xi1_half, "True")
dx_1  = scipy.sparse.lil_matrix(scipy.sparse.diags(Jx1_bl, format="lil") @ dFDq_xi1)
dxx_1 = scipy.sparse.lil_matrix(scipy.sparse.diags(Jx1_bl ** 2, format="lil") @ d2FDq_xi1 + scipy.sparse.diags(Jxx1_bl, format="lil") @ dFDq_xi1)

# BL Pre-Mapping (Y-Direction)
eta1_bl, Jy1_bl, Jyy1_bl = bl.BL_Map(eta1, 1, eta1_half, "True")
dy_1  = scipy.sparse.lil_matrix(scipy.sparse.diags(Jy1_bl, format="lil") @ dFDq_eta1)
dyy_1 = scipy.sparse.lil_matrix(scipy.sparse.diags(Jy1_bl ** 2, format="lil") @ d2FDq_eta1 + scipy.sparse.diags(Jyy1_bl, format="lil") @ dFDq_eta1)

# 2D Mapping
DFDq_xi1 = scipy.sparse.kron(scipy.sparse.identity(Ny_1 + 1, format="lil"), dFDq_xi1, format="lil")
D2FDq_xi1 = scipy.sparse.kron(scipy.sparse.identity(Ny_1 + 1, format="lil"), d2FDq_xi1, format="lil")

DFDq_eta1 = scipy.sparse.kron(dFDq_eta1, scipy.sparse.identity(Nx_1 + 1, format="lil"), format="lil")
D2FDq_eta1 = scipy.sparse.kron(d2FDq_eta1, scipy.sparse.identity(Nx_1 + 1, format="lil"), format="lil")

# xi, eta, Dxi, Deta, Theta, Theta_top, xmin, xmax, ymin, ymax, Ceil
x1, y1, dxidx_1, dxidy_1, detadx_1, detady_1, dxdxi_1, dxdeta_1, dydxi_1, dydeta_1 = \
    RoMap.RotexMap(xi1_bl, eta1_bl, DFDq_xi1, DFDq_eta1, Theta_c, Theta_ceil, x1_min, x1_max, y1_min, y1_max, H_ceil, y1_max - x1_max * np.tan(np.deg2rad(Theta_c)))

# Reshaping Mapping Metrics
order = "C"
dxidx1 = np.reshape(dxidx_1, ((Nx_1 + 1) * (Ny_1 + 1)), order=order)
dxidy1 = np.reshape(dxidy_1, ((Nx_1 + 1) * (Ny_1 + 1)), order=order)
dxdxi1 = np.reshape(dxdxi_1, ((Nx_1 + 1) * (Ny_1 + 1)), order=order)
dxdeta1 = np.reshape(dxdeta_1, ((Nx_1 + 1) * (Ny_1 + 1)), order=order)
detadx1 = np.reshape(detadx_1, ((Nx_1 + 1) * (Ny_1 + 1)), order=order)
detady1 = np.reshape(detady_1, ((Nx_1 + 1) * (Ny_1 + 1)), order=order)
dydxi1 = np.reshape(dydxi_1, ((Nx_1 + 1) * (Ny_1 + 1)), order=order)
dydeta1 = np.reshape(dydeta_1, ((Nx_1 + 1) * (Ny_1 + 1)), order=order)

Jac1 = 1 / (dxdxi1 * dydeta1 - dxdeta1 * dydxi1)

D1_x = scipy.sparse.lil_matrix(scipy.sparse.diags(dxidx1, format="lil") @ DFDq_xi1 + scipy.sparse.diags(detadx1, format="lil") @ DFDq_eta1)

D1_xx = scipy.sparse.diags(dxidx1 ** 2, format="lil") @ D2FDq_xi1 + scipy.sparse.diags(detadx1 ** 2, format="lil") @ D2FDq_eta1 + scipy.sparse.diags(2 * dxidx1 * detadx1, format="lil") @ (DFDq_xi1 @ DFDq_eta1) + \
       scipy.sparse.diags(Jac1, format="lil") @ (DFDq_xi1 @ scipy.sparse.diags((dxidx1 ** 2) / Jac1, format="lil") + DFDq_eta1 @ scipy.sparse.diags((dxidx1 * detadx1) / Jac1, format="lil")) @ DFDq_xi1 + \
       scipy.sparse.diags(Jac1, format="lil") @ (DFDq_xi1 @ scipy.sparse.diags((dxidx1 * detadx1) / Jac1, format="lil") + DFDq_eta1 @ scipy.sparse.diags((detadx1 ** 2) / Jac1, format="lil")) @ DFDq_eta1

D1_y = scipy.sparse.lil_matrix(scipy.sparse.diags(dxidy1, format="lil") @ DFDq_xi1 + scipy.sparse.diags(detady1, format="lil") @ DFDq_eta1)

D1_yy = scipy.sparse.diags(dxidy1 ** 2, format="lil") @ D2FDq_xi1 + scipy.sparse.diags(detady1 ** 2, format="lil") @ D2FDq_eta1 + scipy.sparse.diags(2 * dxidy1 * detady1, format="lil") @ (DFDq_xi1 @ DFDq_eta1) + \
       scipy.sparse.diags(Jac1, format="lil") @ (DFDq_xi1 @ scipy.sparse.diags((dxidy1 ** 2) / Jac1, format="lil") + DFDq_eta1 @ scipy.sparse.diags((dxidy1 * detady1) / Jac1, format="lil")) @ DFDq_xi1 + \
       scipy.sparse.diags(Jac1, format="lil") @ (DFDq_xi1 @ scipy.sparse.diags((dxidy1 * detady1) / Jac1, format="lil") + DFDq_eta1 @ scipy.sparse.diags((detady1 ** 2) / Jac1, format="lil")) @ DFDq_eta1


##### Domain 2
# BL Pre-Mapping (X-Direction)
xi2_bl, Jx2_bl, Jxx2_bl = bl.BL_Map(xi2, 1, xi2_half, "True")
dx_2  = scipy.sparse.lil_matrix(scipy.sparse.diags(Jx2_bl, format="lil") @ dFDq_xi2)
dxx_2 = scipy.sparse.lil_matrix(scipy.sparse.diags(Jx2_bl ** 2, format="lil") @ d2FDq_xi2 + scipy.sparse.diags(Jxx2_bl, format="lil") @ dFDq_xi2)

# BL Pre-Mapping (Y-Direction)
eta2_bl, Jy2_bl, Jyy2_bl = bl.BL_Map(eta2, 1, eta2_half, "True")
dy_2  = scipy.sparse.lil_matrix(scipy.sparse.diags(Jy2_bl, format="lil") @ dFDq_eta2)
dyy_2 = scipy.sparse.lil_matrix(scipy.sparse.diags(Jy2_bl ** 2, format="lil") @ d2FDq_eta2 + scipy.sparse.diags(Jyy2_bl, format="lil") @ dFDq_eta2)

# 2D Mapping
DFDq_xi2 = scipy.sparse.kron(scipy.sparse.identity(Ny_2 + 1, format="lil"), dFDq_xi2, format="lil")
D2FDq_xi2 = scipy.sparse.kron(scipy.sparse.identity(Ny_2 + 1, format="lil"), d2FDq_xi2, format="lil")

DFDq_eta2 = scipy.sparse.kron(dFDq_eta2, scipy.sparse.identity(Nx_2 + 1, format="lil"), format="lil")
D2FDq_eta2 = scipy.sparse.kron(d2FDq_eta2, scipy.sparse.identity(Nx_2 + 1, format="lil"), format="lil")

# xi, eta, Dxi, Deta, Theta, Theta_top, xmin, xmax, ymin, ymax, Ceil
x2, y2, dxidx_2, dxidy_2, detadx_2, detady_2, dxdxi_2, dxdeta_2, dydxi_2, dydeta_2 = \
    RoMap.RotexMap(xi2_bl, eta2_bl, DFDq_xi2, DFDq_eta2, 0, Theta_ceil, x2_min, x2_max, y2_min, y2_max, H_ceil + (x2_max - x2_min) * np.tan(np.deg2rad(Theta_ceil)), y2_max)

# Reshaping Mapping Metrics
order = "C"
dxidx2 = np.reshape(dxidx_2, ((Nx_2 + 1) * (Ny_2 + 1)), order=order)
dxidy2 = np.reshape(dxidy_2, ((Nx_2 + 1) * (Ny_2 + 1)), order=order)
dxdxi2 = np.reshape(dxdxi_2, ((Nx_2 + 1) * (Ny_2 + 1)), order=order)
dxdeta2 = np.reshape(dxdeta_2, ((Nx_2 + 1) * (Ny_2 + 1)), order=order)
detadx2 = np.reshape(detadx_2, ((Nx_2 + 1) * (Ny_2 + 1)), order=order)
detady2 = np.reshape(detady_2, ((Nx_2 + 1) * (Ny_2 + 1)), order=order)
dydxi2 = np.reshape(dydxi_2, ((Nx_2 + 1) * (Ny_2 + 1)), order=order)
dydeta2 = np.reshape(dydeta_2, ((Nx_2 + 1) * (Ny_2 + 1)), order=order)

Jac2 = 1 / (dxdxi2 * dydeta2 - dxdeta2 * dydxi2)

D2_x = scipy.sparse.lil_matrix(scipy.sparse.diags(dxidx2, format="lil") @ DFDq_xi2 + scipy.sparse.diags(detadx2, format="lil") @ DFDq_eta2)

D2_xx = scipy.sparse.diags(dxidx2 ** 2, format="lil") @ D2FDq_xi2 + scipy.sparse.diags(detadx2 ** 2, format="lil") @ D2FDq_eta2 + scipy.sparse.diags(2 * dxidx2 * detadx2, format="lil") @ (DFDq_xi2 @ DFDq_eta2) + \
       scipy.sparse.diags(Jac2, format="lil") @ (DFDq_xi2 @ scipy.sparse.diags((dxidx2 ** 2) / Jac2, format="lil") + DFDq_eta2 @ scipy.sparse.diags((dxidx2 * detadx2) / Jac2, format="lil")) @ DFDq_xi2 + \
       scipy.sparse.diags(Jac2, format="lil") @ (DFDq_xi2 @ scipy.sparse.diags((dxidx2 * detadx2) / Jac2, format="lil") + DFDq_eta2 @ scipy.sparse.diags((detadx2 ** 2) / Jac2, format="lil")) @ DFDq_eta2

D2_y = scipy.sparse.lil_matrix(scipy.sparse.diags(dxidy2, format="lil") @ DFDq_xi2 + scipy.sparse.diags(detady2, format="lil") @ DFDq_eta2)

D2_yy = scipy.sparse.diags(dxidy2 ** 2, format="lil") @ D2FDq_xi2 + scipy.sparse.diags(detady2 ** 2, format="lil") @ D2FDq_eta2 + scipy.sparse.diags(2 * dxidy2 * detady2, format="lil") @ (DFDq_xi2 @ DFDq_eta2) + \
       scipy.sparse.diags(Jac2, format="lil") @ (DFDq_xi2 @ scipy.sparse.diags((dxidy2 ** 2) / Jac2, format="lil") + DFDq_eta2 @ scipy.sparse.diags((dxidy2 * detady2) / Jac2, format="lil")) @ DFDq_xi2 + \
       scipy.sparse.diags(Jac2, format="lil") @ (DFDq_xi2 @ scipy.sparse.diags((dxidy2 * detady2) / Jac2, format="lil") + DFDq_eta2 @ scipy.sparse.diags((detady2 ** 2) / Jac2, format="lil")) @ DFDq_eta2


d_Y = [dFDq_eta1, dFDq_eta2]
d_x = [dFDq_xi1, dFDq_xi2]

D1_n = scipy.sparse.lil_matrix(np.cos(np.deg2rad(Theta_c)) * D1_y - np.sin(np.deg2rad(Theta_c)) * D1_x)

D_Y = [D1_y, D2_y]
D_X = [D1_x, D2_x]

D_YY = [D1_yy, D2_yy]
D_XX = [D1_xx, D2_xx]

# Domain Grids
grids_x = [x1, x2]
grids_y = [y1, y2]

del dFDq_xi1, dFDq_xi2
del dFDq_eta1, dFDq_eta2
del dx_1, dx_2
del dy_1, dy_2
del dxx_1, dxx_2
del dyy_1, dyy_2
del dxidx_1, dxidy_1, detadx_1, detady_1, dxdxi_1, dxdeta_1, dydxi_1, dydeta_1
del dxidx1, dxidy1, detadx1, detady1, dxdxi1, dxdeta1, dydxi1, dydeta1
del dxidx_2, dxidy_2, detadx_2, detady_2, dxdxi_2, dxdeta_2, dydxi_2, dydeta_2
del dxidx2, dxidy2, detadx2, detady2, dxdxi2, dxdeta2, dydxi2, dydeta2

# Plotting structured meshes ##########################################
# Opening input text file

MESH_NAME = config["SETTINGS"]["MESH_NAME"]
Paraview2.Export_Mesh("../Output/VTK/MESH/" + MESH_NAME + ".vtk", [grids_x, grids_y], Dimensions + 1)

# -----------------------------------------------#
#                   Baseflow                     #
# -----------------------------------------------#

# # Exporting Stability Grid
np.savetxt(CWD + "/../Input/MESH/StabZ1.csv", x1, delimiter=",")
np.savetxt(CWD + "/../Input/MESH/StabZ2.csv", x2, delimiter=",")

np.savetxt(CWD + "/../Input/MESH/StabY1.csv", y1, delimiter=",")
np.savetxt(CWD + "/../Input/MESH/StabY2.csv", y2, delimiter=",")

if RUN == "Mesh":
    sys.exit()

RHO1 = np.array(pd.read_csv(CWD + "/../Input/BASEFLOW/Rho1.csv", header=None))
RHO2 = np.array(pd.read_csv(CWD + "/../Input/BASEFLOW/Rho2.csv", header=None))

U1   = np.array(pd.read_csv(CWD + "/../Input/BASEFLOW/U1.csv", header=None))
U2   = np.array(pd.read_csv(CWD + "/../Input/BASEFLOW/U2.csv", header=None))

V1   = np.array(pd.read_csv(CWD + "/../Input/BASEFLOW/V1.csv", header=None))
V2   = np.array(pd.read_csv(CWD + "/../Input/BASEFLOW/V2.csv", header=None))

W1   = np.zeros((Nx_1 + 1, Ny_1 + 1))
W2   = np.zeros((Nx_2 + 1, Ny_2 + 1))

T1   = np.array(pd.read_csv(CWD + "/../Input/BASEFLOW/T1.csv", header=None))
T2   = np.array(pd.read_csv(CWD + "/../Input/BASEFLOW/T2.csv", header=None))


if RUN == "Visualize_BF":
    # Paraview
    x = [x1, x2]
    y = [y1, y2]
    RHO_BF = [RHO1, RHO2]
    Paraview.Export_VTK(RHO_BF, x, y, Dimensions, "BF" + "_RHO")

    sys.exit()

ResultantTime = time.time() - StartTime
print("Derivatives formed, BASEFLOW loaded [" + str(ResultantTime / 60) + " min]")

# -----------------------------------------------#
#            Multidomain Discretization          #
# -----------------------------------------------#

Re    = np.double(config["STABILITY"]["Re"])
Mach  = np.double(config["STABILITY"]["Mach"])
gamma = np.double(config["STABILITY"]["gamma"])
T_ref = np.double(config["STABILITY"]["T_ref"])
wave_num  = np.double(config["STABILITY"]["wave_num"])
N_eig = int(config["STABILITY"]["N_eig"])
N_Out = int(config["STABILITY"]["N_Out"])

SHIFT_IM = 1j * np.double(config["STABILITY"]["SHIFT_IM"])
SHIFT_RE = np.double(config["STABILITY"]["SHIFT_RE"])
SHIFT = SHIFT_RE + SHIFT_IM

L1 = scipy.sparse.lil_matrix(CLNSEBiOp.BiOpDiagGen(RHO1, U1, V1, W1, T1, D1_y, D1_yy, D1_x, D1_xx, Re, Mach, wave_num, T_ref, MODE, eta1_bl, y1))
L2 = scipy.sparse.lil_matrix(CLNSEBiOp.BiOpDiagGen(RHO2, U2, V2, W2, T2, D2_y, D2_yy, D2_x, D2_xx, Re, Mach, wave_num, T_ref, MODE, eta2_bl, y2))

# -----------------------------------------------#
#               Multidomain Settings             #
# -----------------------------------------------#

Domains = [L1, L2]  # List of subdomains
del L1, L2

N_eq = 5
Discretization = "Spectral"  # Parameter which affects the BC implementation (For spectral, number of nodes is stated as N + 1)

# Routine to restate the number of points based on definition of the discretization: Spectral = N + 1...
Dim = copy.copy(Dimensions)
if Discretization == "Spectral":
    Dim = Dimensions + 1
elif Discretization == "XSpectral":
    for i in range(len(Dimensions[:, 0])):
        Dim[i, 1] = Dimensions[i, 1] + 1
elif Discretization == "YSpectral":
    for i in range(len(Dimensions[:, 0])):
        Dim[i, 0] = Dimensions[i, 0] + 1

# Specifying which BC are applied to which subdomains
# Dirichlet
Dirichlet_BC_Hor_min = [1, 2]
Dirichlet_BC_Hor_max = [1, 2]
Dirichlet_BC_Ver_min = [1]
Dirichlet_BC_Ver_max = []

# Length of Skip variable must be == N-Domains, blanks are used for unused domain or BCs for all variables at once
SkipDir_hor_min = [[0], [0]]
SkipDir_hor_max = [[], []]
SkipDir_ver_min = [[], []]
SkipDir_ver_max = [[], []]

# Neumann
Neumann_BC_Hor_min = []
Neumann_BC_Hor_max = []
Neumann_BC_Ver_min = []
Neumann_BC_Ver_max = [2]

SkipNeu_hor_min = [[], []]
SkipNeu_hor_max = [[], []]
SkipNeu_ver_min = [[], []]
SkipNeu_ver_max = [[], []]

# Custom BC (Pressure)
Custom_BC_Hor_min = [1, 2]
Custom_BC_Hor_max = []
Custom_BC_Ver_min = []
Custom_BC_Ver_max = []

# Specifying which domains have which interface conditions
Ver_Interfaces_C0 = np.array([[1, 2]])
Ver_Interfaces_C1 = np.array([[1, 2]])
Hor_Interfaces_C0 = np.array([])
Hor_Interfaces_C1 = np.array([])

# -----------------------------------------------#
#                   Dirichlet BC                 #
# -----------------------------------------------#

# Applying Dirichlet BC
Domains = md.Dir_BC_hor_min(N_eq, Discretization, Domains, Dimensions, Dirichlet_BC_Hor_min,
                            SkipDir_hor_min)  # BC for chosen domains (Horizontal, min)
Domains = md.Dir_BC_hor_max(N_eq, Discretization, Domains, Dimensions, Dirichlet_BC_Hor_max,
                            SkipDir_hor_max)  # BC for chosen domains (Horizontal, max)
Domains = md.Dir_BC_ver_min(N_eq, Discretization, Domains, Dimensions, Dirichlet_BC_Ver_min,
                            SkipDir_ver_min)  # BC for chosen domains (Vertical, min)
Domains = md.Dir_BC_ver_max(N_eq, Discretization, Domains, Dimensions, Dirichlet_BC_Ver_max,
                            SkipDir_ver_max)  # BC for chosen domains (Vertical, min)

# Forming A matrix with implemented Dirichlet conditions
L = scipy.sparse.block_diag((Domains[0],
                             Domains[1]), format="lil")

del Domains

# -----------------------------------------------#
#                    Neumann BC                  #
# -----------------------------------------------#

# Applying Neumann BC
L = md.Neu_BC_hor_min(L, N_eq, Discretization, Dimensions, Neumann_BC_Hor_min, D_YY, SkipNeu_hor_min)
L = md.Neu_BC_hor_max(L, N_eq, Discretization, Dimensions, Neumann_BC_Hor_max, D_YY, SkipNeu_hor_max)
L = md.Neu_BC_ver_min(L, N_eq, Discretization, Dimensions, Neumann_BC_Ver_min, D_XX, SkipNeu_ver_min)
L = md.Neu_BC_ver_max(L, N_eq, Discretization, Dimensions, Neumann_BC_Ver_max, D_XX, SkipNeu_ver_max)

# -----------------------------------------------#
#                Custom BC (Pressure...)         #
# -----------------------------------------------#
# Density Equivalent to Wall-Normal Pressure Neumann BC
PBCz = [PBCO.FormPBC(RHO1, T1, D1_n), PBCO.FormPBC(RHO2, T2, D2_x)]
PBCy = [PBCO.FormPBC(RHO1, T1, D1_n), PBCO.FormPBC(RHO2, T2, D2_y)]

del D1_n
del D1_y, D2_y
del D1_x, D2_x

# Imposing Custom Pressure Boundary y=0
for i in Custom_BC_Hor_min:
    offset = 0
    for j in range(N_eq):
        for k in range(i - 1):
            if i > 1:
                offset = offset + (Dim[k, 0] * Dim[k, 1])

    PBC = PBCy[i - 1]
    start = offset
    end = offset + 5 * (Dim[i - 1, 0] * Dim[i - 1, 1])

    for j in range(Dim[i - 1, 1]):
        L[offset + j, start:end] = PBC[j, :]

# Imposing Custom Pressure Boundary y=y_max (Most Probably unused...)
for i in Custom_BC_Hor_max:
    offset = 0
    for j in range(N_eq):
        for k in range(i - 1):
            if i > 1:
                offset = offset + (Dim[k, 0] * Dim[k, 1])

    PBC = PBCy[i - 1]
    jump = Dim[i - 1, 1] * (Dim[i - 1, 0] - 1)
    start = offset
    end = offset + 5 * (Dim[i - 1, 0] * Dim[i - 1, 1])

    for j in range(Dim[i - 1, 1]):
        L[offset + jump + j, start:end] = PBC[jump + j, :]

# Imposing Custom Pressure Boundary z=0
for i in Custom_BC_Ver_min:
    offset = 0
    for j in range(N_eq):
        for k in range(i - 1):
            if i > 1:
                offset = offset + (Dim[k, 0] * Dim[k, 1])

    PBC = PBCz[i - 1]
    move = Dim[i - 1, 1]
    start = offset
    end = offset + 5 * (Dim[i - 1, 0] * Dim[i - 1, 1])

    for j in range(Dim[i - 1, 0]):
        L[offset + move * j, start:end] = PBC[move * j, :]

# Imposing Custom Pressure Boundary z=z_max
for i in Custom_BC_Ver_max:
    offset = 0
    for j in range(N_eq):
        for k in range(i - 1):
            if i > 1:
                offset = offset + (Dim[k, 0] * Dim[k, 1])

    PBC = PBCz[i - 1]
    move = Dim[i - 1, 1]
    start = offset
    end = offset + 5 * (Dim[i - 1, 0] * Dim[i - 1, 1])

    for j in range(Dim[i - 1, 0]):
        L[offset + move + move * j - 1, start:end] = PBC[move + move * j - 1, :]

# -----------------------------------------------#
#                    Interfacing                 #
# -----------------------------------------------#

# Sequentially applying Interface conditions to the already formed A matrix
L = md.Vert_Interface_C1(L, N_eq, Discretization, Dimensions, Ver_Interfaces_C1, D_X, grids_y, q)
L = md.Horz_Interface_C1(L, N_eq, Discretization, Dimensions, Hor_Interfaces_C1, D_Y, grids_x, q)
L = md.Horz_Interface_C0(L, N_eq, Discretization, Dimensions, Hor_Interfaces_C0, grids_x, q)
L = md.Vert_Interface_C0(L, N_eq, Discretization, Dimensions, Ver_Interfaces_C0, grids_y, q)

# -----------------------------------------------#
#                   Mass Matrix                  #
# -----------------------------------------------#

M1 = np.ones(N_eq * (Nx_1 + 1) * (Ny_1 + 1))
M2 = np.ones(N_eq * (Nx_2 + 1) * (Ny_2 + 1))

M = np.block([M1, M2])
del M1, M2
M = scipy.sparse.diags(M, format="lil")

# -----------------------------------------------#
#             Mass Matrix Dirichlet BC           #
# -----------------------------------------------#

# BC f = 0 at y = y_0
for i in range(len(Dirichlet_BC_Hor_min)):
    offset = 0
    for j in range(N_eq):
        for k in range(Dirichlet_BC_Hor_min[i] - 1):
            if Dirichlet_BC_Hor_min[i] > 1:
                offset = offset + (Dim[k, 0] * Dim[k, 1])

    for k in range(N_eq):
        if k not in SkipDir_hor_min[Dirichlet_BC_Hor_min[i - 1] - 1]:
            for l in range(Dim[Dirichlet_BC_Hor_min[i] - 1, 1]):
                if len(M.shape) > 1:
                    M[(Dim[Dirichlet_BC_Hor_min[i] - 1, 0] * Dim[Dirichlet_BC_Hor_min[i] - 1, 1]) * k + offset + l,
                    :] = 0
                else:
                    M[(Dim[Dirichlet_BC_Hor_min[i] - 1, 0] * Dim[Dirichlet_BC_Hor_min[i] - 1, 1]) * k + offset + l] = 0

# BC f = 0 at y = y_max
for i in range(len(Dirichlet_BC_Hor_max)):
    offset = 0
    for j in range(N_eq):
        for k in range(Dirichlet_BC_Hor_max[i] - 1):
            if Dirichlet_BC_Hor_max[i] > 1:
                offset = offset + (Dim[k, 0] * Dim[k, 1])

    for k in range(N_eq):
        if k not in SkipDir_hor_max[Dirichlet_BC_Hor_max[i - 1] - 1]:
            for l in range(Dim[Dirichlet_BC_Hor_max[i] - 1, 1]):
                if len(M.shape) > 1:
                    M[(Dim[Dirichlet_BC_Hor_max[i] - 1, 0] * Dim[Dirichlet_BC_Hor_max[i] - 1, 1]) * k + offset + (
                            (Dim[Dirichlet_BC_Hor_max[i] - 1, 0] - 1) * Dim[Dirichlet_BC_Hor_max[i] - 1, 1]) + l, :] = 0
                else:
                    M[(Dim[Dirichlet_BC_Hor_max[i] - 1, 0] * Dim[Dirichlet_BC_Hor_max[i] - 1, 1]) * k + offset + (
                            (Dim[Dirichlet_BC_Hor_max[i] - 1, 0] - 1) * Dim[Dirichlet_BC_Hor_max[i] - 1, 1]) + l] = 0

# BC f = 0 at z = z_0
for i in range(len(Dirichlet_BC_Ver_min)):
    offset = 0
    for j in range(N_eq):
        for k in range(Dirichlet_BC_Ver_min[i] - 1):
            if Dirichlet_BC_Ver_min[i] > 1:
                offset = offset + (Dim[k, 0] * Dim[k, 1])

        for k in range(N_eq):
            if k not in SkipDir_ver_min[Dirichlet_BC_Ver_min[i - 1] - 1]:
                for l in range(Dim[Dirichlet_BC_Ver_min[i] - 1, 0]):
                    if len(M.shape) > 1:
                        M[
                        (Dim[Dirichlet_BC_Ver_min[i] - 1, 0] * Dim[Dirichlet_BC_Ver_min[i] - 1, 1]) * k + offset + Dim[
                            Dirichlet_BC_Ver_min[i] - 1, 1] * l, :] = 0
                    else:
                        M[(Dim[Dirichlet_BC_Ver_min[i] - 1, 0] * Dim[Dirichlet_BC_Ver_min[i] - 1, 1]) * k + offset +
                          Dim[
                              Dirichlet_BC_Ver_min[i] - 1, 1] * l] = 0

# BC f = 0 at z = z_max
for i in range(len(Dirichlet_BC_Ver_max)):
    offset = 0
    for j in range(N_eq):
        for k in range(Dirichlet_BC_Ver_max[i] - 1):
            if Dirichlet_BC_Ver_max[i] > 1:
                offset = offset + (Dim[k, 0] * Dim[k, 1])

        for k in range(N_eq):
            if k not in SkipDir_ver_max[Dirichlet_BC_Ver_max[i - 1] - 1]:
                for l in range(Dim[Dirichlet_BC_Ver_max[i] - 1, 0]):
                    if len(M.shape) > 1:
                        M[
                        (Dim[Dirichlet_BC_Ver_max[i] - 1, 0] * Dim[Dirichlet_BC_Ver_max[i] - 1, 1]) * k + offset + Dim[
                            Dirichlet_BC_Ver_max[i] - 1, 1] +
                        Dim[Dirichlet_BC_Ver_max[i] - 1, 1] * l - 1, :] = 0
                    else:
                        M[(Dim[Dirichlet_BC_Ver_max[i] - 1, 0] * Dim[Dirichlet_BC_Ver_max[i] - 1, 1]) * k + offset +
                          Dim[Dirichlet_BC_Ver_max[i] - 1, 1] +
                          Dim[Dirichlet_BC_Ver_max[i] - 1, 1] * l - 1] = 0

# -----------------------------------------------#
#              Mass Matrix Neumann BC            #
# -----------------------------------------------#

for i in Neumann_BC_Hor_min:
    offset = 0
    for j in range(N_eq):
        for k in range(i - 1):
            if i > 1:
                offset = offset + (Dim[k, 0] * Dim[k, 1])

    for j in range(N_eq):
        if j not in SkipDir_hor_min[i - 1]:
            for k in range(Dim[i - 1, 1]):
                start = Dim[i - 1, 1] * Dim[i - 1, 0] * j
                ymove = k
                if len(M.shape) > 1:
                    M[offset + start + ymove, :] = 0
                else:
                    M[offset + start + ymove] = 0

for i in Neumann_BC_Hor_max:
    offset = 0
    for j in range(N_eq):
        for k in range(i - 1):
            if i > 1:
                offset = offset + (Dim[k, 0] * Dim[k, 1])

    for j in range(N_eq):
        if j not in SkipDir_hor_max[i - 1]:
            for k in range(Dim[i - 1, 1]):
                start = Dim[i - 1, 1] * Dim[i - 1, 0] * j
                ymove = Dim[i - 1, 1] * (Dim[i - 1, 0] - 1) + k
                if len(M.shape) > 1:
                    M[offset + start + ymove, :] = 0
                else:
                    M[offset + start + ymove] = 0

# BC f = 0 at z = z_0
for i in Neumann_BC_Ver_min:
    offset = 0
    for j in range(N_eq):
        for k in range(i - 1):
            if i > 1:
                offset = offset + (Dim[k, 0] * Dim[k, 1])

    for j in range(N_eq):
        if j not in SkipNeu_ver_min[i - 1]:
            for k in range(Dim[i - 1, 0]):
                start = Dim[i - 1, 1] * Dim[i - 1, 0] * j
                ymove = Dim[i - 1, 1] * k
                if len(M.shape) > 1:
                    M[offset + start + ymove, :] = 0
                else:
                    M[offset + start + ymove] = 0

# BC f = 0 at z = z_max
for i in Neumann_BC_Ver_max:
    offset = 0
    for j in range(N_eq):
        for k in range(i - 1):
            if i > 1:
                offset = offset + (Dim[k, 0] * Dim[k, 1])

    for j in range(N_eq):
        if j not in SkipNeu_ver_max[i - 1]:
            for k in range(Dim[i - 1, 0]):
                ystart = Dim[i - 1, 1] * Dim[i - 1, 0] * j
                ymove = Dim[i - 1, 1] * k + (Dim[i - 1, 1] - 1)
                if len(M.shape) > 1:
                    M[offset + ystart + ymove, :] = 0
                else:
                    M[offset + ystart + ymove] = 0

# -----------------------------------------------#
#              Mass Matrix Custom BC             #
# -----------------------------------------------#

# Imposing Custom Pressure Boundary y=0
for i in Custom_BC_Hor_min:
    offset = 0
    for j in range(N_eq):
        for k in range(i - 1):
            if i > 1:
                offset = offset + (Dim[k, 0] * Dim[k, 1])

    for j in range(Dim[i - 1, 1]):
        if len(M.shape) > 1:
            M[offset + j, :] = 0

# Imposing Custom Pressure Boundary y=y_max (Most Probably unused...)
for i in Custom_BC_Hor_max:
    offset = 0
    for j in range(N_eq):
        for k in range(i - 1):
            if i > 1:
                offset = offset + (Dim[k, 0] * Dim[k, 1])

    jump = Dim[i - 1, 1] * (Dim[i - 1, 0] - 1)

    for j in range(Dim[i - 1, 1]):
        M[offset + jump + j, :] = 0

# Imposing Custom Pressure Boundary z=0
for i in Custom_BC_Ver_min:
    offset = 0
    for j in range(N_eq):
        for k in range(i - 1):
            if i > 1:
                offset = offset + (Dim[k, 0] * Dim[k, 1])

    move = Dim[i - 1, 1]

    for j in range(Dim[i - 1, 0]):
        M[offset + move * j, :] = 0

# Imposing Custom Pressure Boundary z=z_max
for i in Custom_BC_Ver_max:
    offset = 0
    for j in range(N_eq):
        for k in range(i - 1):
            if i > 1:
                offset = offset + (Dim[k, 0] * Dim[k, 1])

    move = Dim[i - 1, 1]

    for j in range(Dim[i - 1, 0]):
        M[offset + move + move * j - 1, :] = 0

# -----------------------------------------------#
#                   Mass Matrix IC               #
# -----------------------------------------------#

if Hor_Interfaces_C0.size > 0:
    # Hor C0 Interfaces
    for i in Hor_Interfaces_C0[:, 0]:
        offset = 0
        for z in range(N_eq):
            for j in range(i - 1):
                if i > 1:
                    offset = offset + (Dim[j, 0] * Dim[j, 1])

            for k in range(N_eq):
                for l in range(Dim[i - 1, 1]):
                    if len(M.shape) > 1:
                        M[(Dim[i - 1, 0] * Dim[i - 1, 1]) * k + offset + ((Dim[i - 1, 0] - 1) * Dim[i - 1, 1]) + l,
                        :] = 0
                    else:
                        M[(Dim[i - 1, 0] * Dim[i - 1, 1]) * k + offset + ((Dim[i - 1, 0] - 1) * Dim[i - 1, 1]) + l] = 0

    # Hor C1 Interfaces
    for i in range(len(Hor_Interfaces_C1[:, 1])):
        offset = 0

        for z in range(N_eq):
            for j in range(Hor_Interfaces_C1[i, 1] - 1):
                if Hor_Interfaces_C1[i, 1] > 1:
                    offset = offset + (Dim[j, 0] * Dim[j, 1])

        for k in range(N_eq):
            for j in range(Dim[Hor_Interfaces_C1[i, 1] - 1, 1]):
                if len(M.shape) > 1:
                    M[(Dim[Hor_Interfaces_C1[i, 1] - 1, 0] * Dim[Hor_Interfaces_C1[i, 1] - 1, 1]) * k + offset + j,
                    :] = 0
                else:
                    M[(Dim[Hor_Interfaces_C1[i, 1] - 1, 0] * Dim[Hor_Interfaces_C1[i, 1] - 1, 1]) * k + offset + j] = 0

if Ver_Interfaces_C0.size > 0:
    # Ver C0 Interfaces
    for i in Ver_Interfaces_C0[:, 0]:
        offset = 0
        for z in range(N_eq):
            for j in range(i - 1):
                if i > 1:
                    offset = offset + (Dim[j, 0] * Dim[j, 1])

            for k in range(N_eq):
                for l in range(Dim[i - 1, 0]):
                    if len(M.shape) > 1:
                        M[(Dim[i - 1, 0] * Dim[i - 1, 1]) * k + offset + (Dim[i - 1, 1]) + (Dim[i - 1, 1]) * l - 1,
                        :] = 0
                    else:
                        M[(Dim[i - 1, 0] * Dim[i - 1, 1]) * k + offset + (Dim[i - 1, 1]) + (Dim[i - 1, 1]) * l - 1] = 0

    # Ver C1 Interfaces
    for i in range(len(Ver_Interfaces_C1[:, 1])):
        offset = 0

        for z in range(N_eq):
            for j in range(Ver_Interfaces_C1[i, 1] - 1):
                if Ver_Interfaces_C1[i, 1] > 1:
                    offset = offset + (Dim[j, 0] * Dim[j, 1])

        for k in range(N_eq):
            for j in range(Dim[Ver_Interfaces_C1[i, 1] - 1, 0]):
                if len(M.shape) > 1:
                    M[(Dim[Ver_Interfaces_C1[i, 1] - 1, 0] * Dim[Ver_Interfaces_C1[i, 1] - 1, 1]) * k + offset + Dim[
                        Ver_Interfaces_C1[i, 1] - 1, 1] * j, :] = 0
                else:
                    M[(Dim[Ver_Interfaces_C1[i, 1] - 1, 0] * Dim[Ver_Interfaces_C1[i, 1] - 1, 1]) * k + offset + Dim[
                        Ver_Interfaces_C1[i, 1] - 1, 1] * j] = 0

# -----------------------------------------------#
#                   Solving GEVP                 #
# -----------------------------------------------#

L = scipy.sparse.csc_matrix(L)  # Preferred sparse format for eigensolver
M = scipy.sparse.csc_matrix(M)  # Preferred sparse format for eigensolver

ResultantTime = time.time() - StartTime
print("Operator formed, BC and IC enforced [" + str(ResultantTime / 60) + " min]")
print("Solving GEVP")

# Actual Eigen-decomposition (Implicitly Restarted Arnoldi Iteration)
eig_val, eig_vec = scipy.sparse.linalg.eigs(L, M=M, sigma=SHIFT, k=N_eig, which="LM")
idx = eig_val.imag.argsort()[::-1]  # Sorting
eig_val = eig_val[idx]
eig_vec = eig_vec[:, idx]
eig_val = np.reshape(eig_val, (len(eig_val), 1))  # Reshaping for reading convenience
print(eig_val[0])

EndTime = time.time()
ResultantTime = EndTime - StartTime
print("Resultant time: " + str(ResultantTime / 60) + " min")

EIGEN = open(CWD + "/../Output/CSV/Eigenvalues/Eigenvalues.csv", 'w')
for i in range(N_eig):
    EIGEN.write("% .15f" % eig_val[i, 0].real + ";" + "% .15f" % eig_val[i, 0].imag + "\n")
EIGEN.close()

# -----------------------------------------------#
#                  Post Processing               #
# -----------------------------------------------#

for i in range(N_Out):
    eig = i
    SD_ID_1 = 5 * ((Nx_1 + 1) * (Ny_1 + 1))
    SD_ID_2 = SD_ID_1 + 5 * ((Nx_2 + 1) * (Ny_2 + 1))

    SolutionOne = eig_vec[0:SD_ID_1, eig]
    SolutionTwo = eig_vec[SD_ID_1:SD_ID_2, eig]

    # Storing individual components
    rho_vec1 = SolutionOne[((Nx_1 + 1) * (Ny_1 + 1)) * 0:((Nx_1 + 1) * (Ny_1 + 1)) * 1]
    rho_vec2 = SolutionTwo[((Nx_2 + 1) * (Ny_2 + 1)) * 0:((Nx_2 + 1) * (Ny_2 + 1)) * 1]

    u_vec1 = SolutionOne[((Nx_1 + 1) * (Ny_1 + 1)) * 1:((Nx_1 + 1) * (Ny_1 + 1)) * 2]
    u_vec2 = SolutionTwo[((Nx_2 + 1) * (Ny_2 + 1)) * 1:((Nx_2 + 1) * (Ny_2 + 1)) * 2]

    v_vec1 = SolutionOne[((Nx_1 + 1) * (Ny_1 + 1)) * 2:((Nx_1 + 1) * (Ny_1 + 1)) * 3]
    v_vec2 = SolutionTwo[((Nx_2 + 1) * (Ny_2 + 1)) * 2:((Nx_2 + 1) * (Ny_2 + 1)) * 3]

    w_vec1 = SolutionOne[((Nx_1 + 1) * (Ny_1 + 1)) * 3:((Nx_1 + 1) * (Ny_1 + 1)) * 4]
    w_vec2 = SolutionTwo[((Nx_2 + 1) * (Ny_2 + 1)) * 3:((Nx_2 + 1) * (Ny_2 + 1)) * 4]

    T_vec1 = SolutionOne[((Nx_1 + 1) * (Ny_1 + 1)) * 4:((Nx_1 + 1) * (Ny_1 + 1)) * 5]
    T_vec2 = SolutionTwo[((Nx_2 + 1) * (Ny_2 + 1)) * 4:((Nx_2 + 1) * (Ny_2 + 1)) * 5]

    # Reshaping for printing
    order = "C"
    rho_vec1 = np.reshape(rho_vec1, ((Ny_1 + 1), (Nx_1 + 1)), order=order)
    rho_vec2 = np.reshape(rho_vec2, ((Ny_2 + 1), (Nx_2 + 1)), order=order)
    rho_vec_RE  = [rho_vec1.real, rho_vec2.real]
    rho_vec_IM = [rho_vec1.imag, rho_vec2.imag]

    u_vec1 = np.reshape(u_vec1, ((Ny_1 + 1), (Nx_1 + 1)), order=order)
    u_vec2 = np.reshape(u_vec2, ((Ny_2 + 1), (Nx_2 + 1)), order=order)
    u_vec_RE = [u_vec1.real, u_vec2.real]
    u_vec_IM = [u_vec1.imag, u_vec2.imag]

    v_vec1 = np.reshape(v_vec1, ((Ny_1 + 1), (Nx_1 + 1)), order=order)
    v_vec2 = np.reshape(v_vec2, ((Ny_2 + 1), (Nx_2 + 1)), order=order)
    v_vec_RE = [v_vec1.real, v_vec2.real]
    v_vec_IM = [v_vec1.imag, v_vec2.imag]

    w_vec1 = np.reshape(w_vec1, ((Ny_1 + 1), (Nx_1 + 1)), order=order)
    w_vec2 = np.reshape(w_vec2, ((Ny_2 + 1), (Nx_2 + 1)), order=order)
    w_vec_RE = [w_vec1.real, w_vec2.real]
    w_vec_IM = [w_vec1.imag, w_vec2.imag]
    vel_vec_RE = [u_vec_RE, v_vec_RE, w_vec_RE]
    vel_vec_IM = [u_vec_IM, v_vec_IM, w_vec_IM]

    T_vec1 = np.reshape(T_vec1, ((Ny_1 + 1), (Nx_1 + 1)), order=order)
    T_vec2 = np.reshape(T_vec2, ((Ny_2 + 1), (Nx_2 + 1)), order=order)
    T_vec_RE = [T_vec1.real, T_vec2.real]
    T_vec_IM = [T_vec1.imag, T_vec2.imag]

    x = [x1, x2]
    y = [y1, y2]

    name = CWD + "/../Output/VTK/Eigenvectors/" + str(i) + ".vtk"

    Paraview2.Export_VTK(name, rho_vec_RE, vel_vec_RE, T_vec_RE, rho_vec_IM, vel_vec_IM, T_vec_IM, [x, y], Dim)