import numpy as np

# Sutherland Law Routine
def mu_suth(Temperature):
    # Values for N2
    # mu_sutherland = 1.663 * (10 ** (-5))
    # T_ref_sutherland = 273.15  # [K]
    # S = 107  # [K]

    # Values for Air
    mu_sutherland = 1.716 * (10 ** (-5))
    T_ref_sutherland = 273.15  # [K]
    S = 110.4  # [K]

    A1 = (mu_sutherland / (T_ref_sutherland ** (3/2))) * (T_ref_sutherland + S)
    Mu = A1 * (Temperature**(3/2)) / (Temperature + S)

    return Mu


# Constants
gamma    = 1.4
R_gas    = 287
Lc       = 0.4066

# Freestream
M_inf    = 6
T_inf    = 67.07
Re_Lc    = 1000000

# Calcs
U_inf    = M_inf * np.sqrt(gamma * R_gas * T_inf)
Mu_inf   = mu_suth(T_inf)
Rho_inf  = Re_Lc * Mu_inf / (U_inf * Lc)
P_inf    = Rho_inf * R_gas * T_inf

