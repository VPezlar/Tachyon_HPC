import numpy as np


def BL_Map(xi, inf, half, metrics):

    # Mapping parameters
    x_inf = inf
    x_half = half
    l = x_inf * x_half / (x_inf - 2 * x_half)
    s = 2 * l / x_inf

    # Mapping into physical space
    x = l * (1 + xi) / (1 + s - xi)

    # Metrics
    J = l*(2+s)/((l+x)**2)
    J2 = -(2*l*(2+s))/((l+x)**3)

    if metrics == "True":
        return [x, J, J2]
    else:
        return x
