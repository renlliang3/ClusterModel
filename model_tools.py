"""
This file contains a library of useful tools.

"""
import astropy.units as u
import numpy as np

#==================================================
# Check radius
#==================================================

def check_qarray(qarr):
    """
    Make sure quantity array are arrays

    Parameters
    ----------
    qarr (quantity): array or float, homogeneous to some unit

    Outputs
    ----------
    qarr (quantity): quantity array

    """

    if type(qarr.to_value()) == float:
        qarr = np.array([qarr.to_value()]) * qarr.unit

    return qarr

