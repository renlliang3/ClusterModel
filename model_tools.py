"""
This file contains a library of useful tools.

"""
import astropy.units as u
import numpy as np

#==================================================
# Check radius
#==================================================

def check_qarray(qarr, unit=None):
    """
    Make sure quantity array are arrays

    Parameters
    ----------
    - qarr (quantity): array or float, homogeneous to some unit

    Outputs
    ----------
    - qarr (quantity): quantity array

    """

    if unit is None:
        if type(qarr) == float:
            qarr = np.array([qarr])
            
    else:
        try:
            test = qarr.to(unit)
        except:
            raise TypeError("Unvalid unit for qarr")

        if type(qarr.to_value()) == float:
            qarr = np.array([qarr.to_value()]) * qarr.unit

    return qarr

#==================================================
# Make 2 d grid from 2 1d arrays
#==================================================

def replicate_array(x, N, T=False):
    """
    Make a two dimension grid based on two 1 dimension arrays,
    such as energy and radius.

    Parameters
    ----------
    - x (quantity): array one
    - Nrep (int): number of time to replicate
    - T (bool): transpose or not

    Outputs
    ----------
    - x_grid (quantity): x1 replicated along one direction 
    as a 2d quantity array

    """
    
    if T:
        x_grid = (np.tile(x, [N,1])).T
    else:
        x_grid = (np.tile(x, [N,1]))

    return x_grid

#==================================================
# Def array based on point per decade, min and max
#==================================================

def sampling_array(xmin, xmax, NptPd=10, unit=False):
    """
    Make an array with a given number of point per decade
    from xmin to xmax

    Parameters
    ----------
    - xmin (quantity): min value of array
    - xmax (quantity): max value of array
    - NptPd (int): the number of point per decade

    Outputs
    ----------
    - array (quantity): the array

    """

    if unit:
        my_unit = xmin.unit
        array = np.logspace(np.log10(xmin.to_value(my_unit)),
                            np.log10(xmax.to_value(my_unit)),
                            int(NptPd*(np.log10(xmax.to_value(my_unit)/xmin.to_value(my_unit)))))*my_unit
    else:
        array = np.logspace(np.log10(xmin), np.log10(xmax), int(NptPd*(np.log10(xmax/xmin))))

    return array

#==================================================
# Integration loglog space with trapezoidale rule
#==================================================

def trapz_loglog(y, x, axis=-1, intervals=False):
    """
    Integrate along the given axis using the composite trapezoidal rule in
    loglog space. Integrate y(x) along given axis in loglog space. y can be a function 
    with multiple dimension. This follows the script in the Naima package.
    
    Parameters
    ----------
    - y (array_like): Input array to integrate.
    - x (array_like):  optional. Independent variable to integrate over.
    - axis (int): Specify the axis.
    - intervals (bool): Return array of shape x not the total integral, default: False
    
    Returns
    -------
    trapz (float): Definite integral as approximated by trapezoidal rule in loglog space.
    """
    
    log10 = np.log10
    
    #----- Check for units
    try:
        y_unit = y.unit
        y = y.value
    except AttributeError:
        y_unit = 1.0
    try:
        x_unit = x.unit
        x = x.value
    except AttributeError:
        x_unit = 1.0

    y = np.asanyarray(y)
    x = np.asanyarray(x)

    #----- Define the slices
    slice1 = [slice(None)] * y.ndim
    slice2 = [slice(None)] * y.ndim
    slice1[axis] = slice(None, -1)
    slice2[axis] = slice(1, None)
    slice1, slice2 = tuple(slice1), tuple(slice2)

    #----- arrays with uncertainties contain objects, remove tiny elements
    if y.dtype == "O":
        from uncertainties.unumpy import log10
        # uncertainties.unumpy.log10 can't deal with tiny values see
        # https://github.com/gammapy/gammapy/issues/687, so we filter out the values
        # here. As the values are so small it doesn't affect the final result.
        # the sqrt is taken to create a margin, because of the later division
        # y[slice2] / y[slice1]
        valid = y > np.sqrt(np.finfo(float).tiny)
        x, y = x[valid], y[valid]

    #----- reshaping x
    if x.ndim == 1:
        shape = [1] * y.ndim
        shape[axis] = x.shape[0]
        x = x.reshape(shape)

    #-----
    with np.errstate(invalid="ignore", divide="ignore"):
        # Compute the power law indices in each integration bin
        b = log10(y[slice2] / y[slice1]) / log10(x[slice2] / x[slice1])
        
        # if local powerlaw index is -1, use \int 1/x = log(x); otherwise use normal
        # powerlaw integration
        trapzs = np.where(np.abs(b + 1.0) > 1e-10,
                          (y[slice1] * (x[slice2] * (x[slice2] / x[slice1]) ** b - x[slice1]))
                          / (b + 1),
                          x[slice1] * y[slice1] * np.log(x[slice2] / x[slice1]))
        
    tozero = (y[slice1] == 0.0) + (y[slice2] == 0.0) + (x[slice1] == x[slice2])
    trapzs[tozero] = 0.0

    if intervals:
        return trapzs * x_unit * y_unit

    ret = np.add.reduce(trapzs, axis) * x_unit * y_unit
    
    return ret

#==================================================
# Compute spectrum (spherical)
#==================================================

def compute_spectrum_spherical(dN_dEdVdt, radius):
    """
    Integrate over the spherical volume to get the spectrum:
    \int_Rmin^Rmax 4 pi r^2 dN_dEdVdt(E,r) dr
    
    Parameters
    ----------
    - dN_dEdVdt (2d array): Input array to integrate.
    - radius (array): Radius variable to integrate over.
    
    Returns
    -------
    dN_dEdt (array): integrated quantity

    """

    dN_dEdt = trapz_loglog(4*np.pi*radius**2*dN_dEdVdt, radius, axis=1, intervals=False)

    return dN_dEdt


#==================================================
# Compute spectrum (cylindrical)
#==================================================

def compute_spectrum_cylindrical_loop(dN_dEdVdt, eng, r2d, los):
    """
    Integrate over the spherical volume to get the spectrum:
    \int_Rmin^Rmax 4 pi r^2 dN_dEdVdt(E,r) dr
    
    Parameters
    ----------
    - dN_dEdVdt (2d array): a function of variable 1 (ex: energy) and radius.
    - eng (quantity array): energy
    - r2d (quantity array): projected radius
    - los (quantity array): line of sight (one side only)
    
    Returns
    -------
    dN_dEdt (array): integrated quantity

    """

    dN_dEdVdt_proj = np.zeros((len(eng), len(r2d)))*u.kpc*u.GeV**-1*u.cm**-3*u.s**-1
    
    for i in range(len(eng)):
        def dN_dEdVdt_1d(r): return dN_dEdVdt(eng[i], r)
        dN_dEdVdt_proj[i,:] = compute_los_integral(dN_dEdVdt_1d, r2d, los)

    dN_dEdt = trapz_loglog(2*np.pi*r2d*dN_dEdVdt_proj, r2d, axis=1, intervals=False)

    return dN_dEdt

#==================================================
# Compute spectrum (cylindrical)
#==================================================

def compute_spectrum_cylindrical(dN_dEdVdt, eng, r2d, los):
    """
    Integrate over the spherical cylindrical volume to get the spectrum:
    \int_Rmin^Rmax 2 pi r dr \int_Rmin^Rmax dN_dEdVdt(E,r) dl
    
    Parameters
    ----------
    - dN_dEdVdt (2d array): a function of energy and radius.
    - eng (quantity array): energy
    - r2d (quantity array): projected radius
    - los (quantity array): line of sight (one side only)
    
    Returns
    -------
    dN_dEdt (array): integrated quantity

    """

    # First compute the los integral assuming a 2d function to get: f(E, r2d)
    dN_dEdVdt_proj = compute_los_integral_2dfunc(dN_dEdVdt, eng, r2d, los)

    # Then integrate over the surface
    dN_dEdt = trapz_loglog(2*np.pi*r2d*dN_dEdVdt_proj, r2d, axis=1, intervals=False)

    return dN_dEdt


#==================================================
# Compute l.o.s. integral for a 2d function
#==================================================

def compute_los_integral_2dfunc(f_E_r, eng, r2d, los):
    """
    Compute the line of sight integral in the case of a 
    function in the case of a dependance of E and r:
    \int_Rmin^Rmax y(E,r) dl
    
    Parameters
    ----------
    - f_E_r (function): a function of energy and radius.
    In practice energy can be anything.
    - eng (quantity array): energy
    - r2d (quantity array): projected radius
    - los (quantity array): line of sight (one side only)
    
    Returns
    -------
    I_los (array): integrated quantity

    """

    import time
    t1 = time.time()
    
    Neng = len(eng)
    Nr2d = len(r2d)
    Nlos = len(los)

    # Compute 2d grids as Nr2d, Nlos to get the 3d radius array
    r2d_g2 = replicate_array(r2d, Nlos, T=True)
    los_g2 = replicate_array(los, Nr2d, T=False)
    r3d_g2 = np.sqrt(r2d_g2**2 + los_g2**2)
    
    # Get a flat array
    r3d_g2_flat = np.ndarray.flatten(r3d_g2)
    
    # Compute f_E_r(E, r)
    f_E_r_g2_flat = f_E_r(eng, r3d_g2_flat)     # Here it takes long: compute on smaller grid and interpolate
    
    # Reshape it as Neng, Nr2d, Nlos
    f_E_r_g3 = np.reshape(f_E_r_g2_flat, (Neng, Nr2d, Nlos))
    
    # compute integral
    I_los = trapz_loglog(2*f_E_r_g3, los, axis=2, intervals=False)
    
    return I_los


#==================================================
# Compute l.o.s. integral for a 2d function
#==================================================

def compute_los_integral(f_r, r2d, los):
    """
    Compute the line of sight integral in the case of a 
    function r only:
    \int_Rmin^Rmax y(r) dl
    
    Parameters
    ----------
    - f_r (function): a function of radius.
    - r2d (quantity array): projected radius
    - los (quantity array): line of sight (one side only)
    
    Returns
    -------
    I_los (array): integrated quantity

    """

    Nr2d = len(r2d)
    Nlos = len(los)

    # Compute 2d grids as Nr2d, Nlos to get the 3d radius array
    r2d_g2 = replicate_array(r2d, Nlos, T=True)
    los_g2 = replicate_array(los, Nr2d, T=False)
    r3d_g2 = np.sqrt(r2d_g2**2 + los_g2**2)

    # Get a flat array
    r3d_g2_flat = np.ndarray.flatten(r3d_g2)

    # Compute f_r(r)
    f_r_g2_flat = f_r(r3d_g2_flat)
    
    # Reshape it as Nr2d, Nlos
    f_r_g2 = np.reshape(f_r_g2_flat, (Nr2d, Nlos))
    
    # Compute integral
    I_los = trapz_loglog(2*f_r_g2, los, axis=1, intervals=False)

    return I_los
