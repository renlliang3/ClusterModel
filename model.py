"""
This script contains the Cluster class. It is dedicated to the construction 
of a Cluster object, definined by its physical properties and with methods associated
to compute derived properties or observables. It focuses on the thermal and non-thermal 
component of the clusters.

"""

#==================================================
# Requested imports
#==================================================

import numpy as np
from scipy.misc import derivative
from scipy.optimize import brentq
import scipy.interpolate as interpolate
import scipy.ndimage as ndimage
import pprint
import pickle
import os

import astropy.units as u
from astropy.io import fits
import astropy.cosmology
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy import constants as const
from astropy.table import Table, Column

import naima

from ClusterTools import cluster_global 
from ClusterTools import cluster_profile 
from ClusterTools import cluster_spectra 
from ClusterTools import cluster_xspec 
from ClusterTools import map_tools

from ClusterModel import model_title
from ClusterModel import model_plots

#==================================================
# Cluster class
#==================================================

class Cluster(object):
    """ Cluster class. 
    This class defines a cluster object. In addition to basic properties such as 
    mass and redshift, it includes the following base properties :
    - pressure profile ;
    - density profile ; 
    - cosmic ray spatial/energy distribution ;

    from which derived properties can be computed (e.g. hydrostatic mass profile)
    
    To do list
    ----------  
    - Compute the X-ray counts for ROSAT
    - Split into subclasses : physics, observables, tools
    - Extract astrometric properties of headers when using them instead of just copying it
    - Include EBL in gamma spectrum
    - Include cluster metallicity instead of local abundances for nuclear enhancement
    - Compute the secondary electron/positrons
    - Include the magnetic field profile
    - Compute the radio synchrotron emission from secondaries

    Attributes
    ----------  
    - output_dir (str): directory where to output data files and plots.
    - silent (bool): print information if False, or not otherwise.
    - cosmology (astropy.cosmology): background cosmological model. Can only be set
    when creating the Cluster object.

    - name (str): the name of the cluster
    - coord (SkyCoord object): the coordinate of the cluster.
    - redshift (float): redshift of the cluster center. Changing the redshift 
    on the fly propagate to cluster properties.
    - D_ang (quantity): can be access but not set directly. The redshift+cosmo
    defines this.
    - D_lum (quantity) : can be access but not set directly. The redshift+cosmo
    defines this.
    - M500 (quantity) : the mass enclosed within R500.
    - R500 (quantity) : the radius in which the density is 500 times the critical 
    density at the cluster redshift
    - theta500 (quantity): the angle corresponding to R500.

    - R_truncation (quantity): the radius at which the cluster stops (similar as virial radius)
    - theta_truncation (quantity): the angle corresponding to R_truncation.
    - helium_mass_fraction (float): the helium mass fraction of the gas (==Yp~0.25 in BBN)
    - abundance (float): the abundance in unit of Zsun
    - hse_bias (float): the hydrostatic mass bias, as Mtrue = (1-b) Mhse
    - pressure_gas_model (dict): the model used for the thermal gas electron pressure 
    profile. It contains the name of the model and the associated model parameters. 
    The available models are: 'GNFW', 'isoT'
    - density_gas_model (dict): the model used for the thermal gas electron density 
    profile. It contains the name of the model and the associated model parameters. 
    The available models are: 'SVM', 'beta', 'doublebeta'

    - X_cr (dict): the cosmic ray to thermal energy and the radius used for normalization
    - nuclear_enhancement (bool): compute the pion model with/without nuclear enhancement 
    from local abundances
    - Epmin (quantity): the minimal energy of protons (default is the threshold energy for 
    pi0 production)
    - Epmax (quantity): the maximal energy of protons (default is 10 PeV)
    - density_crp_model (dict): the definition of the cosmic ray proton radial shape
    - spectrum_crp_model (dict): the definition of the cosmic ray proton energy shape

    - map_header (standard header): this allows the user to provide a header directly.
    In this case, the map coordinates, field of view and resolution will be extracted 
    from the header and the projection can be arbitrary. If the header is not provided,
    then the projection will be standard RA-DEC tan projection.
    - map_coord (SkyCoord object): the map center coordinates.
    - map_reso (quantity): the map pixel size, homogeneous to degrees.
    - map_fov (list of quantity):  the map field of view as [FoV_x, FoV_y], homogeneous to deg.

    Methods
    ----------  
    - print_param(self): print the parameters.
    - save_param(self): save the current parameters describing the cluster object.
    - load_param(self, param_file): load a given pre-saved parameter file. The parameter
    file should contain the right parameters to avoid issues latter on.
    - set_pressure_gas_gNFW_param(self, pressure_model='P13UPP'): replace the electron gas 
    pressure profile GNFW parameters by the ones given by the user.
    - get_map_header(self) : return the map header.

    - get_pressure_gas_profile(self, radius=np.logspace(1,5,1000)*u.kpc): compute the electron
    gas pressure profile.
    - get_density_gas_profile(self, radius=np.logspace(0,4,1000)*u.kpc): compute the electron
    gas density profile.
    - get_temperature_gas_profile(self, radius=np.logspace(0,4,1000)*u.kpc): compute the gas
    temperature profile (same as electron or ions)
    - get_entropy_gas_profile(self, radius=np.logspace(0,4,1000)*u.kpc) : compute the entropy profile
    - get_hse_mass_profile(self, radius=np.logspace(0,4,1000)*u.kpc) : compute the hydrostatic 
    mass profile
    - get_overdensity_contrast_profile(self, radius=np.logspace(0,4,1000)*u.kpc): compute the 
    overdensity contrast.
    - get_mdelta_from_profile(self, delta=500, Rmin=10*u.kpc, Rmax=1e4*u.kpc): compute the mass 
    and radius corresponding to a given overdensity contrast, e.g. M500 or R500, from the hydrostatic
    mass profile and a mean hydrostatic mass bias.
    - get_gas_mass_profile(self, radius=np.logspace(0,4,1000)*u.kpc): compute the gas mass profile
    - get_fgas_profile(self, radius=np.logspace(0,4,1000)*u.kpc): compute the gas fraction profile
    - get_thermal_energy_profile(self, radius=np.logspace(0,4,1000)*u.kpc): compute the thermal
    energy profile

    - get_ysph_profile(self, radius=np.logspace(0,4,1000)*u.kpc): compute the spherically 
    integrated compton parameter profile
    - get_ycyl_profile(self, radius=np.logspace(0,4,1000)*u.kpc): compute the cylindrincally 
    integrated Compton parameter profile
    - get_y_compton_profile(self, radius=np.logspace(0,4,1000)*u.kpc, NR500max=5.0, Npt_los=100):
    compute the Compton parameter profile
    - get_ymap(self, FWHM=None, NR500max=5.0, Npt_los=100): compute a Compton parameter map.

    - get_normed_density_crp_profile(self, radius=np.logspace(0,4,1000)*u.kpc): get the radial 
    part of the cosmic ray protons distribution, f(r), in dN/dEdV = A f(r) f(E)
    - get_normed_crp_spectrum(self, energy=np.logspace(-2,7,1000)*u.GeV): get the spectral part
    of the cosmic ray proton distribution, f(E), in dN/dEdV = A f(r) f(E)
    - get_crp_normalization(self): compute the normalization of the cosmic ray proton distribution, 
    A, in dN/dEdV = A f(r) f(E)
    - get_density_crp_profile(self, radius=np.logspace(0,4,1000)*u.kpc, Emin=None, Emax=None, 
    Energy_density=False): compute the cosmic ray proton density profile integrating over the energy 
    between Emin and Emax.
    - get_crp_spectrum(self, energy=np.logspace(-2,7,1000)*u.GeV, Rmax=None): compute the cosmic ray proton
    spectrum integrating over the volume up to Rmax
    - get_crp_to_thermal_energy_profile(self, radius=np.logspace(0,4,1000)*u.kpc, Emin=None, Emax=None):
    compute the cosmic ray proton energy (between Emin and Emax) to thermal energy profile.
    - get_gamma_spectrum(self, energy=np.logspace(-2,6,1000)*u.GeV, Rmax=None,type_integral='spherical', 
    NR500max=5.0, Npt_los=100): compute the gamma ray spectrum integrating over the volume up to Rmax
    - get_gamma_profile(self, radius=np.logspace(0,4,1000)*u.kpc, Emin=10.0*u.MeV, Emax=1.0*u.PeV, 
    Energy_density=False, NR500max=5.0, Npt_los=100): compute the gamma ray profile, integrating over 
    the energy between the gamma ray energy Emin and Emax.
    - get_gamma_flux(self, Rmax=None, type_integral='spherical', NR500max=5.0, Npt_los=100,
    Emin=10.0*u.MeV, Emax=1.0*u.PeV, Energy_density=False): compute the gamma ray flux between 
    energy range and for R>Rmax.
    - get_gamma_template_map(self, NR500max=5.0, Npt_los=100): compute the gamma ray template map, 
    normalized so that the integral over the overall cluster is 1.

    - make_xspec_table(self, Emin=0.1*u.keV, Emax=2.4*u.keV,Tmin=0.1*u.keV, Tmax=50.0*u.keV, nbin=100,
    nH=0.0/u.cm**2, file_HI=None,Kcor=False): compute a temperature versus counts/Sx table to be interpolated
    when getting profiles and maps

    - _save_txt_file(self, filename, col1, col2, col1_name, col2_name, ndec=20): internal method 
    dedicated to save data in special format
    - save_profile(self, radius=np.logspace(0,4,1000)*u.kpc, prod_list=['all'], NR500max=5.0, 
    Npt_los=100, Energy_density=False, Epmin=None, Epmax=None, Egmin=10.0*u.MeV, Egmax=1.0*u.PeV):
    Save the profiles as fits and txt files.
    - save_spectra(self, energy=np.logspace(-2,6,1000)*u.GeV, prod_list=['all'], Rmax=None,
    NR500max=5.0, Npt_los=100): save the spectra as fits and txt files
    - save_map(self, prod_list=['all'], NR500max=5.0, Npt_los=100): save the maps as fits files

    """

    #==================================================
    # Initialize the cluster object
    #==================================================

    def __init__(self,
                 name='Cluster',
                 RA=0.0*u.deg, Dec=0.0*u.deg,
                 redshift=0.01,
                 M500=1e15*u.Unit('Msun'),
                 cosmology=astropy.cosmology.Planck15,
                 silent=False,
    ):
        """
        Initialize the cluster object. Several attributes of the class cannot 
        be defined externally because of intrications between parameters. For 
        instance, the cosmology cannot be changed on the fly because this would 
        mess up the internal consistency.
        
        Parameters
        ----------
        - name (str): cluster name 
        - RA, Dec (quantity): coordinates or the cluster in equatorial frame
        - redshift (float) : the cluster center cosmological redshift
        - M500 (quantity): the cluster mass 
        - cosmology (astropy.cosmology): the name of the cosmology to use.
        - silent (bool): set to true in order not to print informations when running 
        
        """
        
        if not silent:
            model_title.show()
        
        #---------- Admin
        self._output_dir = './ClusterModel'
        self._silent     = silent

        if hasattr(cosmology, 'h') and hasattr(cosmology, 'Om0'):
            self._cosmo = cosmology
        else:
            raise TypeError("Input cosmology must be an instance of astropy.cosmology")
        
        #---------- Global properties
        self._name     = name
        self._coord    = SkyCoord(RA, Dec, frame="icrs")
        self._redshift = redshift
        self._D_ang    = self._cosmo.angular_diameter_distance(self._redshift)
        self._D_lum    = self._cosmo.luminosity_distance(self._redshift)
        self._M500     = M500
        self._R500     = cluster_global.Mdelta_to_Rdelta(self._M500.to_value('Msun'),
                                                         self._redshift, delta=500, cosmo=self._cosmo)*u.kpc
        self._theta500 = ((self._R500 / self._D_ang).to('') * u.rad).to('deg')

        #---------- Thermal gas physics
        # Cluster boundery
        self._R_truncation     = 3*self._R500
        self._theta_truncation = ((self._R_truncation / self._D_ang).to('') * u.rad).to('deg')
        
        # He fraction and metal abundances (in unit of Z_sun)
        self._helium_mass_fraction = 0.245
        self._abundance = 0.3

        # HSE bias
        self._hse_bias = 0.2
        
        # Electronic pressure
        Pnorm = cluster_global.gNFW_normalization(self._redshift, self._M500.to_value('Msun'), cosmo=self._cosmo)
        pppar = [6.410, 1.810, 0.3100, 1.3300, 4.1300]
        self._pressure_gas_model = {"name": "GNFW",
                                    "P_0" : pppar[0]*Pnorm*u.Unit('keV cm-3'),
                                    "c500": pppar[1],
                                    "r_p" : self._R500/pppar[1],
                                    "a"   : pppar[3],
                                    "b"   : pppar[4],
                                    "c"   : pppar[2]}
        
        # Electronic density
        self._density_gas_model = {"name"   : "SVM",
                                   "n_0"    : 3e-3*u.Unit('cm-3'),
                                   "r_c"    : 500.0*u.kpc,
                                   "beta"   : 0.75,
                                   "r_s"    : 800.0*u.kpc,
                                   "alpha"  : 0.6,
                                   "gamma"  : 3.0,
                                   "epsilon": 0.0}

        #---------- Cosmic ray physics
        self._X_cr = {'X':0.01, 'Rcut':self._R500}
        self._nuclear_enhancement = True
        self._Epmin = cluster_spectra.pp_pion_kinematic_energy_threshold() * u.GeV
        self._Epmax = 10.0 * u.PeV
        
        self._density_crp_model = {"name"   : "beta",
                                   "r_c"    : 100.0*u.kpc,
                                   "beta"   : 0.75}
        self._spectrum_crp_model = {'name'       : 'PowerLaw',
                                    'PivotEnergy': 1.0*u.TeV,
                                    'Index'      : 2.5}

        #---------- Sampling
        self._map_header = None
        self._map_coord  = SkyCoord(RA, Dec, frame="icrs")
        self._map_reso   = 0.02*u.deg
        self._map_fov    = [5.0, 5.0]*u.deg

    #==================================================
    # Get the hidden variable
    #==================================================

    #========== Global properties
    @property
    def output_dir(self):
        if not self._silent: print("Getting the output_dir value")
        return self._output_dir

    @property
    def silent(self):
        if not self._silent: print("Getting the silent value")
        return self._silent
    
    @property
    def cosmo(self):
        if not self._silent: print("Getting cosmology")
        return self._cosmo

    @property
    def name(self):
        if not self._silent: print("Getting the name value")
        return self._name
    
    @property
    def coord(self):
        if not self._silent: print("Getting the coordinates")
        return self._coord
    
    @property
    def redshift(self):
        if not self._silent: print("Getting redshift value")
        return self._redshift

    @property
    def D_ang(self):
        if not self._silent: print("Getting D_ang value")
        return self._D_ang
    
    @property
    def D_lum(self):
        if not self._silent: print("Getting D_lum value")
        return self._D_lum
    
    @property
    def M500(self):
        if not self._silent: print("Getting M500 value")
        return self._M500

    @property
    def R500(self):
        if not self._silent: print("Getting R500 value")
        return self._R500

    @property
    def theta500(self):
        if not self._silent: print("Getting theta500 value")
        return self._theta500

    #========== Thermal gas physics
    @property
    def R_truncation(self):
        if not self._silent: print("Getting R_truncation value")
        return self._R_truncation
    
    @property
    def theta_truncation(self):
        if not self._silent: print("Getting theta_truncation value")
        return self._theta_truncation

    @property
    def helium_mass_fraction(self):
        if not self._silent: print("Getting helium mass fraction value")
        return self._helium_mass_fraction

    @property
    def abundance(self):
        if not self._silent: print("Getting the abundance value")
        return self._abundance

    @property
    def hse_bias(self):
        if not self._silent: print("Getting hydrostatic mass bias value")
        return self._hse_bias
    
    @property
    def pressure_gas_model(self):
        if not self._silent: print("Getting the gas electron pressure profile model value")
        return self._pressure_gas_model
    
    @property
    def density_gas_model(self):
        if not self._silent: print("Getting the gas electron density profile model value")
        return self._density_gas_model

    #========== Cosmic Ray physics
    @property
    def X_cr(self):
        if not self._silent: print("Getting the cosmic ray / thermal pressure")
        return self._X_cr

    @property
    def nuclear_enhancement(self):
        if not self._silent: print("Getting the nuclear enhancement")
        return self._nuclear_enhancement
    
    @property
    def Epmin(self):
        if not self._silent: print("Getting the minimal proton energy")
        return self._Epmin

    @property
    def Epmax(self):
        if not self._silent: print("Getting the maximal proton energy")
        return self._Epmax

    @property
    def density_crp_model(self):
        if not self._silent: print("Getting the cosmic ray proton density profile model value")
        return self._density_crp_model

    @property
    def spectrum_crp_model(self):
        if not self._silent: print("Getting the cosmic ray proton spectrum parameters value")
        return self._spectrum_crp_model

    #========== Maps parameters
    @property
    def map_header(self):
        if not self._silent: print("Getting the map header value")
        return self._map_header

    @property
    def map_coord(self):
        if not self._silent: print("Getting the map coord value")
        return self._map_coord

    @property
    def map_reso(self):
        if not self._silent: print("Getting the map resolution value")
        return self._map_reso

    @property
    def map_fov(self):
        if not self._silent: print("Getting the map field of view value")
        return self._map_fov

    #==================================================
    # Defines how the user can pass arguments
    #==================================================

    #========== Global properties
    @output_dir.setter
    def output_dir(self, value):
        # Check value and set
        if type(value) == str:
            self._output_dir = value
        else:
            raise TypeError("The output_dir should be a string.")

        # Information
        if not self._silent: print("Setting output_dir value")

    @silent.setter
    def silent(self, value):
        # Check value and set
        if type(value) == bool:
            self._silent = value
        else:
            raise TypeError("The silent parameter should be a bool.")

        # Information
        if not self._silent: print("Setting silent value")

    @cosmo.setter
    def cosmo(self, value):
        if not self._silent: print("The cosmology can only be set when defining the cluster object,")
        if not self._silent: print("as clust = Cluster(cosmology=astropy.cosmology.YourCosmology). ")
        if not self._silent: print("Doing nothing.                                                 ")

    @name.setter
    def name(self, value):
        # Check value and set
        if value == str:
            self._name = value
        else:
            raise TypeError("The name should be a string.")

        # Information
        if not self._silent: print("Setting name value")

    @coord.setter
    def coord(self, value):
        # Case value is a SkyCoord object
        if type(value) == astropy.coordinates.sky_coordinate.SkyCoord:
            self._coord = value

        # Case value is standard coordinates
        elif type(value) == dict:
            
            # It is not possible to have both RA-Dec and Glat-Glon, or just RA and not Dec, etc
            cond1 = 'RA'  in value.keys() and 'Glat' in value.keys()
            cond2 = 'RA'  in value.keys() and 'Glon' in value.keys()
            cond3 = 'Dec' in value.keys() and 'Glat' in value.keys()
            cond4 = 'Dec' in value.keys() and 'Glon' in value.keys()
            if cond1 or cond2 or cond3 or cond4:
                raise TypeError("The coordinates can be a coord object, or a {'RA','Dec'} or {'Glon', 'Glat'} dictionary.")
            
            # Case where RA-Dec is used
            if 'RA' in value.keys() and 'Dec' in value.keys():
                self._coord = SkyCoord(value['RA'], value['Dec'], frame="icrs")

            # Case where Glon-Glat is used
            elif 'Glon' in value.keys() and 'Glat' in value.keys():
                self._coord = SkyCoord(value['Glon'], value['Glat'], frame="galactic")

            # Otherwise, not appropriate value
            else:
                raise TypeError("The coordinates can be a coord object, a {'RA','Dec'} dictionary, or a {'Glon', 'Glat'} dictionary.")

        # Case value is not accepted
        else:
            raise TypeError("The coordinates can be a coord object, a {'RA','Dec'} dictionary, or a {'Glon', 'Glat'} dictionary.")

        # Information
        if not self._silent: print("Setting coord value")

    @redshift.setter
    def redshift(self, value):
        # value check
        if value < 0 :
            raise ValueError("The redshift should be larger than 0.")

        # Setting parameters
        self._redshift = value
        self._D_ang = self._cosmo.angular_diameter_distance(self._redshift)
        self._D_lum = self._cosmo.luminosity_distance(self._redshift)
        self._R500  = cluster_global.Mdelta_to_Rdelta(self._M500.to_value('Msun'),
                                                      self._redshift, delta=500, cosmo=self._cosmo)*u.kpc            
        self._theta500 = ((self._R500 / self._D_ang).to('') * u.rad).to('deg')
        self._theta_truncation = ((self._R_truncation / self._D_ang).to('') * u.rad).to('deg')
        
        # Information
        if not self._silent: print("Setting redshift value")
        if not self._silent: print("Setting: D_ang, D_lum, R500, theta500, theta_truncation ; Fixing: cosmo.")
        
    @D_ang.setter
    def D_ang(self, value):
        if not self._silent: print("The angular diameter distance cannot be set directly, the redshift has to be used instead.")
        if not self._silent: print("Doing nothing.                                                                            ")

    @D_lum.setter
    def D_lum(self, value):
        if not self._silent: print("The luminosity distance cannot be set directly, the redshift has to be used instead.")
        if not self._silent: print("Doing nothing.                                                                      ")
        
    @M500.setter
    def M500(self, value):
        # Value check
        if value <= 0 :
            raise ValueError("Mass M500 should be larger than 0")
        try:
            test = value.to('Msun')
        except:
            raise TypeError("The mass M500 should be a quantity homogeneous to Msun.")

        # Setting parameters
        self._M500 = value
        self._R500 = cluster_global.Mdelta_to_Rdelta(self._M500.to_value('Msun'),
                                                     self._redshift, delta=500, cosmo=self._cosmo)*u.kpc
        self._theta500 = ((self._R500 / self._D_ang).to('') * u.rad).to('deg')
        
        # Information
        if not self._silent: print("Setting M500 value")
        if not self._silent: print("Setting: R500, theta500 ; Fixing: redshift, cosmo, D_ang")
        
    @R500.setter
    def R500(self, value):
        # check value
        if value < 0 :
            raise ValueError("Radius R500 should be larger than 0")
        try:
            test = value.to('kpc')
        except:
            raise TypeError("The radius R500 should be a quantity homogeneous to kpc.")

        # Setting parameter
        self._R500 = value
        self._theta500 = ((self._R500 / self._D_ang).to('') * u.rad).to('deg')
        self._M500 = cluster_global.Rdelta_to_Mdelta(self._R500.to_value('kpc'),
                                                     self._redshift, delta=500, cosmo=self._cosmo)*u.Msun
        
        # Information
        if not self._silent: print("Setting R500 value")
        if not self._silent: print("Setting: theta500, M500 ; Fixing: redshift, cosmo, D_ang")
        
    @theta500.setter
    def theta500(self, value):
        # check value
        if value <= 0 :
            raise ValueError("Angle theta500 should be larger than 0")
        try:
            test = value.to('deg')
        except:
            raise TypeError("The angle theta500 should be a quantity homogeneous to deg.")

        # Setting parameters
        self._theta500 = value
        self._R500 = value.to_value('rad')*self._D_ang
        self._M500 = cluster_global.Rdelta_to_Mdelta(self._R500.to_value('kpc'),
                                                     self._redshift, delta=500, cosmo=self._cosmo)*u.Msun
        
        # Information
        if not self._silent: print("Setting theta500 value")
        if not self._silent: print("Setting: R500, M500 ; Fixing: redshift, cosmo, D_ang")
        
    #========== Thermal gas physics
    @R_truncation.setter
    def R_truncation(self, value):
        # check value
        if value <= self._R500 :
            raise ValueError("Radius R_truncation should be larger than R500 for internal consistency.")
        try:
            test = value.to('kpc')
        except:
            raise TypeError("The radius R_truncation should be a quantity homogeneous to kpc.")

        # Set parameters
        self._R_truncation = value
        self._theta_truncation = ((self._R_truncation / self._D_ang).to('') * u.rad).to('deg')
        
        # Information
        if not self._silent: print("Setting R_truncation value")
        if not self._silent: print("Setting: theta_truncation ; Fixing: D_ang")
        
    @theta_truncation.setter
    def theta_truncation(self, value):
        # check value
        if value <= self._theta500 :
            raise ValueError("Angle theta_truncation should be larger than theta500 for internal consistency.")
        try:
            test = value.to('deg')
        except:
            raise TypeError("The angle theta_truncation should be a quantity homogeneous to deg.")

        # Set parameters
        self._theta_truncation = value
        self._R_truncation = value.to_value('rad') * self._D_ang
        
        # Information
        if not self._silent: print("Setting theta_truncation value")
        if not self._silent: print("Setting: R_truncation ; Fixing: D_ang")
        
    @helium_mass_fraction.setter
    def helium_mass_fraction(self, value):
        # Check value
        if type(value) == float:
            if value <= 1.0 and value >= 0.0:
                self._helium_mass_fraction = value
            else:
                raise ValueError("The helium mass fraction should be between 0 and 1")
        else:
            raise TypeError("The helium mass fraction should be a float")
        
        # Information
        if not self._silent: print("Setting helium mass fraction value")
        
    @abundance.setter
    def abundance(self, value):
        # Check value
        if type(value) == float:
            if value >= 0.0:
                self._abundance = value
            else:
                raise ValueError("The abundance should be >= 0")
        else:
            raise TypeError("The abundance should be a float")
        
        # Information
        if not self._silent: print("Setting abundance value")
        
    @hse_bias.setter
    def hse_bias(self, value):
        # Check value
        if type(value) == float:
            self._hse_bias = value
        else:
            raise TypeError("The hydrostatic mass bias should be a float")
        
        # Information
        if not self._silent: print("Setting hydrostatic mass bias value")

    @pressure_gas_model.setter
    def pressure_gas_model(self, value):
        # check type
        if type(value) != dict :
            raise TypeError("The pressure gas model should be a dictionary containing the name key and relevant parameters")
        # Continue if ok
        else:
            # Check that the name is provided
            if 'name' not in value.keys() :
                raise ValueError("The pressure gas model should be a dictionary containing the name key and relevant parameters")
            # Check parameters according to the model
            else:

                #---------- Case of GNFW model
                if value['name'] == 'GNFW':
                    # Check the content of the dictionary
                    cond1 = 'P_0' in value.keys() and 'a' in value.keys() and 'b' in value.keys() and 'c' in value.keys()
                    cond2 = 'c500' in value.keys() or 'r_p' in value.keys()
                    cond3 = not('c500' in value.keys() and 'r_p' in value.keys())
                    if cond1 and cond2 and cond3:
                        # Check units and values
                        try:
                            test = value['P_0'].to('keV cm-3')
                        except:
                            raise TypeError("P_0 should be homogeneous to keV cm-3")
                        if value['P_0'] < 0:
                            raise ValueError("P_0 should be >=0")
                        if 'r_p' in value.keys():
                            try:
                                test = value['r_p'].to('kpc')
                            except:
                                raise TypeError("r_p should be homogeneous to kpc")
                            if value['r_p'] <= 0:
                                raise ValueError("r_p should be >0")
                        if 'c500' in value.keys():
                            if value['c500'] <=0:
                                raise ValueError("c500 should be > 0")
                        # All good at this stage, setting parameters
                        if 'c500' in value.keys():
                            c500 = value['c500']
                            r_p = self._R500/value['c500']
                        if 'r_p' in value.keys():
                            c500 = (self._R500/value['r_p']).to_value('')
                            r_p = value['r_p']
                        self._pressure_gas_model = {"name": 'GNFW',
                                                    "P_0" : value['P_0'].to('keV cm-3'),
                                                    "c500": c500,
                                                    "r_p" : r_p.to('kpc'),
                                                    "a":value['a'],
                                                    "b":value['b'],
                                                    "c":value['c']}
                    # The content of the dictionary is not good
                    else:
                        raise ValueError("The GNFW model should contain: {'P_0','c500' or 'r_p' (not both), 'a', 'b', 'c'}.")

                #---------- Case of isothermal model
                elif value['name'] == 'isoT':
                    # Check the content of the dictionary
                    if 'T' in value.keys():
                        # Check units
                        try:
                            test = value['T'].to('keV')
                        except:
                            raise TypeError("T should be homogeneous to keV")
                        # All good at this stage, setting parameters
                        self._pressure_gas_model = {"name": 'isoT',
                                                    "T" : value['T'].to('keV')}
                    # The content of the dictionary is not good
                    else :
                        raise ValueError("The isoT model should contain {'T'}")

                #---------- Case of no match            
                else:
                    raise ValueError("Available models are: 'GNFW', 'isoT'")

        # Information
        if not self._silent: print("Setting pressure_gas_model value")

    @density_gas_model.setter
    def density_gas_model(self, value):
        # check type
        if type(value) != dict :
            raise TypeError("The density gas model should be a dictionary containing the name key and relevant parameters")
        # Continue if ok
        else:
            # Check that the name is provided
            if 'name' not in value.keys() :
                raise ValueError("The density gas model should be a dictionary containing the name key and relevant parameters")
            # Check parameters according to the model
            else:

                #---------- Case of SVM model
                if value['name'] == 'SVM':
                    # Check the content of the dictionary
                    cond1 = 'n_0' in value.keys() and 'r_c' in value.keys() and 'beta' in value.keys()
                    cond2 = 'r_s' in value.keys() and 'alpha' in value.keys() and 'epsilon' in value.keys() and 'gamma' in value.keys()
                    if cond1 and cond2:
                        # Check units
                        try:
                            test = value['n_0'].to('cm-3')
                        except:
                            raise TypeError("n_0 should be homogeneous to cm-3")
                        try:
                            test = value['r_c'].to('kpc')
                        except:
                            raise TypeError("r_c should be homogeneous to kpc")
                        try:
                            test = value['r_s'].to('kpc')
                        except:
                            raise TypeError("r_s should be homogeneous to kpc")
                        # Check values
                        if value['n_0'] < 0:
                            raise ValueError("n_0 should be >= 0")
                        if value['r_c'] <= 0:
                            raise ValueError("r_c should be larger than 0")
                        if value['r_s'] <= 0:
                            raise ValueError("r_s should be larger than 0")                        
                        # All good at this stage, setting parameters
                        self._density_gas_model = {"name"    : 'SVM',
                                                    "n_0"    : value['n_0'].to('cm-3'),
                                                    "r_c"    : value['r_c'].to('kpc'),
                                                    "r_s"    : value['r_s'].to('kpc'),
                                                    "alpha"  : value['alpha'],
                                                    "beta"   : value['beta'],
                                                    "gamma"  : value['gamma'],
                                                    "epsilon": value['epsilon']}
                    # The content of the dictionary is not good
                    else:
                        raise ValueError("The SVM model should contain: {'n_0','beta','r_c','r_s', 'alpha', 'gamma', 'epsilon'}.")

                #---------- Case of beta model
                elif value['name'] == 'beta':
                    # Check the content of the dictionary
                    cond1 = 'n_0' in value.keys() and 'r_c' in value.keys() and 'beta' in value.keys()
                    if cond1:
                        # Check units
                        try:
                            test = value['n_0'].to('cm-3')
                        except:
                            raise TypeError("n_0 should be homogeneous to cm-3")
                        try:
                            test = value['r_c'].to('kpc')
                        except:
                            raise TypeError("r_c should be homogeneous to kpc")
                        # Check values
                        if value['n_0'] < 0:
                            raise ValueError("n_0 should be >= 0")
                        if value['r_c'] <= 0:
                            raise ValueError("r_c should be larger than 0")                   
                        # All good at this stage, setting parameters
                        self._density_gas_model = {"name"    : 'beta',
                                                    "n_0"    : value['n_0'].to('cm-3'),
                                                    "r_c"    : value['r_c'].to('kpc'),
                                                    "beta"   : value['beta']}
                    # The content of the dictionary is not good
                    else:
                        raise ValueError("The beta model should contain: {'n_0','beta','r_c'}.")

                #---------- Case of double beta model
                elif value['name'] == 'doublebeta':
                    # Check the content of the dictionary
                    cond1 = 'n_01' in value.keys() and 'r_c1' in value.keys() and 'beta1' in value.keys()
                    cond2 = 'n_02' in value.keys() and 'r_c2' in value.keys() and 'beta2' in value.keys()
                    if cond1 and cond2:
                        # Check units
                        try:
                            test = value['n_01'].to('cm-3')
                        except:
                            raise TypeError("n_01 should be homogeneous to cm-3")
                        try:
                            test = value['r_c1'].to('kpc')
                        except:
                            raise TypeError("r_c1 should be homogeneous to kpc")
                        try:
                            test = value['n_02'].to('cm-3')
                        except:
                            raise TypeError("n_02 should be homogeneous to cm-3")
                        try:
                            test = value['r_c2'].to('kpc')
                        except:
                            raise TypeError("r_c2 should be homogeneous to kpc")
                        # Check values
                        if value['n_01'] < 0:
                            raise ValueError("n_01 should be >= 0")
                        if value['r_c1'] <= 0:
                            raise ValueError("r_c1 should be larger than 0")
                        if value['n_02'] < 0:
                            raise ValueError("n_02 should be >= 0")
                        if value['r_c2'] <= 0:
                            raise ValueError("r_c2 should be larger than 0")
                        # All good at this stage, setting parameters
                        self._density_gas_model = {"name"    : 'doublebeta',
                                                   "n_01"    : value['n_01'].to('cm-3'),
                                                   "r_c1"    : value['r_c1'].to('kpc'),
                                                   "beta1"   : value['beta1'],
                                                   "n_02"    : value['n_02'].to('cm-3'),
                                                   "r_c2"    : value['r_c2'].to('kpc'),
                                                   "beta2"   : value['beta2']}
                    # The content of the dictionary is not good
                    else:
                        raise ValueError("The double beta model should contain: {'n_01','beta1','r_c1','n_02','beta2','r_c2'}.")
                    
                #---------- Case of no match            
                else:
                    raise ValueError("Available models are: 'SVM', 'beta', 'doublebeta'")

        # Information
        if not self._silent: print("Setting density_gas_model value")

        
    #========== Cosmic Ray physics

    @X_cr.setter
    def X_cr(self, value):
        # Check type and content
        if type(value) != dict :
            raise TypeError("The cosmic/thermal energy should be a dictionary as {'X':CR/th fraction, 'Rcut':enclosed radius}.")
        if 'X' in value.keys() and 'Rcut' in value.keys():
            # Check units and value
            try:
                test = value['Rcut'].to('kpc')
            except:
                raise TypeError("Rcut should be homogeneous to kpc")
            if value['X'] < 0:
                raise ValueError("The cosmic ray to thermal pressure ratio X should be >= 0")
            if value['Rcut'].to_value('kpc') <= 0:
                raise ValueError("The enclosed radius should be > 0")
            # Implement
            self._X_cr = {'X':value['X'], 'Rcut':value['Rcut'].to('kpc')}            
        else:
            raise TypeError("The cosmic/thermal energy should be a dictionary as {'X':CR/th fraction, 'Rcut':enclosed radius}.")
        # Information
        if not self._silent: print("Setting cosmic ray to thermal pressure ratio value")


    @nuclear_enhancement.setter
    def nuclear_enhancement(self, value):
        # Check type and content
        if type(value) != bool :
            raise TypeError("The nuclear_enhancement should be a boolean")
        # Implement
        self._nuclear_enhancement = value
        # Information
        if not self._silent: print("Setting nuclear_enhancement value")
        
    @Epmin.setter
    def Epmin(self, value):
        # Value check
        if value <= 0 :
            raise ValueError("Energy Epmin should be larger than 0")
        try:
            test = value.to('GeV')
        except:
            raise TypeError("The minimal proton energy sould be a quantity homogeneous to GeV.")

        # Setting parameters
        self._Epmin = value
        
        # Information
        if not self._silent: print("Setting Epmin value")
        
    @Epmax.setter
    def Epmax(self, value):
        # Value check
        if value <= 0 :
            raise ValueError("Energy Epmax should be larger than 0")
        try:
            test = value.to('GeV')
        except:
            raise TypeError("The maximal proton energy sould be a quantity homogeneous to GeV.")

        # Setting parameters
        self._Epmax = value
        
        # Information
        if not self._silent: print("Setting Epmax value")
        
    @density_crp_model.setter
    def density_crp_model(self, value):
        # check type
        if type(value) != dict :
            raise TypeError("The density CRp model should be a dictionary containing the name key and relevant parameters")
        # Continue if ok
        else:
            # Check that the name is provided
            if 'name' not in value.keys() :
                raise ValueError("The density CRp model should be a dictionary containing the name key and relevant parameters")
            # Check parameters according to the model
            else:
                
                #---------- Case of SVM model
                if value['name'] == 'SVM':
                    # Check the content of the dictionary
                    cond1 = 'r_c' in value.keys() and 'beta' in value.keys()
                    cond2 = 'r_s' in value.keys() and 'alpha' in value.keys() and 'epsilon' in value.keys() and 'gamma' in value.keys()
                    if cond1 and cond2:
                        # Check units
                        try:
                            test = value['r_c'].to('kpc')
                        except:
                            raise TypeError("r_c should be homogeneous to kpc")
                        try:
                            test = value['r_s'].to('kpc')
                        except:
                            raise TypeError("r_s should be homogeneous to kpc")
                        # Check values
                        if value['r_c'] <= 0:
                            raise ValueError("r_c should be larger than 0")
                        if value['r_s'] <= 0:
                            raise ValueError("r_s should be larger than 0")                        
                        # All good at this stage, setting parameters
                        self._density_crp_model = {"name"    : 'SVM',
                                                   "r_c"    : value['r_c'].to('kpc'),
                                                   "r_s"    : value['r_s'].to('kpc'),
                                                   "alpha"  : value['alpha'],
                                                   "beta"   : value['beta'],
                                                   "gamma"  : value['gamma'],
                                                   "epsilon": value['epsilon']}
                    # The content of the dictionary is not good
                    else:
                        raise ValueError("The SVM model should contain: {'beta','r_c','r_s', 'alpha', 'gamma', 'epsilon'}.")

                #---------- Case of beta model
                elif value['name'] == 'beta':
                    # Check the content of the dictionary
                    cond1 = 'r_c' in value.keys() and 'beta' in value.keys()
                    if cond1:
                        # Check units
                        try:
                            test = value['r_c'].to('kpc')
                        except:
                            raise TypeError("r_c should be homogeneous to kpc")
                        # Check values
                        if value['r_c'] <= 0:
                            raise ValueError("r_c should be larger than 0")
                        # All good at this stage, setting parameters
                        self._density_crp_model = {"name"    : 'beta',
                                                   "r_c"    : value['r_c'].to('kpc'),
                                                   "beta"   : value['beta']}
                    # The content of the dictionary is not good
                    else:
                        raise ValueError("The beta model should contain: {'beta','r_c'}.")

                #---------- Case of double beta model
                elif value['name'] == 'doublebeta':
                    # Check the content of the dictionary
                    cond1 = 'n_01' in value.keys() and 'r_c1' in value.keys() and 'beta1' in value.keys()
                    cond2 = 'n_02' in value.keys() and 'r_c2' in value.keys() and 'beta2' in value.keys()
                    if cond1 and cond2:
                        # Check units
                        if type(value['n_01']) != float:
                            raise TypeError("n_0{1,2} should be a unitless float, because they are only relative normalizations")
                        if type(value['n_02']) != float:
                            raise TypeError("n_0{1,2} should be a unitless float, because they are only relative normalizations")
                        try:
                            test = value['r_c1'].to('kpc')
                        except:
                            raise TypeError("r_c1 should be homogeneous to kpc")
                        try:
                            test = value['r_c2'].to('kpc')
                        except:
                            raise TypeError("r_c2 should be homogeneous to kpc")
                        # Check values
                        if value['n_01'] < 0:
                            raise ValueError("n_01 should be >= 0")
                        if value['r_c1'] <= 0:
                            raise ValueError("r_c1 should be larger than 0")
                        if value['n_02'] < 0:
                            raise ValueError("n_02 should be >= 0")
                        if value['r_c2'] <= 0:
                            raise ValueError("r_c2 should be larger than 0")
                        # All good at this stage, setting parameters
                        self._density_crp_model = {"name"    : 'doublebeta',
                                                   "n_01"    : value['n_01'],
                                                   "r_c1"    : value['r_c1'].to('kpc'),
                                                   "beta1"   : value['beta1'],
                                                   "n_02"    : value['n_02'],
                                                   "r_c2"    : value['r_c2'].to('kpc'),
                                                   "beta2"   : value['beta2']}
                    # The content of the dictionary is not good
                    else:
                        raise ValueError("The double beta model should contain: {'n_01','beta1','r_c1','n_02','beta2','r_c2'}.")

                #---------- Case of GNFW
                elif value['name'] == 'GNFW':
                    # Check the content of the dictionary
                    cond1 = 'a' in value.keys() and 'b' in value.keys() and 'c' in value.keys()
                    cond2 = 'c500' in value.keys() or 'r_p' in value.keys()
                    cond3 = not('c500' in value.keys() and 'r_p' in value.keys())
                    if cond1 and cond2 and cond3:
                        # Check units and values
                        if 'r_p' in value.keys():
                            try:
                                test = value['r_p'].to('kpc')
                            except:
                                raise TypeError("r_p should be homogeneous to kpc")
                            if value['r_p'] <= 0:
                                raise ValueError("r_p should be >0")
                        if 'c500' in value.keys():
                            if value['c500'] <=0:
                                raise ValueError("c500 should be > 0")
                        # All good at this stage, setting parameters
                        if 'c500' in value.keys():
                            c500 = value['c500']
                            r_p = self._R500/value['c500']
                        if 'r_p' in value.keys():
                            c500 = (self._R500/value['r_p']).to_value('')
                            r_p = value['r_p']
                        self._density_crp_model = {"name": 'GNFW',
                                                   "c500": c500,
                                                   "r_p" : r_p.to('kpc'),
                                                   "a":value['a'],
                                                   "b":value['b'],
                                                   "c":value['c']}
                    # The content of the dictionary is not good
                    else:
                        raise ValueError("The GNFW model should contain: {'c500' or 'r_p' (not both), 'a', 'b', 'c'}.")
                    
                #---------- Case of no match            
                else:
                    raise ValueError("Available models are: 'SVM', 'beta', 'doublebeta'")

        # Information
        if not self._silent: print("Setting density_crp_model value")

    @spectrum_crp_model.setter
    def spectrum_crp_model(self, value):
        # check type
        if type(value) != dict :
            raise TypeError("The spectrum CRp model should be a dictionary containing the name key and relevant parameters")
        # Continue if ok
        else:
            # Check that the name is provided
            if 'name' not in value.keys() :
                raise ValueError("The spectrum CRp model should be a dictionary containing the name key and relevant parameters")
            # Check parameters according to the model
            else:

                #---------- Case of PowerLaw model
                if value['name'] == 'PowerLaw':
                    # Check the content of the dictionary
                    cond1 = 'Index' in value.keys()
                    if cond1:
                        self._spectrum_crp_model = {"name"       : 'PowerLaw',
                                                    "Index"       : value['Index']}
                    # The content of the dictionary is not good
                    else:
                        raise ValueError("The PowerLaw model should contain: {'Index'}.")

                #---------- Case of PowerLawExpCutoff model
                elif value['name'] == 'ExponentialCutoffPowerLaw':
                    # Check the content of the dictionary
                    cond1 = 'Index' in value.keys() and 'CutoffEnergy' in value.keys()
                    if cond1:
                        # Check units
                        try:
                            test = value['CutoffEnergy'].to('TeV')
                        except:
                            raise TypeError("CutoffEnergy should be homogeneous to TeV")
                        # Check values
                        if value['CutoffEnergy'] < 0:
                            raise ValueError("CutoffEnergy should be <= 0")                   
                        # All good at this stage, setting parameters
                        self._spectrum_crp_model = {"name"        : 'ExponentialCutoffPowerLaw',
                                                    "Index"       : value['Index'],
                                                    "CutoffEnergy": value['CutoffEnergy'].to('TeV')}
                    # The content of the dictionary is not good
                    else:
                        raise ValueError("The ExponentialCutoffPowerLaw model should contain: {'Index', 'CutoffEnergy'}.")
                
                #---------- Case of no match            
                else:
                    raise ValueError("Available models are: 'PowerLaw', 'ExponentialCutoffPowerLaw'")

        # Information
        if not self._silent: print("Setting spectrum_crp_model value")
        
    #========== Maps
    @map_header.setter
    def map_header(self, value):
        # Check the header by reading it with WCS
        try:
            w = WCS(value)
        except:
            raise TypeError("It seems that the header you provided is not really a header.")
        
        # set the value
        self._map_header = value
        self._map_coord  = None
        self._map_reso   = None
        self._map_fov    = None

        # Information
        if not self._silent: print("Setting the map header")
        if not self._silent: print("Setting: map_coord, map_reso, map_fov to None, as the header will be used")
    
    @map_coord.setter
    def map_coord(self, value):
        # Case value is a SkyCoord object
        if type(value) == astropy.coordinates.sky_coordinate.SkyCoord:
            self._map_coord = value

        # Case value is standard coordinates
        elif type(value) == dict:
            
            # It is not possible to have both RA-Dec and Glat-Glon, or just RA and not Dec, etc
            cond1 = 'RA'  in value.keys() and 'Glat' in value.keys()
            cond2 = 'RA'  in value.keys() and 'Glon' in value.keys()
            cond3 = 'Dec' in value.keys() and 'Glat' in value.keys()
            cond4 = 'Dec' in value.keys() and 'Glon' in value.keys()
            if cond1 or cond2 or cond3 or cond4:
                raise ValueError("The coordinates can be a coord object, or a {'RA','Dec'} or {'Glon', 'Glat'} dictionary.")
            
            # Case where RA-Dec is used
            if 'RA' in value.keys() and 'Dec' in value.keys():
                self._map_coord = SkyCoord(value['RA'], value['Dec'], frame="icrs")

            # Case where Glon-Glat is used
            elif 'Glon' in value.keys() and 'Glat' in value.keys():
                self._map_coord = SkyCoord(value['Glon'], value['Glat'], frame="galactic")

            # Otherwise, not appropriate value
            else:
                raise TypeError("The coordinates can be a coord object, a {'RA','Dec'} dictionary, or a {'Glon', 'Glat'} dictionary.")

        # Case value is not accepted
        else:
            raise TypeError("The coordinates can be a coord object, a {'RA','Dec'} dictionary, or a {'Glon', 'Glat'} dictionary.")

        # Header to None
        self._map_header = None

        # Information
        if not self._silent: print("Setting the map coordinates")
        if not self._silent: print("Setting: map_header to None, as map properties are now set by hand")


    @map_reso.setter
    def map_reso(self, value):
        # check value
        try:
            test = value.to('deg')
        except:
            raise TypeError("The map resolution should be a quantity homogeneous to deg.")

        if type(value.value) != float:        
            raise TypeError("The map resolution should be a scalar, e.i. reso_x = reso_y.")

        # Set parameters
        self._map_reso = value
        self._map_header = None
        
        # Information
        if not self._silent: print("Setting the map resolution value")
        if not self._silent: print("Setting: map_header to None, as map properties are now set by hand")

    @map_fov.setter
    def map_fov(self, value):
        # check that the unit are fine
        try:
            test = value.to('deg')
        except:
            raise TypeError("The map field of view should be a quantity homogeneous to deg.")

        # Set parameters for single value application
        if type(value.value) == float:        
            self._map_fov = [value.to_value('deg'), value.to_value('deg')] * u.deg
    
        # Set parameters for single value application
        elif type(value.value) == np.ndarray:
            # check the dimension
            if len(value) == 2:
                self._map_fov = value
            else:
                raise TypeError("The map field of view is either a scalar, or a 2d list quantity.")

        # No other options
        else:
            raise TypeError("The map field of view is either a scalar, or a 2d list quantity.")

        self._map_header = None

        # Information
        if not self._silent: print("Setting the map field of view")
        if not self._silent: print("Setting: map_header to None, as map properties are now set by hand")


    #==================================================
    # Print parameters
    #==================================================
    
    def print_param(self):
        """
        Print the current parameters describing the cluster.
        
        Parameters
        ----------
            
        Outputs
        ----------
        The parameters are printed in the terminal
            
        """
        pp = pprint.PrettyPrinter(indent=4)

        par = self.__dict__
        keys = par.keys()

        for k in range(len(keys)):
            print('--- '+(keys[k])[1:])
            print('    '+str(par[keys[k]]))
            print('    '+str(type(par[keys[k]]))+'')


    #==================================================
    # Print parameters
    #==================================================
    
    def save_param(self):
        """
        Save the current parameters.
        
        Parameters
        ----------
            
        Outputs
        ----------
        The parameters are saved in the output directory

        """

        # Create the output directory if needed
        if not os.path.exists(self._output_dir): os.mkdir(self._output_dir)

        # Save
        with open(self._output_dir+'/parameters.pkl', 'wb') as pfile:
            pickle.dump(self.__dict__, pfile, pickle.HIGHEST_PROTOCOL)

        # Text file for user
        par = self.__dict__
        keys = par.keys()
        with open(self._output_dir+'/parameters.txt', 'w') as txtfile:
            for k in range(len(keys)):
                txtfile.write('--- '+(keys[k])[1:]+'\n')
                txtfile.write('    '+str(par[keys[k]])+'\n')
                txtfile.write('    '+str(type(par[keys[k]]))+'\n')

                
    #==================================================
    # Print parameters
    #==================================================
    
    def load_param(self, param_file):
        """
        Read the a given parameter file to re-initialize the cluster object.
        
        Parameters
        ----------
        param_file (str): the parameter file to be read
            
        Outputs
        ----------
            
        """

        with open(param_file, 'rb') as pfile:
            par = pickle.load(pfile)
        
        self.__dict__ = par


    #==================================================
    # Set a given pressure UPP profile
    #==================================================
    
    def set_pressure_gas_gNFW_param(self, pressure_model='P13UPP'):
        """
        Set the parameters of the pressure profile:
        P0, c500 (and r_p given R500), gamma, alpha, beta
        
        Parameters
        ----------
        - pressure_model (str): available models are 'A10UPP' (Universal Pressure Profile from 
        Arnaud et al. 2010), 'A10CC' (Cool-Core Profile from Arnaud et al. 2010), 'A10MD' 
        (Morphologically-Disturbed Profile from Arnaud et al. 2010), or 'P13UPP' (Planck 
        Intermediate Paper V (2013) Universal Pressure Profile).

        """

        # Arnaud et al. (2010) : Universal Pressure Profile parameters
        if pressure_model == 'A10UPP':
            if not self._silent: print('Setting gNFW Arnaud et al. (2010) UPP to compute it from M and z.')
            pppar = [8.403, 1.177, 0.3081, 1.0510, 5.4905]

        # Arnaud et al. (2010) : Cool-Core clusters parameters
        elif pressure_model == 'A10CC':
            if not self._silent: print('Setting gNFW Arnaud et al. (2010) cool-core to compute it from M and z.')
            pppar = [3.249, 1.128, 0.7736, 1.2223, 5.4905]

        # Arnaud et al. (2010) : Morphologically-Disturbed clusters parameters
        elif pressure_model == 'A10MD':
            if not self._silent: print('Setting gNFW Arnaud et al. (2010) morphologically disturbed to compute it from M and z.')
            pppar = [3.202, 1.083, 0.3798, 1.4063, 5.4905]

        # Planck Intermediate Paper V (2013) : Universal Pressure Profile parameters
        elif pressure_model == 'P13UPP':
            if not self._silent: print('Setting gNFW Planck coll. (2013) UPP to compute it from M and z.')
            pppar = [6.410, 1.810, 0.3100, 1.3300, 4.1300]

        # No other profiles available
        else:
            raise ValueError('Pressure profile requested model not available. Use A10UPP, A10CC, A10MD or P13UPP.')

        # Compute the normalization
        Pnorm = cluster_global.gNFW_normalization(self._redshift, self._M500.to_value('Msun'), cosmo=self._cosmo)
        
        # Set the parameters accordingly
        self._pressure_gas_model = {"name": 'GNFW',
                                    "P_0" : pppar[0]*Pnorm*u.Unit('keV cm-3'),
                                    "c500": pppar[1],
                                    "r_p" : self._R500/pppar[1],
                                    "a":pppar[3],
                                    "b":pppar[4],
                                    "c":pppar[2]}

        
    #==================================================
    # Extract the header
    #==================================================
    
    def get_map_header(self):
        """
        Extract the header of the map
        
        Parameters
        ----------

        Outputs
        ----------
        - header (astropy object): the header associated to the map

        """

        # Get the needed parameters in case of map header
        if self._map_header != None:
            header = self._map_header
        # Get the needed parameters in case of set-by-hand map parameters
        elif (self._map_coord != None) and (self._map_reso != None) and (self._map_fov != None):
            header = map_tools.define_std_header(self._map_coord.icrs.ra.to_value('deg'),
                                                 self._map_coord.icrs.dec.to_value('deg'),
                                                 self._map_fov.to_value('deg')[0],
                                                 self._map_fov.to_value('deg')[1],
                                                 self._map_reso.to_value('deg'))
        # Otherwise there is a problem
        else:
            raise TypeError("A header, or the map_{coord & reso & fov} should be defined.")

        return header        
        
        
    #==================================================
    # Get the gas electron pressure profile
    #==================================================

    def get_pressure_gas_profile(self, radius=np.logspace(0,4,1000)*u.kpc):
        """
        Get the thermal electron pressure profile.
        
        Parameters
        ----------
        - radius (quantity) : the physical 3d radius in units homogeneous to kpc, as a 1d array

        Outputs
        ----------
        - radius (quantity): the 3d radius in unit of kpc
        - p_r (quantity): the electron pressure profile in unit of keV cm-3

        """

        # In case the input is not an array
        if type(radius.to_value()) == float:
            radius = np.array([radius.to_value()]) * radius.unit
            
        # Case of GNFW profile
        if self._pressure_gas_model['name'] == 'GNFW':
            P0 = self._pressure_gas_model["P_0"].to_value('keV cm-3')
            rp = self._pressure_gas_model["r_p"].to_value('kpc')
            a  = self._pressure_gas_model["a"]
            b  = self._pressure_gas_model["b"]
            c  = self._pressure_gas_model["c"]
            r3d_kpc = radius.to_value('kpc')
            p_r = cluster_profile.gNFW_model(r3d_kpc, P0, rp, slope_a=a, slope_b=b, slope_c=c)
            
            p_r[radius > self._R_truncation] *= 0
            return radius, p_r*u.Unit('keV cm-3')

        # Case of isoT model
        elif self._pressure_gas_model['name'] == 'isoT':
            radius, n_r = self.get_density_gas_profile(radius=radius)
            p_r = n_r * self._pressure_gas_model['T']

            p_r[radius > self._R_truncation] *= 0
            return radius, p_r.to('keV cm-3')
            
        # Otherwise nothing is done
        else :
            if not self._silent: print('Only the GNFW and isoT pressure profile is available for now.')

            
    #==================================================
    # Get the gas electron density profile
    #==================================================
    
    def get_density_gas_profile(self, radius=np.logspace(0,4,1000)*u.kpc):
        """
        Get the thermal electron density profile.
        
        Parameters
        ----------
        - radius (quantity) : the physical 3d radius in units homogeneous to kpc, as a 1d array

        Outputs
        ----------
        - radius (quantity): the 3d radius in unit of kpc
        - n_r (quantity): the electron density profile in unit of cm-3

        """

        # In case the input is not an array
        if type(radius.to_value()) == float:
            radius = np.array([radius.to_value()]) * radius.unit

        # Case of SVM profile
        if self._density_gas_model['name'] == 'SVM':
            n0      = self._density_gas_model["n_0"].to_value('cm-3')
            rc      = self._density_gas_model["r_c"].to_value('kpc')
            rs      = self._density_gas_model["r_s"].to_value('kpc')
            alpha   = self._density_gas_model["alpha"]
            beta    = self._density_gas_model["beta"]
            gamma   = self._density_gas_model["gamma"]
            epsilon = self._density_gas_model["epsilon"]
            r3d_kpc = radius.to_value('kpc')
            n_r = cluster_profile.svm_model(r3d_kpc, n0, rc, beta, rs, gamma, epsilon, alpha)
            
            n_r[radius > self._R_truncation] *= 0
            return radius, n_r*u.Unit('cm-3')

        # beta model
        elif self._density_gas_model['name'] == 'beta':
            n0      = self._density_gas_model["n_0"].to_value('cm-3')
            rc      = self._density_gas_model["r_c"].to_value('kpc')
            beta    = self._density_gas_model["beta"]
            r3d_kpc = radius.to_value('kpc')
            n_r = cluster_profile.beta_model(r3d_kpc, n0, rc, beta)
            
            n_r[radius > self._R_truncation] *= 0
            return radius, n_r*u.Unit('cm-3')

        # double beta model
        elif self._density_gas_model['name'] == 'doublebeta':
            n01      = self._density_gas_model["n_01"].to_value('cm-3')
            rc1      = self._density_gas_model["r_c1"].to_value('kpc')
            beta1    = self._density_gas_model["beta1"]
            n02      = self._density_gas_model["n_02"].to_value('cm-3')
            rc2      = self._density_gas_model["r_c2"].to_value('kpc')
            beta2    = self._density_gas_model["beta2"]
            r3d_kpc = radius.to_value('kpc')
            n_r1 = cluster_profile.beta_model(r3d_kpc, n01, rc1, beta1)
            n_r2 = cluster_profile.beta_model(r3d_kpc, n02, rc2, beta2)
            n_r = n_r1 + n_r2
            
            n_r[radius > self._R_truncation] *= 0
            return radius, n_r*u.Unit('cm-3')

        # Otherwise nothing is done
        else :
            if not self._silent: print('Only the SVM, beta and doublebeta density profile are available for now.')

            
    #==================================================
    # Get the gas electron temperature profile
    #==================================================
    
    def get_temperature_gas_profile(self, radius=np.logspace(0,4,1000)*u.kpc):
        """
        Get the thermal temperature profile.
        
        Parameters
        ----------
        - radius (quantity): the physical 3d radius in units homogeneous to kpc, as a 1d array

        Outputs
        ----------
        - radius (quantity): the 3d radius in unit of kpc
        - T_r (quantity): the temperature profile in unit of keV

        """
        
        # In case the input is not an array
        if type(radius.to_value()) == float:
            radius = np.array([radius.to_value()]) * radius.unit

        # Compute n and P
        radius, n_r = self.get_density_gas_profile(radius=radius)
        radius, P_r = self.get_pressure_gas_profile(radius=radius)

        # Get Temperature
        n_r[n_r <= 0] = np.nan
        T_r = P_r / n_r

        # Apply truncation
        T_r[radius > self._R_truncation] = np.nan
        
        return radius, T_r.to('keV')

    
    #==================================================
    # Get the gas entropy profile
    #==================================================
    
    def get_entropy_gas_profile(self, radius=np.logspace(0,4,1000)*u.kpc):
        """
        Get the entropy profile.
        
        Parameters
        ----------
        - radius (quantity): the physical 3d radius in units homogeneous to kpc, as a 1d array

        Outputs
        ----------
        - radius (quantity): the 3d radius in unit of kpc
        - K_r (quantity): the entropy profile in unit of keV cm2

        """

        # In case the input is not an array
        if type(radius.to_value()) == float:
            radius = np.array([radius.to_value()]) * radius.unit

        # Compute n and P
        radius, n_r = self.get_density_gas_profile(radius=radius)
        radius, P_r = self.get_pressure_gas_profile(radius=radius)

        # Get K
        n_r[n_r <= 0] = np.nan
        K_r = P_r / n_r**(5.0/3)

        # Apply truncation
        K_r[radius > self._R_truncation] = np.nan
        
        return radius, K_r.to('keV cm2')

    
    #==================================================
    # Get the hydrostatic mass profile
    #==================================================

    def get_hse_mass_profile(self, radius=np.logspace(0,4,1000)*u.kpc):
        """
        Get the hydrostatic mass profile using exact analytical expressions.
        
        Parameters
        ----------
        - radius (quantity): the physical 3d radius in units homogeneous to kpc, as a 1d array
        
        Outputs
        ----------
        - radius (quantity): the 3d radius in unit of kpc
        - Mhse_r (quantity): the hydrostatic mass profile in unit of Msun

        """

        # In case the input is not an array
        if type(radius.to_value()) == float:
            radius = np.array([radius.to_value()]) * radius.unit
        
        #---------- Mean molecular weights
        mu_gas, mu_e, mu_p, mu_alpha = cluster_global.mean_molecular_weight(self._helium_mass_fraction)

        #---------- Get the electron density profile
        radius, n_r = self.get_density_gas_profile(radius=radius)

        #---------- Get dP/dr
        # Case of GNFW profile
        if self._pressure_gas_model['name'] == 'GNFW':
            P0 = self._pressure_gas_model["P_0"].to_value('keV cm-3')
            rp = self._pressure_gas_model["r_p"].to_value('kpc')
            a  = self._pressure_gas_model["a"]
            b  = self._pressure_gas_model["b"]
            c  = self._pressure_gas_model["c"]
            r3d_kpc = radius.to_value('kpc')
            dpdr_r = cluster_profile.gNFW_model_derivative(r3d_kpc, P0, rp, slope_a=a, slope_b=b, slope_c=c) * u.Unit('keV cm-3 kpc-1')
            dpdr_r[radius > self._R_truncation] *= 0

        # Case of isoT model
        elif self._pressure_gas_model['name'] == 'isoT':

            # Case of SVM profile
            if self._density_gas_model['name'] == 'SVM':
                n0      = self._density_gas_model["n_0"].to_value('cm-3')
                rc      = self._density_gas_model["r_c"].to_value('kpc')
                rs      = self._density_gas_model["r_s"].to_value('kpc')
                alpha   = self._density_gas_model["alpha"]
                beta    = self._density_gas_model["beta"]
                gamma   = self._density_gas_model["gamma"]
                epsilon = self._density_gas_model["epsilon"]
                r3d_kpc = radius.to_value('kpc')
                dndr_r = cluster_profile.svm_model_derivative(r3d_kpc, n0, rc, beta, rs, gamma, epsilon, alpha) * u.Unit('cm-3 kpc-1')
                dndr_r[radius > self._R_truncation] *= 0

                # beta model
            elif self._density_gas_model['name'] == 'beta':
                n0      = self._density_gas_model["n_0"].to_value('cm-3')
                rc      = self._density_gas_model["r_c"].to_value('kpc')
                beta    = self._density_gas_model["beta"]
                r3d_kpc = radius.to_value('kpc')
                dndr_r = cluster_profile.beta_model_derivative(r3d_kpc, n0, rc, beta) * u.Unit('cm-3 kpc-1')
                dndr_r[radius > self._R_truncation] *= 0

                # double beta model
            elif self._density_gas_model['name'] == 'doublebeta':
                n01      = self._density_gas_model["n_01"].to_value('cm-3')
                rc1      = self._density_gas_model["r_c1"].to_value('kpc')
                beta1    = self._density_gas_model["beta1"]
                n02      = self._density_gas_model["n_02"].to_value('cm-3')
                rc2      = self._density_gas_model["r_c2"].to_value('kpc')
                beta2    = self._density_gas_model["beta2"]
                r3d_kpc = radius.to_value('kpc')
                dndr_r1 = cluster_profile.beta_model_derivative(r3d_kpc, n01, rc1, beta1) * u.Unit('cm-3 kpc-1')
                dndr_r2 = cluster_profile.beta_model_derivative(r3d_kpc, n02, rc2, beta2) * u.Unit('cm-3 kpc-1')
                dndr_r = dndr_r1 + dndr_r2
                dndr_r[radius > self._R_truncation] *= 0

            # Otherwise nothing is done
            else :
                if not self._silent: print('Only the SVM, beta and doublebeta density profile are available for now.')

            dpdr_r = (dndr_r * self._pressure_gas_model['T']).to('keV cm-3 kpc-1')

        # Otherwise nothing is done
        else :
            if not self._silent: print('Only the GNFW and isoT pressure profile is available for now.')

        #---------- Compute the mass
        n_r[n_r <= 0] = np.nan
        Mhse_r = -radius**2 / n_r * dpdr_r / (mu_gas*const.m_p*const.G)
        
        Mhse_r[radius > self._R_truncation] = np.nan
        
        return radius, Mhse_r.to('Msun')


    #==================================================
    # Get the overdensity contrast profile
    #==================================================

    def get_overdensity_contrast_profile(self, radius=np.logspace(0,4,1000)*u.kpc):
        """
        Get the overdensity contrast profile.
        
        Parameters
        ----------
        - radius (quantity): the physical 3d radius in units homogeneous to kpc, as a 1d array
        
        Outputs
        ----------
        - radius (quantity): the 3d radius in unit of kpc
        - delta_r: the overdensity contrast profile

        """

        # In case the input is not an array
        if type(radius.to_value()) == float:
            radius = np.array([radius.to_value()]) * radius.unit

        # Compute delta from the mass profile
        r, mhse = self.get_hse_mass_profile(radius)       
        delta_r = mhse/(1.0-self._hse_bias) / (4.0/3.0*np.pi*radius**3 * self._cosmo.critical_density(self._redshift))

        return radius, delta_r.to_value('')*u.adu


    #==================================================
    # Get the R500 profile
    #==================================================

    def get_mdelta_from_profile(self, delta=500, Rmin=10*u.kpc, Rmax=1e4*u.kpc):
        """
        Get R_delta and M_delta from the overdensity profile, given HSE equilibrium and HSE bias.
        
        Parameters
        ----------
        - delta : the overdensity considered, e.g. 2500, 500, 200
        - Rmin (quantity): the minimal range to search for Rdelta
        - Rmax (quantity): the maximal range to search for Rdelta
        
        Outputs
        ----------
        - Rdelta (quantity): the radius within which delta times the critical density is enclose, e.g. R500
        - Mdelta (quantity): the mass corresponding to Rdelta
        
        """

        # defines the function where to search for roots
        def overdensity_delta_difference(rkpc):
            rod, od = self.get_overdensity_contrast_profile(rkpc*u.kpc)
            func = od.to_value('adu') - delta
            func[np.isnan(od)] = -1.0 # Undefined overdensity (e.g. if truncation) should not be the root
            return func

        # Search the root
        Rdelta = brentq(overdensity_delta_difference, Rmin.to_value('kpc'), Rmax.to_value('kpc'))

        # In case the root is >= R_truncation, Rdelta was not reached
        if Rdelta >= self._R_truncation.to_value('kpc'):
            if not self._silent: print('The truncation was reached before R'+str(delta))
            Rdelta = np.nan

        # Get Mdelta as well
        Mdelta = cluster_global.Rdelta_to_Mdelta(Rdelta, self._redshift, delta=delta, cosmo=self._cosmo)
    
        return Rdelta*u.kpc, Mdelta*u.Msun


    #==================================================
    # Compute Mgas
    #==================================================
    
    def get_gas_mass_profile(self, radius=np.logspace(0,4,1000)*u.kpc):
        """
        Get the gas mass profile.
        
        Parameters
        ----------
        - radius (quantity) : the physical 3d radius in units homogeneous to kpc, as a 1d array

        Outputs
        ----------
        - radius (quantity): the 3d radius in unit of kpc
        - Mgas_r (quantity): the gas mass profile 
        
        """

        #---------- In case the input is not an array
        if type(radius.to_value()) == float:
            radius = np.array([radius.to_value()]) * radius.unit

        #---------- Mean molecular weights
        mu_gas, mu_e, mu_p, mu_alpha = cluster_global.mean_molecular_weight(self._helium_mass_fraction)

        #---------- Define radius associated to the density
        dens_radius = cluster_profile.define_safe_radius_array(radius.to_value('kpc'), Rmin=1.0, Nptmin=1000)*u.kpc
        
        #---------- Get the density profile
        rad, n_r = self.get_density_gas_profile(radius=dens_radius)

        #---------- Integrate the mass
        I_n_gas_r = np.zeros(len(radius))
        for i in range(len(radius)):
            I_n_gas_r[i] = cluster_profile.get_volume_any_model(rad.to_value('kpc'), n_r.to_value('cm-3'),
                                                                radius.to_value('kpc')[i], Npt=1000)

        Mgas_r = mu_e*const.m_p * I_n_gas_r*u.Unit('cm-3 kpc3')

        return radius, Mgas_r.to('Msun')

    
    #==================================================
    # Compute fgas
    #==================================================

    def get_fgas_profile(self, radius=np.logspace(0,4,1000)*u.kpc):
        """
        Get the gas fraction profile.
        
        Parameters
        ----------
        - radius : the physical 3d radius in units homogeneous to kpc, as a 1d array

        Outputs
        ----------
        - radius (quantity): the 3d radius in unit of kpc
        - fgas_r (quantity): the gas mass profile 

        """

        # In case the input is not an array
        if type(radius.to_value()) == float:
            radius = np.array([radius.to_value()]) * radius.unit

        # Compute fgas from Mgas and Mhse
        r, mgas = self.get_gas_mass_profile(radius)
        r, mhse = self.get_hse_mass_profile(radius)       
        fgas_r = mgas.to_value('Msun') / mhse.to_value('Msun') * (1.0 - self._hse_bias) 

        return radius, fgas_r*u.adu

    
    #==================================================
    # Compute thermal energy
    #==================================================
    
    def get_thermal_energy_profile(self, radius=np.logspace(0,4,1000)*u.kpc):
        """
        Compute the thermal energy profile
        
        Parameters
        ----------
        - radius (quantity) : the physical 3d radius in units homogeneous to kpc, as a 1d array

        Outputs
        ----------
        - Uth (quantity) : the thermal energy, in GeV

        """
        
        # In case the input is not an array
        if type(radius.to_value()) == float:
            radius = np.array([radius.to_value()]) * radius.unit

        #---------- Mean molecular weights
        mu_gas, mu_e, mu_p, mu_alpha = cluster_global.mean_molecular_weight(self._helium_mass_fraction)

        #---------- Define radius associated to the pressure
        press_radius = cluster_profile.define_safe_radius_array(radius.to_value('kpc'), Rmin=1.0, Nptmin=1000)*u.kpc
        
        #---------- Get the density profile
        rad, p_r = self.get_pressure_gas_profile(radius=press_radius)
        u_gas = (3.0/2.0)*(mu_e/mu_gas) * p_r # Gas energy density (non-relativistic limit)
        
        #---------- Integrate the pressure in 3d
        Uth_r = np.zeros(len(radius))
        for i in range(len(radius)):
            Uth_r[i] = cluster_profile.get_volume_any_model(rad.to_value('kpc'), u_gas.to_value('keV cm-3'),
                                                            radius.to_value('kpc')[i], Npt=1000)
        
        Uth = Uth_r*u.Unit('keV cm-3 kpc3')

        return radius, Uth.to('erg')


    #==================================================
    # Compute Ysph
    #==================================================

    def get_ysph_profile(self, radius=np.logspace(0,4,1000)*u.kpc):
        """
        Get the spherically integrated Compton parameter profile.
        
        Parameters
        ----------
        - radius (quantity) : the physical 3d radius in units homogeneous to kpc, as a 1d array

        Outputs
        ----------
        - radius (quantity): the 3d radius in unit of kpc
        - Ysph_r (quantity): the integrated Compton parameter (homogeneous to kpc^2)

        """

        # In case the input is not an array
        if type(radius.to_value()) == float:
            radius = np.array([radius.to_value()]) * radius.unit

        #---------- Define radius associated to the pressure
        press_radius = cluster_profile.define_safe_radius_array(radius.to_value('kpc'), Rmin=1.0, Nptmin=1000)*u.kpc
        
        #---------- Get the density profile
        rad, p_r = self.get_pressure_gas_profile(radius=press_radius)

        #---------- Integrate the pressure in 3d
        I_p_gas_r = np.zeros(len(radius))
        for i in range(len(radius)):
            I_p_gas_r[i] = cluster_profile.get_volume_any_model(rad.to_value('kpc'), p_r.to_value('keV cm-3'),
                                                                radius.to_value('kpc')[i], Npt=1000)
        
        Ysph_r = const.sigma_T/(const.m_e*const.c**2) * I_p_gas_r*u.Unit('keV cm-3 kpc3')

        return radius, Ysph_r.to('kpc2')


    #==================================================
    # Compute Ycyl
    #==================================================

    def get_ycyl_profile(self, radius=np.logspace(0,4,1000)*u.kpc, NR500max=5.0, Npt_los=100):
        """
        Get the integrated cylindrical Compton parameter profile.
        
        Parameters
        ----------
        - radius (quantity) : the physical 3d radius in units homogeneous to kpc, as a 1d array
        - NR500max (float): the integration will stop at NR500max x R500
        - Npt_los (int): the number of points for line of sight integration
        
        Outputs
        ----------
        - radius (quantity): the 3d radius in unit of kpc
        - Ycyl_r : the integrated Compton parameter

        """

        # In case the input is not an array
        if type(radius.to_value()) == float:
            radius = np.array([radius.to_value()]) * radius.unit

        #---------- Define radius associated to the Compton parameter
        y_radius = cluster_profile.define_safe_radius_array(radius.to_value('kpc'), Rmin=1.0, Nptmin=1000)*u.kpc
        
        #---------- Get the Compton parameter profile
        r2d, y_r = self.get_y_compton_profile(y_radius, NR500max=NR500max, Npt_los=Npt_los)

        #---------- Integrate the Compton parameter in 2d
        Ycyl_r = np.zeros(len(radius))
        for i in range(len(radius)):
            Ycyl_r[i] = cluster_profile.get_surface_any_model(r2d.to_value('kpc'), y_r.to_value('adu'),
                                                              radius.to_value('kpc')[i], Npt=1000)
        
        return radius, Ycyl_r*u.Unit('kpc2')

    
    #==================================================
    # Compute y profile
    #==================================================
    
    def get_y_compton_profile(self, radius=np.logspace(0,4,1000)*u.kpc, NR500max=5.0, Npt_los=100):
        """
        Get the Compton parameter profile.
        
        Parameters
        ----------
        - radius (quantity): the physical 3d radius in units homogeneous to kpc, as a 1d array
        - NR500max (float): the integration will stop at NR500max x R500
        - Npt_los (int): the number of points for line of sight integration
        
        Outputs
        ----------
        - radius (quantity): the projected 2d radius in unit of kpc
        - y_r : the Compton parameter

        Note
        ----------
        The pressure profile is truncated at R500 along the line-of-sight.

        """
        
        # In case the input is not an array
        if type(radius.to_value()) == float:
            radius = np.array([radius.to_value()]) * radius.unit

        # Define radius associated to the pressure
        p_radius = cluster_profile.define_safe_radius_array(radius.to_value('kpc'),
                                                            Rmin=1.0, Rmax=NR500max*self._R500.to_value('kpc'),
                                                            Nptmin=1000)*u.kpc
        
        # Get the pressure profile
        rad3d, p_r = self.get_pressure_gas_profile(radius=p_radius)

        # Project it
        Rmax = np.amax(NR500max*self._R500.to_value('kpc'))  # Max radius to integrate in 3d
        Rpmax = np.amax(radius.to_value('kpc'))              # Max radius to which we get the profile
        Rproj, Pproj = cluster_profile.proj_any_model(rad3d.to_value('kpc'), p_r.to_value('keV cm-3'),
                                                      Npt=Npt_los, Rmax=Rmax, Rpmax=Rpmax, Rp_input=radius.to_value('kpc'))
        
        # Get the Compton parameter
        Rproj *= u.kpc
        y_compton = Pproj*u.Unit('keV cm-3 kpc') * const.sigma_T/(const.m_e*const.c**2)


        # Apply truncation in case
        y_compton[Rproj > self._R_truncation] = 0.0
        
        return Rproj.to('kpc'), y_compton.to_value('')*u.adu

    
    #==================================================
    # Compute y map 
    #==================================================
    
    def get_ymap(self, FWHM=None, NR500max=5.0, Npt_los=100):
        """
        Compute a Compton parameter ymap.
        
        Parameters
        ----------
        - FWHM (quantity) : the beam smoothing FWHM (homogeneous to deg)
        - NR500max (float): the integration will stop at NR500max x R500
        - Npt_los (int): the number of points for line of sight integration

        Outputs
        ----------
        - ymap (adu) : the Compton parameter map

        """
        
        # Get the header
        header = self.get_map_header()
        w = WCS(header)
        if w.wcs.has_cd():
            if w.wcs.cd[1,0] != 0 or w.wcs.cd[0,1] != 0:
                print('!!! WARNING: R.A and Dec. is rotated wrt x and y. The extracted resolution was not checked in such situation.')
            map_reso_x = np.sqrt(w.wcs.cd[0,0]**2 + w.wcs.cd[1,0]**2)
            map_reso_y = np.sqrt(w.wcs.cd[1,1]**2 + w.wcs.cd[0,1]**2)
        else:
            map_reso_x = np.abs(w.wcs.cdelt[0])
            map_reso_y = np.abs(w.wcs.cdelt[1])
        
        # Get a R.A-Dec. map
        ra_map, dec_map = map_tools.get_radec_map(header)

        # Get a cluster distance map
        dist_map = map_tools.greatcircle(ra_map, dec_map, self._coord.icrs.ra.to_value('deg'), self._coord.icrs.dec.to_value('deg'))

        # Define the radius used fo computing the Compton parameter profile
        theta_max = np.amax(dist_map) # maximum angle from the cluster
        theta_min = np.amin(dist_map) # minimum angle from the cluster (~0 if cluster within FoV)
        if theta_min == 0:
            theta_min = 1e-4 # Zero will cause bug, put <1arcsec in this case
        rmax = theta_max*np.pi/180 * self._D_ang.to_value('kpc')
        rmin = theta_min*np.pi/180 * self._D_ang.to_value('kpc')
        radius = np.logspace(np.log10(rmin), np.log10(rmax), 1000)*u.kpc

        # Compute the Compton parameter projected profile
        r_proj, y_profile = self.get_y_compton_profile(radius, NR500max=NR500max, Npt_los=Npt_los) # kpc, [y]
        theta_proj = (r_proj/self._D_ang).to_value('')*180.0/np.pi                                 # degrees
        
        # Interpolate the profile onto the map
        ymap = map_tools.profile2map(y_profile.to_value('adu'), theta_proj, dist_map)
        
        # Avoid numerical residual ringing from interpolation
        ymap[dist_map > self._theta_truncation.to_value('deg')] = 0
        
        # Smooth the ymap if needed
        if FWHM != None:
            FWHM2sigma = 1.0/(2.0*np.sqrt(2*np.log(2)))
            ymap = ndimage.gaussian_filter(ymap, sigma=(FWHM2sigma*FWHM.to_value('deg')/map_reso_x,
                                                        FWHM2sigma*FWHM.to_value('deg')/map_reso_y), order=0)

        return ymap*u.adu

    
    #==================================================
    # Get normalized CR density profile
    #==================================================
    
    def get_normed_density_crp_profile(self, radius=np.logspace(0,4,1000)*u.kpc):
        """
        Get the normalized cosmic ray proton density profile.
        
        Parameters
        ----------
        - radius (quantity) : the physical 3d radius in units homogeneous to kpc, as a 1d array

        Outputs
        ----------
        - radius (quantity): the 3d radius in unit of kpc
        - n_r (quantity): the normalized density profile, unitless

        """

        # In case the input is not an array
        if type(radius.to_value()) == float:
            radius = np.array([radius.to_value()]) * radius.unit

        # Case of SVM profile
        if self._density_crp_model['name'] == 'SVM':
            rc      = self._density_crp_model["r_c"].to_value('kpc')
            rs      = self._density_crp_model["r_s"].to_value('kpc')
            alpha   = self._density_crp_model["alpha"]
            beta    = self._density_crp_model["beta"]
            gamma   = self._density_crp_model["gamma"]
            epsilon = self._density_crp_model["epsilon"]
            r3d_kpc = radius.to_value('kpc')
            n_r = cluster_profile.svm_model(r3d_kpc, 1.0, rc, beta, rs, gamma, epsilon, alpha)
            
            n_r[radius > self._R_truncation] *= 0
            return radius, n_r*u.adu

        # beta model
        elif self._density_crp_model['name'] == 'beta':
            rc      = self._density_crp_model["r_c"].to_value('kpc')
            beta    = self._density_crp_model["beta"]
            r3d_kpc = radius.to_value('kpc')
            n_r = cluster_profile.beta_model(r3d_kpc, 1.0, rc, beta)
            
            n_r[radius > self._R_truncation] *= 0
            return radius, n_r*u.adu

        # double beta model
        elif self._density_crp_model['name'] == 'doublebeta':
            n01      = self._density_crp_model["n_01"]
            rc1      = self._density_crp_model["r_c1"].to_value('kpc')
            beta1    = self._density_crp_model["beta1"]
            n02      = self._density_crp_model["n_02"]
            rc2      = self._density_crp_model["r_c2"].to_value('kpc')
            beta2    = self._density_crp_model["beta2"]
            r3d_kpc  = radius.to_value('kpc')
            n_r1 = cluster_profile.beta_model(r3d_kpc, n01/(n01+n02), rc1, beta1)
            n_r2 = cluster_profile.beta_model(r3d_kpc, n02/(n01+n02), rc2, beta2)
            n_r = n_r1 + n_r2
            
            n_r[radius > self._R_truncation] *= 0
            return radius, n_r*u.adu

        # GNFW model
        elif self._density_crp_model['name'] == 'GNFW':
            a       = self._density_crp_model["a"]
            b       = self._density_crp_model["b"]
            c       = self._density_crp_model["c"]
            rp      = self._density_crp_model["r_p"].to_value('kpc')
            r3d_kpc = radius.to_value('kpc')
            n_r = cluster_profile.gNFW_model(r3d_kpc, 1.0, rp, slope_a=a, slope_b=b, slope_c=c)
            
            n_r[radius > self._R_truncation] *= 0
            return radius, n_r*u.adu
        
        # Otherwise nothing is done
        else :
            if not self._silent: print('Only the SVM, beta, doublebeta and GNFW density profile are available for now.')


    #==================================================
    # Get normalized CR density profile
    #==================================================
    
    def get_normed_crp_spectrum(self, energy=np.logspace(-2,7,1000)*u.GeV):
        """
        Get the normalized cosmic ray proton spectrum.
        
        Parameters
        ----------
        - energy (quantity) : the physical energy of CR protons

        Outputs
        ----------
        - radius (quantity): the 3d radius in unit of kpc
        - S_E (quantity): the normalized spectrum profile, unitless

        """

        # In case the input is not an array
        if type(energy.to_value()) == float:
            energy = np.array([energy.to_value()]) * energy.unit

        # Case of Power Law spectrum
        if self._spectrum_crp_model['name'] == 'PowerLaw':
            index   = self._spectrum_crp_model["Index"]
            eng_GeV = energy.to_value('GeV')
            S_E = cluster_spectra.powerlaw_model(eng_GeV, 1.0, index)
            S_E[energy > self._Epmax] *= 0
            S_E[energy < self._Epmin] *= 0
            return energy, S_E*u.adu

        # Case of Exponential Cutoff Power Law
        elif self._spectrum_crp_model['name'] == 'ExponentialCutoffPowerLaw':
            index   = self._spectrum_crp_model["Index"]
            Ecut   = self._spectrum_crp_model["CutoffEnergy"].to_value('GeV')
            eng_GeV = energy.to_value('GeV')
            S_E = cluster_spectra.exponentialcutoffpowerlaw_model(eng_GeV, 1.0, index, Ecut)
            S_E[energy > self._Epmax] *= 0
            S_E[energy < self._Epmin] *= 0
            return energy, S_E*u.adu

        # Otherwise nothing is done
        else :
            if not self._silent: print('Only the PowerLaw and ExponentialCutoffPowerLaw spectrum are available for now.')


    #==================================================
    # Get the CR proton normalization
    #==================================================
    
    def get_crp_normalization(self):
        """
        Compute the normalization of the cosmic ray proton distribution:
        dN/dE/dV = Norm f(E) f(r)
        with f(E) the spectral form and f(r) the spatial form
        Norm is in unit of GeV-1 cm-3
        
        Parameters
        ----------

        Outputs
        ----------
        - Norm (quantity): in unit of GeV-1 cm-3

        """

        Rcut = self._X_cr['Rcut']
        
        # Get the thermal energy
        rad_uth, U_th = self.get_thermal_energy_profile(Rcut)
        
        # Get the spatial form volume
        r3d = cluster_profile.define_safe_radius_array(np.array([Rcut.to_value('kpc')]), Rmin=1.0)*u.kpc
        radius, f_cr_r = self.get_normed_density_crp_profile(r3d)
        Vcr = cluster_profile.get_volume_any_model(radius.to_value('kpc'), f_cr_r.to_value('adu'), Rcut.to_value('kpc')) * u.Unit('kpc3')
        
        # Get the energy enclosed in the spectrum
        energy = np.logspace(np.log10(self._Epmin.to_value('GeV')), np.log10(self._Epmax.to_value('GeV')), 1000) * u.GeV
        eng, spectrum = self.get_normed_crp_spectrum(energy)
        Ienergy = cluster_spectra.get_integral_any_model(eng.to_value('GeV'), eng.to_value('GeV')*spectrum.to_value('adu'),
                                                         self._Epmin.to_value('GeV'), self._Epmax.to_value('GeV')) * u.GeV**2
        
        # Compute the normalization
        Norm = self._X_cr['X'] * U_th / Vcr / Ienergy

        return Norm.to('GeV-1 cm-3')

    
    #==================================================
    # Get the CR proton density profile
    #==================================================
    
    def get_density_crp_profile(self, radius=np.logspace(0,4,1000)*u.kpc, Emin=None, Emax=None, Energy_density=False):
        """
        Compute the cosmic ray proton density profile, integrating energies 
        between Emin and Emax.
        
        Parameters
        ----------
        - radius (quantity): the physical 3d radius in units homogeneous to kpc, as a 1d array
        - Emin (quantity): the lower bound for energy integration
        - Emax (quantity): the upper bound for energy integration
        - Energy_density (bool): if True, then the energy density is computed. Otherwise, 
        the number density is computed.

        Outputs
        ----------
        - density (quantity): in unit of cm-3

        """
        
        if Emin == None:
            Emin = self._Epmin
        if Emax == None:
            Emax = self._Epmax
            
        # Get the normalization
        norm = self.get_crp_normalization()
        
        # Get the radial form
        rad, f_r = self.get_normed_density_crp_profile(radius)
        
        # Get the energy enclosed in the spectrum
        energy = np.logspace(np.log10(Emin.to_value('GeV')), np.log10(Emax.to_value('GeV')), 1000) * u.GeV
        eng, spectrum = self.get_normed_crp_spectrum(energy)

        if Energy_density:
            Ienergy = cluster_spectra.get_integral_any_model(eng.to_value('GeV'), eng.to_value('GeV')*spectrum.to_value('adu'),
                                                             Emin.to_value('GeV'), Emax.to_value('GeV')) * u.GeV**2
            density = (norm * f_r.to_value('adu') * Ienergy).to('GeV cm-3')            
        else:
            Ienergy = cluster_spectra.get_integral_any_model(eng.to_value('GeV'), spectrum.to_value('adu'),
                                                             Emin.to_value('GeV'), Emax.to_value('GeV')) * u.GeV
            density = (norm * f_r.to_value('adu') * Ienergy).to('cm-3')            
            
        return radius, density

    
    #==================================================
    # Get the CR proton spectrum
    #==================================================
    
    def get_crp_spectrum(self, energy=np.logspace(-2,7,1000)*u.GeV, Rmax=None):
        """
        Compute the cosmic ray proton spectrum, integrating radius 
        between 0 and Rmax.
        
        Parameters
        ----------
        - energy (quantity) : the physical energy of CR protons
        - Rmax (quantity): the radius within with the spectrum is computed 
        (default is R500)

        Outputs
        ----------
        - spectrum (quantity): in unit of GeV-1

        """

        if Rmax == None:
            Rmax = self._R500
            
        # Get the normalization
        norm = self.get_crp_normalization()
        
        # Get the spatial form volume
        r3d = cluster_profile.define_safe_radius_array(np.array([Rmax.to_value('kpc')]), Rmin=1.0)*u.kpc
        radius, f_cr_r = self.get_normed_density_crp_profile(r3d)
        Vcr = cluster_profile.get_volume_any_model(radius.to_value('kpc'), f_cr_r.to_value('adu'),
                                                   Rmax.to_value('kpc')) * u.Unit('kpc3')

        # Get the energy form
        eng, f_E = self.get_normed_crp_spectrum(energy)
        
        # compute the spectrum
        spectrum = (norm * Vcr * f_E.to_value('adu')).to('GeV-1')

        return energy, spectrum.to('GeV-1')

    
    #==================================================
    # Get the CR to thermal energy profile
    #==================================================
    
    def get_crp_to_thermal_energy_profile(self, radius=np.logspace(0,4,1000)*u.kpc, Emin=None, Emax=None):
        """
        Compute the X_cr profile, i.e. the cosmic ray to thermal energy enclosed within R
        profile.
        
        Parameters
        ----------
        - radius (quantity): the physical 3d radius in units homogeneous to kpc, as a 1d array
        - Emin (quantity): the lower bound for energy integration
        - Emax (quantity): the upper bound for energy integration

        Outputs
        ----------
        - x_r (np.ndarray): the profile

        """

        # In case the input is not an array
        if type(radius.to_value()) == float:
            radius = np.array([radius.to_value()]) * radius.unit

        # Thermal energy
        r_uth, Uth_r = self.get_thermal_energy_profile(radius)

        # CR energy density profile
        r3d = cluster_profile.define_safe_radius_array(radius.to_value('kpc'), Rmin=1.0)*u.kpc
        r_cr, e_cr = self.get_density_crp_profile(r3d, Emin=Emin, Emax=Emax, Energy_density=True)

        # Integrate CR energy density profile
        Ucr_r = np.zeros(len(radius))
        for i in range(len(radius)):
            Ucr_r[i] = cluster_profile.get_volume_any_model(r_cr.to_value('kpc'), e_cr.to_value('GeV cm-3'),
                                                            radius.to_value('kpc')[i], Npt=1000)

        U_cr = Ucr_r * u.Unit('GeV cm-3 kpc3')
        
        # X(<R)
        x_r = U_cr.to_value('GeV') / Uth_r.to_value('GeV')

        return radius, x_r*u.adu

    
    #==================================================
    # Compute gamma ray spectrum
    #==================================================
    
    def get_gamma_spectrum(self, energy=np.logspace(-2,6,1000)*u.GeV, Rmax=None,
                           type_integral='spherical', NR500max=5.0, Npt_los=100):
        """
        Compute the gamma ray emission enclosed within Rmax, in 3d (i.e. spherically 
        integrated), or the gamma ray emmission enclosed within an circular area (i.e.
        cylindrical).
        
        Parameters
        ----------
        - energy (quantity) : the physical energy of gamma rays
        - Rmax (quantity): the radius within with the spectrum is computed 
        (default is R500)
        - type_integral (string): either 'spherical' or 'cylindrical'
        - NR500max (float): the line-of-sight integration will stop at NR500max x R500. 
        This is used only for cylindrical case
        - Npt_los (int): the number of points for line of sight integration.
        This is used only for cylindrical case

        Outputs
        ----------
        - energy (quantity) : the physical energy of gamma rays
        - dN_dEdSdt (np.ndarray) : the spectrum in units of MeV-1 cm-2 s-1

        """
        if np.amin(energy) < 9.9999*u.MeV or np.amax(energy) > 1.00001*u.PeV:
            print('!!! WARNING: outside of 10MeV-1PeV, Naima appears to return wrong spectra (flat).')
            print('    E_min - Emax range : '+str(np.amin(energy.to_value('MeV')))+' MeV - '+str(np.amax(energy.to_value('PeV')))+' PeV')
            
        if Rmax == None:
            Rmax = self._R500
        
        # Define a cosmic ray proton object normalized to 1 GeV-1 at E_CR = 1 GeV
        if self._spectrum_crp_model['name'] == 'PowerLaw':
            CRp = naima.models.PowerLaw(1.0/u.GeV, 1.0*u.GeV, self._spectrum_crp_model['Index'])
        elif self._spectrum_crp_model['name'] == 'ExponentialCutoffPowerLaw':
            CRp = naima.models.ExponentialCutoffPowerLaw(1.0/u.GeV, 1.0*u.GeV,
                                                          self._spectrum_crp_model['Index'], self._spectrum_crp_model['PivotEnergy'])
        else:
            raise ValueError("The available spectra are PowerLaw and ExponentialCutoffPowerLaw for now")

        # Get the pion decay model for 1 GeV-1 CRp in the volume and for 1 cm-3 of thermal gas
        gamma = naima.models.PionDecay(CRp, nh=1.0*u.Unit('cm**-3'), nuclear_enhancement=self._nuclear_enhancement)

        # Normalize the energy of CRp in the Volume (the choice of R is arbitrary)
        CRenergy_Rcut = self._X_cr['X'] * self.get_thermal_energy_profile(self._X_cr['Rcut'])[1][0]
        gamma.set_Wp(CRenergy_Rcut, Epmin=self._Epmin, Epmax=self._Epmax)

        # Compute the normalization volume and the integration cross density volume
        r3d1 = cluster_profile.define_safe_radius_array(np.array([self._X_cr['Rcut'].to_value('kpc')]), Rmin=1.0)*u.kpc
        radius1, f_crp_r1 = self.get_normed_density_crp_profile(r3d1)
        V_CRenergy = cluster_profile.get_volume_any_model(radius1.to_value('kpc'), f_crp_r1.to_value('adu'),
                                                          self._X_cr['Rcut'].to_value('kpc'))*u.kpc**3

        #---------- Compute the integral spherical volume
        if type_integral == 'spherical':
            r3d2 = cluster_profile.define_safe_radius_array(np.array([Rmax.to_value('kpc')]), Rmin=1.0)*u.kpc
            radius2, n_gas_r2  = self.get_density_gas_profile(r3d2)
            radius2, f_crp_r2 = self.get_normed_density_crp_profile(r3d2)
            
            V_ncr_ngas = cluster_profile.get_volume_any_model(radius2.to_value('kpc'),
                                                              n_gas_r2.to_value('cm-3')*f_crp_r2.to_value('adu'),
                                                              Rmax.to_value('kpc'))*u.kpc**3
        
        #---------- Compute the integral sperical volume
        elif type_integral == 'cylindrical':
            Rlosmax = np.amax(NR500max*self._R500.to_value('kpc'))  # Max radius to integrate in 3d
            Rpmax = Rmax.to_value('kpc')                            # Max radius to which we get the profile
            
            # First project the integrand
            r3d2 = cluster_profile.define_safe_radius_array(np.array([Rlosmax]), Rmin=1.0)*u.kpc
            radius2, n_gas_r2  = self.get_density_gas_profile(r3d2)
            radius2, f_crp_r2 = self.get_normed_density_crp_profile(r3d2)

            r2d3 = cluster_profile.define_safe_radius_array(np.array([Rpmax]), Rmin=1.0)*u.kpc
            Rproj, Pproj = cluster_profile.proj_any_model(radius2.to_value('kpc'), n_gas_r2.to_value('cm-3')*f_crp_r2.to_value('adu'),
                                                          Npt=Npt_los, Rmax=Rlosmax, Rpmax=Rpmax, Rp_input=r2d3.to_value('kpc'))
            Rproj *= u.kpc
            Pproj *= u.kpc
            
            # Then compute the integral cylindrical volume
            V_ncr_ngas = cluster_profile.get_surface_any_model(Rproj.to_value('kpc'), Pproj.to_value('kpc'),
                                                               Rmax.to_value('kpc'), Npt=1000)*u.kpc**3
            
        #---------- Error case
        else:
            raise ValueError('Only spherical or cylindrical options are available.')    
            
        # Compute the spectrum, within Rcut, assuming 1cm-3 gas, normalize by the energy computation volume and multiply by volume term
        dN_dEdSdt = (V_ncr_ngas/V_CRenergy).to_value('')*gamma.flux(energy, distance=self._D_lum).to('MeV-1 cm-2 s-1')

        return energy, dN_dEdSdt.to('MeV-1 cm-2 s-1')

    
    #==================================================
    # Compute gamma ray profile
    #==================================================
    
    def get_gamma_profile(self, radius=np.logspace(0,4,1000)*u.kpc, Emin=10.0*u.MeV, Emax=1.0*u.PeV, Energy_density=False,
                          NR500max=5.0, Npt_los=100):
        """
        Compute the gamma ray emission profile within Emin-Emax.
        
        Parameters
        ----------
        - radius (quantity): the physical 3d radius in units homogeneous to kpc, as a 1d array
        - Emin (quantity): the lower bound for gamma ray energy integration
        - Emax (quantity): the upper bound for gamma ray energy integration
        - Energy_density (bool): if True, then the energy density is computed. Otherwise, 
        the number density is computed.
        - NR500max (float): the line-of-sight integration will stop at NR500max x R500. 
        - Npt_los (int): the number of points for line of sight integration.

        Outputs
        ----------
        - dN_dSdtdO (np.ndarray) : the spectrum in units of cm-2 s-1 sr-1 or GeV cm-2 s-1 sr-1

        """

        if Emin < 9.9999*u.MeV or Emax > 1.00001*u.PeV:
            print('!!! WARNING: outside of 10MeV-1PeV, Naima appears to return wrong spectra (flat).')

        
        # Define a cosmic ray proton object normalized to 1 GeV-1 at E_CR = 1 GeV
        if self._spectrum_crp_model['name'] == 'PowerLaw':
            CRp = naima.models.PowerLaw(1.0/u.GeV, 1.0*u.GeV, self._spectrum_crp_model['Index'])
        elif self._spectrum_crp_model['name'] == 'ExponentialCutoffPowerLaw':
            CRp = naima.models.ExponentialCutoffPowerLaw(1.0/u.GeV, 1.0*u.GeV,
                                                          self._spectrum_crp_model['Index'], self._spectrum_crp_model['PivotEnergy'])
        else:
            raise ValueError("The available spectra are PowerLaw and ExponentialCutoffPowerLaw for now")

        # Get the pion decay model for 1 GeV-1 CRp in the volume and for 1 cm-3 of thermal gas
        gamma = naima.models.PionDecay(CRp, nh=1.0*u.Unit('cm**-3'), nuclear_enhancement=self._nuclear_enhancement)

        # Normalize the energy of CRp in the Volume (the choice of R is arbitrary)
        CRenergy_Rcut = self._X_cr['X'] * self.get_thermal_energy_profile(self._X_cr['Rcut'])[1][0]
        gamma.set_Wp(CRenergy_Rcut, Epmin=self._Epmin, Epmax=self._Epmax)

        # Compute the normalization volume and the integration cross density volume
        r3d1 = cluster_profile.define_safe_radius_array(np.array([self._X_cr['Rcut'].to_value('kpc')]), Rmin=1.0)*u.kpc
        radius1, f_crp_r1 = self.get_normed_density_crp_profile(r3d1)
        V_CRenergy = cluster_profile.get_volume_any_model(radius1.to_value('kpc'), f_crp_r1, self._X_cr['Rcut'].to_value('kpc'))*u.kpc**3
        
        # Compute the spectral part and integrate
        energy = np.logspace(np.log10(Emin.to_value('GeV')), np.log10(Emax.to_value('GeV')), 1000)*u.GeV
        dN_dEdSdt = gamma.flux(energy, distance=self._D_lum).to('MeV-1 cm-2 s-1')
        
        if Energy_density:
            dN_dSdt = cluster_spectra.get_integral_any_model(energy.to_value('GeV'),
                                                             energy.to_value('GeV')*dN_dEdSdt.to_value('GeV-1 cm-2 s-1'),
                                                             Emin.to_value('GeV'), Emax.to_value('GeV')) * u.Unit('GeV cm-2 s-1')
        else:
            dN_dSdt = cluster_spectra.get_integral_any_model(energy.to_value('GeV'),
                                                             dN_dEdSdt.to_value('GeV-1 cm-2 s-1'),
                                                             Emin.to_value('GeV'), Emax.to_value('GeV')) * u.Unit('cm-2 s-1')
        
        # Project the integrand
        Rlosmax = np.amax(NR500max*self._R500.to_value('kpc'))  # Max radius to integrate in 3d

        r3d2 = cluster_profile.define_safe_radius_array(np.array([Rlosmax]), Rmin=1.0)*u.kpc
        radius2, n_gas_r2  = self.get_density_gas_profile(r3d2)
        radius2, f_crp_r2 = self.get_normed_density_crp_profile(r3d2)
        
        Rproj, Pproj = cluster_profile.proj_any_model(radius2.to_value('kpc'), n_gas_r2.to_value('cm-3')*f_crp_r2,
                                                      Npt=Npt_los, Rmax=Rlosmax, Rp_input=radius.to_value('kpc'))
        Rproj *= u.kpc
        Pproj *= u.kpc

        # Apply truncation in case
        Pproj[Rproj > self._R_truncation] = 0.0
        
        # Get the final result
        dN_dSdtdO = dN_dSdt / V_CRenergy * self._D_ang**2 * Pproj * u.Unit('sr-1')

        if Energy_density:
            dN_dSdtdO = dN_dSdtdO.to('GeV cm-2 s-1 sr-1')
        else :
            dN_dSdtdO = dN_dSdtdO.to('cm-2 s-1 sr-1')
            
        return Rproj, dN_dSdtdO


    #==================================================
    # Compute gamma ray flux
    #==================================================
    
    def get_gamma_flux(self, Rmax=None, type_integral='spherical', NR500max=5.0, Npt_los=100,
                       Emin=10.0*u.MeV, Emax=1.0*u.PeV, Energy_density=False):
        
        """
        Compute the gamma ray emission enclosed within Rmax, in 3d (i.e. spherically 
        integrated), or the gamma ray emmission enclosed within an circular area (i.e.
        cylindrical), and in a givn energy band.
        
        Parameters
        ----------
        - Rmax (quantity): the radius within with the spectrum is computed 
        (default is R500)
        - type_integral (string): either 'spherical' or 'cylindrical'
        - NR500max (float): the line-of-sight integration will stop at NR500max x R500. 
        This is used only for cylindrical case
        - Npt_los (int): the number of points for line of sight integration.
        This is used only for cylindrical case
        - Emin (quantity): the lower bound for gamma ray energy integration
        - Emax (quantity): the upper bound for gamma ray energy integration
        - Energy_density (bool): if True, then the energy density is computed. Otherwise, 
        the number density is computed.

        Outputs
        ----------
        - flux (quantity) : the gamma ray flux either in GeV/cm2/s or ph/cm2/s, depending
        on parameter Energy_density

        """
        
        energy = np.logspace(np.log10(Emin.to_value('GeV')),np.log10(Emax.to_value('GeV')),1000)*u.GeV
        energy, dN_dEdSdt = self.get_gamma_spectrum(energy, Rmax=Rmax, type_integral=type_integral, NR500max=NR500max, Npt_los=Npt_los)

        if Energy_density:
            flux = cluster_spectra.get_integral_any_model(energy.to_value('GeV'),
                                                          energy.to_value('GeV')*dN_dEdSdt.to_value('GeV-1 cm-2 s-1'),
                                                          Emin.to_value('GeV'), Emax.to_value('GeV')) * u.Unit('GeV cm-2 s-1')
        else:
            flux = cluster_spectra.get_integral_any_model(energy.to_value('GeV'),
                                                          dN_dEdSdt.to_value('GeV-1 cm-2 s-1'),
                                                          Emin.to_value('GeV'), Emax.to_value('GeV')) * u.Unit('cm-2 s-1')
            
        return flux


    #==================================================
    # Compute gamma map
    #==================================================
    
    def get_gamma_template_map(self, NR500max=5.0, Npt_los=100):
        """
        Compute the gamma ray template map. The map is normalized so that the integral 
        of the map over the cluster volume is 1 (up to Rmax=5R500). In case the map is 
        smaller than the cluster extent, the map may not be normalized to one, but the 
        missing
        
        Parameters
        ----------
        - NR500max (float): the integration will stop at NR500max x R500
        - Npt_los (int): the number of points for line of sight integration

        Outputs
        ----------
        gamma_template (np.ndarray) : the map in units of sr-1

        """

        # Get the header
        header = self.get_map_header()

        # Get a R.A-Dec. map
        ra_map, dec_map = map_tools.get_radec_map(header)

        # Get a cluster distance map
        dist_map = map_tools.greatcircle(ra_map, dec_map, self._coord.icrs.ra.to_value('deg'), self._coord.icrs.dec.to_value('deg'))

        # Define the radius used fo computing the Compton parameter profile
        theta_max = np.amax(dist_map) # maximum angle from the cluster
        theta_min = np.amin(dist_map) # minimum angle from the cluster (~0 if cluster within FoV)
        rmax = theta_max*np.pi/180 * self._D_ang.to_value('kpc')
        rmin = theta_min*np.pi/180 * self._D_ang.to_value('kpc')
        radius = np.logspace(np.log10(rmin), np.log10(rmax), 1000)*u.kpc

        # Project the integrand
        Rlosmax = np.amax(NR500max*self._R500.to_value('kpc'))  # Max radius to integrate in 3d

        r3d = cluster_profile.define_safe_radius_array(np.array([Rlosmax]), Rmin=1.0, Nptmin=1000)*u.kpc
        rad, n_gas_r  = self.get_density_gas_profile(r3d)
        rad, f_crp_r = self.get_normed_density_crp_profile(r3d)
        
        r_proj, p_proj = cluster_profile.proj_any_model(rad.to_value('kpc'), n_gas_r.to_value('cm-3')*f_crp_r,
                                                        Npt=Npt_los, Rmax=Rlosmax, Rp_input=radius.to_value('kpc'))

        # Convert to angle and interpolate onto a map
        r_proj *= u.kpc
        theta_proj = (r_proj/self._D_ang).to_value('')*180.0/np.pi   # degrees

        gamma_template = map_tools.profile2map(p_proj, theta_proj, dist_map)

        # Avoid numerical residual ringing from interpolation
        gamma_template[dist_map > self._theta_truncation.to_value('deg')] = 0
        
        # Compute the normalization
        r_proj2, p_proj2 = cluster_profile.proj_any_model(rad.to_value('kpc'), n_gas_r.to_value('cm-3')*f_crp_r,
                                                        Npt=Npt_los, Rmax=Rlosmax, Rp_input=r3d.to_value('kpc'))
        r_proj2 *= u.kpc

        surface = cluster_profile.get_surface_any_model(r_proj2.to_value('kpc'), p_proj2, Rlosmax, Npt=1000)*u.kpc**2
        omega = (surface / self._D_ang**2).to_value('')*u.sr

        # Normalize
        gamma_template_norm = gamma_template / omega
        
        return gamma_template_norm.to('sr-1')

    
    #==================================================
    # Compute a Xspec table versus temperature
    #==================================================
    
    def make_xspec_table(self, Emin=0.1*u.keV, Emax=2.4*u.keV,
                         Tmin=0.1*u.keV, Tmax=50.0*u.keV, nbin=100,
                         nH=0.0/u.cm**2, file_HI=None, visu_nH=False,
                         model='APEC',
                         resp_file=None, data_file=None, app_nH_model=False,
                         Kcor=False):
        """
        Generate an xspec table as a function of temperature, for the cluster.
        This require xspec to be installed, and having a hydrogen column density 
        map in the Healpix format.
        
        Parameters
        ----------
        - Emin (quantity): Xray band minimal energy (RASS Hard is 0.5-2.4 keV, RASS Soft is 0.1-0.4 keV)
        - Emax (quantity): Xray band maximal energy
        - Tmin (quantity): min table temperature (in keV)
        - Tmax (quantity): max table temperature (in keV)
        - nbin (int): number of temperature point in the table (in unit of cm^-2, not 10^22 cm^-2)
        - nH (quantity): H I column density at the cluster
        - file_HI (str): full path to the map file
        - visu_nH (bool): show the nH maps and histogram when extracted from data
        - model (str): which model to use (APEC or MEKAL)
        - resp_file (str): full path to the response file of e.g., ROSAT PSPC
        - data_file (str): full path to any data spectrum file needed for template in xspec 
        (see https://heasarc.gsfc.nasa.gov/FTP/rosat/doc/xselect_guide/xselect_guide_v1.1.1/xselect_ftools.pdf,
        section 5)
        - app_nH_model (bool): apply nH absorbtion to the initial model without instrumental effects
        - Kcor (bool): shift the energy by 1+z to go to the cluster frame

        Outputs
        ----------
        XSPEC table created in the output directory

        """
        
        # Create the output directory if needed
        if not os.path.exists(self._output_dir): os.mkdir(self._output_dir)
        
        # In case we want nH from real data at cluster location
        if file_HI != None :
            # Make sure the FoV and resolution are ok
            if self._map_fov == None or self._map_reso == None:
                fov = 1.0
                reso = 0.1
            else:
                fov = self._map_fov.to_value('deg')
                reso = self._map_reso.to_value('deg')
                
            nH2use, nH2use_err = cluster_xspec.get_nH(file_HI,
                                                      self._coord.icrs.ra.to_value('deg'), self._coord.icrs.dec.to_value('deg'),
                                                      fov=fov, reso=reso,
                                                      save_file=None, visu=visu_nH)
            if nH2use/nH2use_err < 5 :
                print('!!! WARNING, nH is not well constain in the field (S/N < 5) and nH='+str(nH2use)+' 10^22 cm-2.')

        # Or give nH directly
        elif nH != None:
            nH2use = 10**-22 * nH.to_value('cm-2')
        else:
            raise ValueError("nH or file_HI should be provided to get the hydrogen column density.")
            
        # Compute the table
        cluster_xspec.make_xspec_table(self._output_dir+'/XSPEC_table.txt', nH2use, self._abundance, self._redshift,
                                       Emin.to_value('keV'), Emax.to_value('keV'), 
                                       Tmin=Tmin.to_value('keV'), Tmax=Tmax.to_value('keV'), nbin=nbin,
                                       Kcor=Kcor,
                                       file_ana=self._output_dir+'/xspec_analysis.txt',
                                       file_out=self._output_dir+'/xspec_analysis_output.txt',
                                       model=model,
                                       resp_file=resp_file, data_file=data_file, app_nH_model=app_nH_model,
                                       cleanup=True,
                                       logspace=True)
        

    #==================================================
    # Read and interpolate xspec tables
    #==================================================
    
    def itpl_xspec_table(self, xspecfile, Tinput):
        """
        Read an Xspec table and interpolate values at a given temperature

        Parameters
        ----------
        - xspecfile (str): full path to Xspec file to read

        Outputs
        ----------
        - dC (quantity): the interpolated differential counts
        - dS (quantity): the interpolated differential surface brightness counts
        - dR (quantity): the interpolated differential rate

        """
        
        file_start = 3
        
        # Read the temperature
        with open(xspecfile) as f: 
            col = zip(*[line.split() for line in f])[0]
            Txspec = np.array(col[file_start:]).astype(np.float)

        # Read Xspec counts
        with open(xspecfile) as f: 
            col = zip(*[line.split() for line in f])[1]
            Cxspec = np.array(col[file_start:]).astype(np.float)

        # Read Xspec surface brightness
        with open(xspecfile) as f: 
            col = zip(*[line.split() for line in f])[2]
            Sxspec = np.array(col[file_start:]).astype(np.float)

        # Read Xspec rate
        with open(xspecfile) as f: 
            col = zip(*[line.split() for line in f])[3]
            Rxspec = np.array(col[file_start:]).astype(np.float)
        
        # Define interpolation and set unit
        Citpl = interpolate.interp1d(Txspec, Cxspec, kind='cubic', fill_value='extrapolate')
        Sitpl = interpolate.interp1d(Txspec, Sxspec, kind='cubic', fill_value='extrapolate')
        Ritpl = interpolate.interp1d(Txspec, Rxspec, kind='cubic', fill_value='extrapolate')
        
        dCxspec = Citpl(Tinput.to_value('keV')) * 1/u.cm**2/u.s/u.cm**-5
        dSxspec = Sitpl(Tinput.to_value('keV')) * u.erg/u.cm**2/u.s/u.cm**-5
        dRxspec = Ritpl(Tinput.to_value('keV')) * 1/u.s/u.cm**-5
        
        # Take care of undefined temperature (i.e. no gas)
        dCxspec[np.isnan(Tinput.to_value('keV'))] = 0.0            
        dSxspec[np.isnan(Tinput.to_value('keV'))] = 0.0
        dRxspec[np.isnan(Tinput.to_value('keV'))] = 0.0
        if np.sum(~np.isnan(dRxspec)) == 0 :
            dRxspec[:] = np.nan
        
        return dCxspec, dSxspec, dRxspec
        
        
    #==================================================
    # Compute a Xspec table versus temperature
    #==================================================
    
    def get_sx_profile(self, radius=np.logspace(0,4,1000)*u.kpc, NR500max=5.0, Npt_los=100,
                       output_type='S'):
        """
        Compute a surface brightness Xray profile. An xspec table file is needed as 
        output_dir+'/XSPEC_table.txt'.
        
        Parameters
        ----------
        - radius (quantity): the physical 3d radius in units homogeneous to kpc, as a 1d array
        - NR500max (float): the integration will stop at NR500max x R500
        - Npt_los (int): the number of points for line of sight integration
        - output_type (str): type of ooutput to provide: 
        S == energy counts in erg/s/cm^2/sr
        C == counts in ph/s/cm^2/sr
        R == count rate in ph/s/sr (accounting for instrumental response)
        
        Outputs
        ----------
        - Rproj (quantity): the projected 2d radius in unit of kpc
        - Sx (quantity): the Xray surface brightness projectes profile

        """

        # In case the input is not an array
        if type(radius.to_value()) == float:
            radius = np.array([radius.to_value()]) * radius.unit

        # Get the gas density profile
        n_radius = cluster_profile.define_safe_radius_array(radius.to_value('kpc'),
                                                            Rmin=1.0, Rmax=NR500max*self._R500.to_value('kpc'),
                                                            Nptmin=1000)*u.kpc

        # Get the density and temperature profile
        rad3d, n_e  = self.get_density_gas_profile(radius=n_radius)
        rad3d, T_g  = self.get_temperature_gas_profile(radius=n_radius)

        # Interpolate Xspec table
        dC_xspec, dS_xspec, dR_xspec = self.itpl_xspec_table(self._output_dir+'/XSPEC_table.txt', T_g)
        if np.sum(~np.isnan(dR_xspec)) == 0 and output_type == 'R':
            raise ValueError("You ask for an output in ph/s/sr (i.e. including instrumental response), but the xspec table was generated without response file.")

        # Get the integrand
        mu_gas, mu_e, mu_p, mu_alpha = cluster_global.mean_molecular_weight(self._helium_mass_fraction)
        constant = 1e-14/(4*np.pi*self._D_ang**2*(1+self._redshift)**2)

        if output_type == 'S':
            integrand = constant.to_value('kpc-2')*dS_xspec.to_value('erg cm3 s-1') * n_e.to_value('cm-3')**2 * mu_e/mu_p
        elif output_type == 'C':
            integrand = constant.to_value('kpc-2')*dC_xspec.to_value('cm3 s-1') * n_e.to_value('cm-3')**2 * mu_e/mu_p
        elif output_type == 'R':
            integrand = constant.to_value('kpc-2')*dR_xspec.to_value('cm5 s-1') * n_e.to_value('cm-3')**2 * mu_e/mu_p
        else:
            raise ValueError("Output type available are S, C and R.")        
        
        # Projection to get Emmission Measure            
        Rmax = np.amax(NR500max*self._R500.to_value('kpc'))  # Max radius to integrate in 3d
        Rpmax = np.amax(radius.to_value('kpc'))              # Max radius to which we get the profile
        Rproj, Sx = cluster_profile.proj_any_model(rad3d.to_value('kpc'), integrand,
                                                   Npt=Npt_los, Rmax=Rmax, Rpmax=Rpmax, Rp_input=radius.to_value('kpc'))
        Rproj *= u.kpc
        Sx[Rproj > self._R_truncation] = 0.0

        # write unit explicitlly
        if output_type == 'S':
            Sx *= u.kpc * u.kpc**-2 * u.erg*u.cm**3/u.s * u.cm**-6
            Sx = (Sx*self._D_ang**2).to('erg s-1 cm-2')/u.sr
            Sx.to('erg s-1 cm-2 sr-1')
        elif output_type == 'C':
            Sx *= u.kpc * u.kpc**-2 * u.cm**3/u.s * u.cm**-6
            Sx = (Sx*self._D_ang**2).to('s-1 cm-2')/u.sr
            Sx.to('s-1 cm-2 sr-1')
        elif output_type == 'R':
            Sx *= u.kpc * u.kpc**-2 * u.cm**5/u.s * u.cm**-6
            Sx = (Sx*self._D_ang**2).to('s-1')/u.sr
            Sx.to('s-1 sr-1')
        else:
            raise ValueError("Output type available are S, C and R.")
        
        return Rproj.to('kpc'), Sx

    
    #==================================================
    # Compute Xray spherical flux
    #==================================================

    def get_fxsph_profile(self, radius=np.logspace(0,4,1000)*u.kpc, output_type='S'):
        """
        Get the spherically integrated Xray flux profile.
        
        Parameters
        ----------
        - radius (quantity) : the physical 3d radius in units homogeneous to kpc, as a 1d array
        - output_type (str): type of ooutput to provide: 
        S == energy counts in erg/s/cm^2/sr
        C == counts in ph/s/cm^2/sr
        R == count rate in ph/s/sr (accounting for instrumental response)
        
        Outputs
        ----------
        - radius (quantity): the 3d radius in unit of kpc
        - Fsph_r (quantity): the integrated Xray flux parameter erg/s/cm2

        """

        # In case the input is not an array
        if type(radius.to_value()) == float:
            radius = np.array([radius.to_value()]) * radius.unit

        #---------- Define radius associated to the density/temperature
        press_radius = cluster_profile.define_safe_radius_array(radius.to_value('kpc'), Rmin=1.0, Nptmin=1000)*u.kpc
        n_radius = cluster_profile.define_safe_radius_array(radius.to_value('kpc'), Rmin=1.0, Nptmin=1000)*u.kpc
        
        #---------- Get the density profile and temperature
        rad, n_e  = self.get_density_gas_profile(radius=n_radius)
        rad, T_g  = self.get_temperature_gas_profile(radius=n_radius)

        #---------- Interpolate the differential surface brightness
        dC_xspec, dS_xspec, dR_xspec = self.itpl_xspec_table(self._output_dir+'/XSPEC_table.txt', T_g)
        
        #---------- Get the integrand
        mu_gas, mu_e, mu_p, mu_alpha = cluster_global.mean_molecular_weight(self._helium_mass_fraction)
        constant = 1e-14/(4*np.pi*self._D_ang**2*(1+self._redshift)**2)
        if output_type == 'S':
            integrand = constant.to_value('kpc-2')*dS_xspec.to_value('erg cm3 s-1') * n_e.to_value('cm-3')**2 * mu_e/mu_p
        elif output_type == 'C':
            integrand = constant.to_value('kpc-2')*dC_xspec.to_value('cm3 s-1') * n_e.to_value('cm-3')**2 * mu_e/mu_p
        elif output_type == 'R':
            integrand = constant.to_value('kpc-2')*dR_xspec.to_value('cm5 s-1') * n_e.to_value('cm-3')**2 * mu_e/mu_p
        else:
            raise ValueError("Output type available are S, C and R.")        
        
        #---------- Integrate in 3d
        EI_r = np.zeros(len(radius))
        for i in range(len(radius)):
            EI_r[i] = cluster_profile.get_volume_any_model(rad.to_value('kpc'), integrand,
                                                           radius.to_value('kpc')[i], Npt=1000)
        if output_type == 'S':
            flux_r = EI_r*u.Unit('kpc-2 erg cm3 s-1 cm-6 kpc3')
            flux_r = flux_r.to('erg s-1 cm-2')
        elif output_type == 'C':
            flux_r = EI_r*u.Unit('kpc-2 cm3 s-1 cm-6 kpc3')
            flux_r = flux_r.to('s-1 cm-2')
        elif output_type == 'R':
            flux_r = EI_r*u.Unit('kpc-2 cm5 s-1 cm-6 kpc3')
            flux_r = flux_r.to('s-1')
        else:
            raise ValueError("Output type available are S, C and R.")        
        
        return radius, flux_r


    #==================================================
    # Compute Xray cylindrical flux
    #==================================================

    def get_fxcyl_profile(self, radius=np.logspace(0,4,1000)*u.kpc, NR500max=5.0, Npt_los=100,
                          output_type='S'):
        """
        Get the cylindrically integrated Xray flux profile.
        
        Parameters
        ----------
        - radius (quantity) : the physical 3d radius in units homogeneous to kpc, as a 1d array
        - NR500max (float): the integration will stop at NR500max x R500
        - Npt_los (int): the number of points for line of sight integration
        - output_type (str): type of ooutput to provide: 
        S == energy counts in erg/s/cm^2/sr
        C == counts in ph/s/cm^2/sr
        R == count rate in ph/s/sr (accounting for instrumental response)
        
        Outputs
        ----------
        - radius (quantity): the 3d radius in unit of kpc
        - Fcyl_r (quantity): the integrated Xray flux parameter erg/s/cm2

        """
        
        # In case the input is not an array
        if type(radius.to_value()) == float:
            radius = np.array([radius.to_value()]) * radius.unit

        #---------- Define radius associated to the Sx profile
        sx_radius = cluster_profile.define_safe_radius_array(radius.to_value('kpc'), Rmin=1.0, Nptmin=1000)*u.kpc
        
        #---------- Get the Sx profile
        r2d, sx_r = self.get_sx_profile(sx_radius, NR500max=NR500max, Npt_los=Npt_los, output_type=output_type)

        #---------- Integrate the Compton parameter in 2d
        if output_type == 'S':
            integrand = sx_r.to_value('erg s-1 cm-2 sr-1')
        elif output_type == 'C':
            integrand = sx_r.to_value('s-1 cm-2 sr-1')
        elif output_type == 'R':
            integrand = sx_r.to_value('s-1 sr-1')
        else:
            raise ValueError("Output type available are S, C and R.")        
        
        Fcyl_r = np.zeros(len(radius))
        for i in range(len(radius)):
            Fcyl_r[i] = cluster_profile.get_surface_any_model(r2d.to_value('kpc'), integrand,
                                                              radius.to_value('kpc')[i], Npt=1000)
        if output_type == 'S':
            flux_r = Fcyl_r / self._D_ang.to_value('kpc')**2 * u.Unit('erg s-1 cm-2')
        elif output_type == 'C':
            flux_r = Fcyl_r / self._D_ang.to_value('kpc')**2 * u.Unit('s-1 cm-2')
        elif output_type == 'R':
            flux_r = Fcyl_r / self._D_ang.to_value('kpc')**2 * u.Unit('s-1')
        else:
            raise ValueError("Output type available are S, C and R.")        
                
        return radius, flux_r


    #==================================================
    # Compute Sx map 
    #==================================================
    
    def get_sxmap(self, FWHM=None, NR500max=5.0, Npt_los=100,
                  output_type='S'):
        """
        Compute a Surface brightness X-ray mmap.
        
        Parameters
        ----------
        - FWHM (quantity) : the beam smoothing FWHM (homogeneous to deg)
        - NR500max (float): the integration will stop at NR500max x R500
        - Npt_los (int): the number of points for line of sight integration
        - output_type (str): type of ooutput to provide: 
        S == energy counts in erg/s/cm^2/sr
        C == counts in ph/s/cm^2/sr
        R == count rate in ph/s/sr (accounting for instrumental response)
        
        Outputs
        ----------
        - sxmap (quantity) : the Sx map

        """
        
        # Get the header
        header = self.get_map_header()
        w = WCS(header)
        if w.wcs.has_cd():
            if w.wcs.cd[1,0] != 0 or w.wcs.cd[0,1] != 0:
                print('!!! WARNING: R.A and Dec. is rotated wrt x and y. The extracted resolution was not checked in such situation.')
            map_reso_x = np.sqrt(w.wcs.cd[0,0]**2 + w.wcs.cd[1,0]**2)
            map_reso_y = np.sqrt(w.wcs.cd[1,1]**2 + w.wcs.cd[0,1]**2)
        else:
            map_reso_x = np.abs(w.wcs.cdelt[0])
            map_reso_y = np.abs(w.wcs.cdelt[1])
        
        # Get a R.A-Dec. map
        ra_map, dec_map = map_tools.get_radec_map(header)

        # Get a cluster distance map
        dist_map = map_tools.greatcircle(ra_map, dec_map, self._coord.icrs.ra.to_value('deg'), self._coord.icrs.dec.to_value('deg'))

        # Define the radius used fo computing the Sx profile
        theta_max = np.amax(dist_map) # maximum angle from the cluster
        theta_min = np.amin(dist_map) # minimum angle from the cluster (~0 if cluster within FoV)
        if theta_min == 0:
            theta_min = 1e-4 # Zero will cause bug, put <1arcsec in this case
        rmax = theta_max*np.pi/180 * self._D_ang.to_value('kpc')
        rmin = theta_min*np.pi/180 * self._D_ang.to_value('kpc')
        radius = np.logspace(np.log10(rmin), np.log10(rmax), 1000)*u.kpc

        # Compute the Compton parameter projected profile
        r_proj, sx_profile = self.get_sx_profile(radius, NR500max=NR500max, Npt_los=Npt_los, output_type=output_type)
        theta_proj = (r_proj/self._D_ang).to_value('')*180.0/np.pi                           # degrees
        
        # Interpolate the profile onto the map
        if output_type == 'S':
            sxmap = map_tools.profile2map(sx_profile.to_value('erg s-1 cm-2 sr-1'), theta_proj, dist_map)
        elif output_type == 'C':
            sxmap = map_tools.profile2map(sx_profile.to_value('s-1 cm-2 sr-1'), theta_proj, dist_map)
        elif output_type == 'R':
            sxmap = map_tools.profile2map(sx_profile.to_value('s-1 sr-1'), theta_proj, dist_map)
        else:
            raise ValueError("Output type available are S, C and R.")        
        
        # Avoid numerical residual ringing from interpolation
        sxmap[dist_map > self._theta_truncation.to_value('deg')] = 0
        
        # Smooth the ymap if needed
        if FWHM != None:
            FWHM2sigma = 1.0/(2.0*np.sqrt(2*np.log(2)))
            sxmap = ndimage.gaussian_filter(sxmap, sigma=(FWHM2sigma*FWHM.to_value('deg')/map_reso_x,
                                                          FWHM2sigma*FWHM.to_value('deg')/map_reso_y), order=0)
        # Units and return
        if output_type == 'S':
            sxmap = sxmap*u.Unit('erg s-1 cm-2 sr-1')
        elif output_type == 'C':
            sxmap = sxmap*u.Unit('s-1 cm-2 sr-1')
        elif output_type == 'R':
            sxmap = sxmap*u.Unit('s-1 sr-1')
        else:
            raise ValueError("Output type available are S, C and R.")        
        
        return sxmap

    
    #==================================================
    # Saving txt file utility function
    #==================================================
    
    def _save_txt_file(self, filename, col1, col2, col1_name, col2_name, ndec=20):
        """
        Save the file with a given format in txt file
        
        Parameters
        ----------
        - filename (str): full path to the file
        - col1 (np.ndarray): the first column of data
        - col2 (np.ndarray): the second column of data
        - col1_name (str): the name of the first column
        - col2_name (str): the name of the second column
        - ndec (int): number of decimal in numbers
        
        ----------
        Files are saved

        """
        
        ncar = ndec + 6

        # Mae sure name are not too long
        col1_name = ('{:.'+str(ncar-1)+'}').format(col1_name)
        col2_name = ('{:.'+str(ncar)+'}').format(col2_name)

        # Correct formating
        col1_name = ('{:>'+str(ncar-1)+'}').format(col1_name)
        col2_name = ('{:>'+str(ncar)+'}').format(col2_name)

        # saving
        sfile = open(filename, 'wb')
        sfile.writelines(['#'+col1_name, ('{:>'+str(ncar)+'}').format(''), col2_name+'\n'])
        for il in range(len(col1)):
            sfile.writelines([('{:.'+str(ndec)+'e}').format(col1[il]),
                              ('{:>'+str(ncar)+'}').format(''),
                              ('{:.'+str(ndec)+'e}').format(col2[il])+'\n'])
        sfile.close()

        
    #==================================================
    # Save profile
    #==================================================
    
    def save_profile(self, radius=np.logspace(0,4,1000)*u.kpc, prod_list=['all'],
                     NR500max=5.0, Npt_los=100,
                     Energy_density=False, Epmin=None, Epmax=None, Egmin=10.0*u.MeV, Egmax=1.0*u.PeV,
                     Sx_type='S'):
        """
        Save the 3D profiles in a file
        
        Parameters
        ----------
        - radius (quantity): the 3d radius in unit of kpc
        - prod_list (str): the outputs to produce. Can include: all, gas_pressure, gas_density,
        gas_temperature, gas_entropy, hse_mass, overdensity, gas_mass, fgas, u_thermal, ysph, 
        ycyl, ycompton, crp_density, crp_energy_fraction, gamma
        - NR500max (float): the integration will stop at NR500max x R500
        Only used for projected profiles.
        - Npt_los (int): the number of points for line of sight integration
        Only used for projected profiles.
        - Energy_density (bool): if True, then the energy density is computed. Otherwise, 
        the number density is computed. In the case of CRp and gamma rays.
        Outputs
        - Epmin (quantity): the lower bound for energy proton integration
        - Epmax (quantity): the upper bound for energy proton integration
        - Egmin (quantity): the lower bound for energy gamma integration
        - Egmax (quantity): the upper bound for energy gamma integration
        ----------
        Files are saved

        """

        if Epmin == None:
            Epmin = self._Epmin
        if Epmax == None:
            Epmax = self._Epmax
        
        # Create the output directory if needed
        if not os.path.exists(self._output_dir): os.mkdir(self._output_dir)
        
        # Create a dataframe to store all spectra in a single fits table
        tab  = Table()
        tab['Radius'] = Column(radius.to_value('kpc'), unit='kpc', description='Radius')
        
        #---------- pressure
        if 'all' in prod_list or 'p_e' in prod_list:
            rad, prof = self.get_pressure_gas_profile(radius)
            tab['p_e'] = Column(prof.to_value('keV cm-3'), unit='keV cm-3', description='Thermal electron pressure')
            self._save_txt_file(self._output_dir+'/PROFILE_pressure_electron.txt',
                                radius.to_value('kpc'), prof.to_value('keV cm-3'), 'radius (kpc)', 'pressure (keV cm-3)')
            
        #---------- density
        if 'all' in prod_list or 'n_e' in prod_list:
            rad, prof = self.get_density_gas_profile(radius)
            tab['n_e'] = Column(prof.to_value('cm-3'), unit='cm-3', description='Thermal electron density')
            self._save_txt_file(self._output_dir+'/PROFILE_density_electron.txt',
                                radius.to_value('kpc'), prof.to_value('cm-3'), 'radius (kpc)', 'density (cm-3)')

        #---------- temperature
        if 'all' in prod_list or 't_gas' in prod_list:
            rad, prof = self.get_temperature_gas_profile(radius)
            tab['t_gas'] = Column(prof.to_value('keV'), unit='keV', description='Thermal gas temperature')
            self._save_txt_file(self._output_dir+'/PROFILE_temperature_gas.txt',
                                radius.to_value('kpc'), prof.to_value('keV'), 'radius (kpc)', 'temperature (keV)')
            
        #---------- Entropy
        if 'all' in prod_list or 'k_gas' in prod_list: 
            rad, prof = self.get_entropy_gas_profile(radius)
            tab['k_gas'] = Column(prof.to_value('keV cm2'), unit='keV cm2', description='Thermal gas entropy')
            self._save_txt_file(self._output_dir+'/PROFILE_entropy_gas.txt',
                                radius.to_value('kpc'), prof.to_value('keV cm2'), 'radius (kpc)', 'entropy (keV cm2)')

        #---------- Masse HSE
        if 'all' in prod_list or 'm_hse' in prod_list:
            rad, prof = self.get_hse_mass_profile(radius)
            tab['m_hse'] = Column(prof.to_value('Msun'), unit='Msun', description='Enclosed hydrostatic mass')
            self._save_txt_file(self._output_dir+'/PROFILE_mass_hydrostatic.txt',
                                radius.to_value('kpc'), prof.to_value('Msun'), 'radius (kpc)', 'mass HSE (Msun)')

        #---------- Overdensity
        if 'all' in prod_list or 'overdensity' in prod_list:
            rad, prof = self.get_overdensity_contrast_profile(radius)
            tab['overdensity'] = Column(prof.to_value('adu'), unit='adu', description='Enclosed overdensity wrt critical density')
            self._save_txt_file(self._output_dir+'/PROFILE_overdensity.txt',
                                radius.to_value('kpc'), prof.to_value('adu'), 'radius (kpc)', 'overdensity')

        #---------- Gas mass
        if 'all' in prod_list or 'm_gas' in prod_list:
            rad, prof = self.get_gas_mass_profile(radius)
            tab['m_gas'] = Column(prof.to_value('Msun'), unit='Msun', description='Enclosed gas mass')
            self._save_txt_file(self._output_dir+'/PROFILE_mass_gas.txt',
                                radius.to_value('kpc'), prof.to_value('Msun'), 'radius (kpc)', 'mass gas (Msun)')

        #---------- fgas profile
        if 'all' in prod_list or 'f_gas' in prod_list:
            rad, prof = self.get_fgas_profile(radius)
            tab['f_gas'] = Column(prof.to_value('adu'), unit='adu', description='Enclosed gas fraction')
            self._save_txt_file(self._output_dir+'/PROFILE_fraction_gas.txt',
                                radius.to_value('kpc'), prof.to_value('adu'), 'radius (kpc)', 'fraction gas')

        #---------- Thermal energy
        if 'all' in prod_list or 'u_th' in prod_list:
            rad, prof = self.get_thermal_energy_profile(radius)
            tab['u_th'] = Column(prof.to_value('erg'), unit='erg', description='Enclosed thermal energy')
            self._save_txt_file(self._output_dir+'/PROFILE_energy_thermal.txt',
                                radius.to_value('kpc'), prof.to_value('erg'), 'radius (kpc)', 'thermal energy (erg)')

        #---------- Spherically integrated Compton
        if 'all' in prod_list or 'y_sph' in prod_list:
            rad, prof = self.get_ysph_profile(radius)
            tab['y_sph'] = Column(prof.to_value('kpc2'), unit='kpc2', description='Spherically integrated Compton parameter')
            self._save_txt_file(self._output_dir+'/PROFILE_compton_spherical.txt',
                                radius.to_value('kpc'), prof.to_value('kpc2'), 'radius (kpc)', 'Y sph (kpc2)')

        #---------- Cylindrically integrated Compton
        if 'all' in prod_list or 'y_cyl' in prod_list:
            rad, prof = self.get_ycyl_profile(radius, NR500max=NR500max, Npt_los=Npt_los)
            tab['y_cyl'] = Column(prof.to_value('kpc2'), unit='kpc2', description='Cylindrically integrated Compton parameter')
            self._save_txt_file(self._output_dir+'/PROFILE_compton_cylindrical.txt',
                                radius.to_value('kpc'), prof.to_value('kpc2'), 'radius (kpc)', 'Y cyl (kpc2)')

        #---------- Compton parameter
        if 'all' in prod_list or 'y_sz' in prod_list:
            rad, prof = self.get_y_compton_profile(radius, NR500max=NR500max, Npt_los=Npt_los)
            tab['y_sz'] = Column(prof.to_value('adu'), unit='adu', description='Compton parameter')
            self._save_txt_file(self._output_dir+'/PROFILE_compton_sz.txt',
                                radius.to_value('kpc'), prof.to_value('adu'), 'radius (kpc)', 'y')

        #---------- Cosmic ray proton
        if 'all' in prod_list or 'n_crp' in prod_list:
            rad, prof = self.get_density_crp_profile(radius, Emin=Epmin, Emax=Epmax, Energy_density=Energy_density)
            tab['n_crp'] = Column(prof.to_value('cm-3'), unit='cm-3', description='Cosmic ray proton density')
            self._save_txt_file(self._output_dir+'/PROFILE_density_cosmic_proton.txt',
                                radius.to_value('kpc'), prof.to_value('cm-3'), 'radius (kpc)', 'density (cm-3)')

        #---------- Cosmic ray to thermal energy
        if 'all' in prod_list or 'x_crp' in prod_list:
            rad, prof = self.get_crp_to_thermal_energy_profile(radius, Emin=Epmin, Emax=Epmax)
            tab['x_crp'] = Column(prof.to_value('adu'), unit='adu', description='Enclosed cosmic ray to thermal energy')
            self._save_txt_file(self._output_dir+'/PROFILE_fraction_energy_cosmic_to_thermal.txt',
                                radius.to_value('kpc'), prof.to_value('adu'), 'radius (kpc)', 'x')

        #---------- Gamma ray profile
        if 'all' in prod_list or 'gamma' in prod_list:
            rad, prof = self.get_gamma_profile(radius, Emin=Egmin, Emax=Egmax, Energy_density=Energy_density,
                                               NR500max=NR500max, Npt_los=Npt_los)
            if Energy_density:
                tab['Sgamma'] = Column(prof.to_value('GeV cm-2 s-1 sr-1'), unit='GeV cm-2 s-1 sr-1', description='Gamma ray surface brightness')
                self._save_txt_file(self._output_dir+'/PROFILE_gamma_ray_surface_brightness.txt',
                                    radius.to_value('kpc'), prof.to_value('GeV cm-2 s-1 sr-1'), 'radius (kpc)', 'gamma SB (GeV cm-2 s-1 sr-1)')
            else:
                tab['Sgamma'] = Column(prof.to_value('cm-2 s-1 sr-1'), unit='cm-2 s-1 sr-1', description='Gamma ray surface brightness')
                self._save_txt_file(self._output_dir+'/PROFILE_gamma_ray_surface_brightness.txt',
                                    radius.to_value('kpc'), prof.to_value('cm-2 s-1 sr-1'), 'radius (kpc)', 'gamma SB (cm-2 s-1 sr-1)')

        #---------- Spherically integrated X flux
        if 'all' in prod_list or 'fx_sph' in prod_list:
            rad, prof = self.get_fxsph_profile(radius, output_type=Sx_type)
            tab['fx_sph'] = Column(prof.to_value('erg s-1 cm-2'), unit='erg s-1 cm-2', description='Spherically integrated Xray fux')
            self._save_txt_file(self._output_dir+'/PROFILE_xray_flux_spherical.txt',
                                radius.to_value('kpc'), prof.to_value('erg s-1 cm-2'), 'radius (kpc)', 'Fx sph (erg s-1 cm-2)')

        #---------- Cylindrically integrated X flux
        if 'all' in prod_list or 'fx_cyl' in prod_list:
            rad, prof = self.get_fxcyl_profile(radius, NR500max=NR500max, Npt_los=Npt_los, output_type=Sx_type)
            tab['fx_cyl'] = Column(prof.to_value('erg s-1 cm-2'), unit='erg s-1 cm-2', description='Cylindrically integrated Xray flux')
            self._save_txt_file(self._output_dir+'/PROFILE_xray_flux_cylindrical.txt',
                                radius.to_value('kpc'), prof.to_value('erg s-1 cm-2'), 'radius (kpc)', 'Fx cyl (erg s-1 cm-2)')

        #---------- Sx
        if 'all' in prod_list or 'sx' in prod_list:
            rad, prof = self.get_sx_profile(radius, NR500max=NR500max, Npt_los=Npt_los, output_type=Sx_type)
            tab['sx'] = Column(prof.to_value('erg s-1 cm-2 sr-1'), unit='erg s-1 cm-2 sr-1', description='Xray surface brightness')
            self._save_txt_file(self._output_dir+'/PROFILE_xray_surface_brightness.txt',
                                radius.to_value('kpc'), prof.to_value('erg s-1 cm-2 sr-1'), 'radius (kpc)', 'Sx (erg s-1 cm-2 sr-1)')

        # Save the data frame in a single file as well
        tab.meta['comments'] = ['Proton spectra are integrated within '+str(Epmin)+' and '+str(Epmax)+'.',
                                'Gamma ray spectra are integrated within '+str(Egmin)+' and '+str(Egmax)+'.',
                                'The projection of line-of-sight integrated profiles stops at '+str(NR500max)+' R500.',
                                'The number of points for the line-of-sight integration is '+str(Npt_los)+'.']
        tab.write(self._output_dir+'/PROFILE.fits', overwrite=True)
        
        
    #==================================================
    # Save 2d profile (projected)
    #==================================================
    
    def save_spectra(self, energy=np.logspace(-2,6,1000)*u.GeV, prod_list=['all'], Rmax=None,
                     NR500max=5.0, Npt_los=100):
        """
        Save the spectra
        
        Parameters
        ----------
        - energy (quantity) : the physical energy of CR protons
        - prod_list (str): the outputs to produce. Can include: all, proton, gamma
        - Rmax (quantity): the radius within with the spectrum is computed 
        (default is R500)
        - NR500max (float): the line-of-sight integration will stop at NR500max x R500. 
        This is used only for projected quantities
        - Npt_los (int): the number of points for line of sight integration.
        This is used only for projected quantities

        Outputs
        ----------
        Files are saved

        """

        if Rmax == None:
            Rmax = self._R500

        # Create the output directory if needed
        if not os.path.exists(self._output_dir): os.mkdir(self._output_dir)

        # Create a dataframe to store all spectra in a single fits table
        tab  = Table()
        tab['Energy'] = Column(energy.to_value('GeV'), unit='GeV', description='Energy')
        
        #---------- proton spectrum
        if 'all' in prod_list or 'proton' in prod_list:
            eng, spec = self.get_crp_spectrum(energy, Rmax=Rmax)
            tab['Spec_CRp'] = Column(spec.to_value('GeV-1'), unit='GeV-1',
                                     description='Spherically integrated cosmic ray proton spectrum')
            self._save_txt_file(self._output_dir+'/SPECTRA_cosmic_ray_proton.txt',
                                eng.to_value('MeV'), spec.to_value('GeV-1'), 'energy (MeV)', 'spectrum (GeV-1)')
            
        #---------- gamma spectrum
        if 'all' in prod_list or 'gamma' in prod_list:
            # Spherical case
            eng, spec = self.get_gamma_spectrum(energy, Rmax=Rmax, type_integral='spherical', NR500max=NR500max, Npt_los=Npt_los)
            tab['Spec_gamma_sph'] = Column(spec.to_value('MeV-1 cm-2 s-1'), unit='MeV-1 cm-2 s-1',
                                           description='Spherically integrated gamma ray spectrum')
            self._save_txt_file(self._output_dir+'/SPECTRA_gamma_ray_spherical.txt',
                                eng.to_value('MeV'), spec.to_value('MeV-1 cm-2 s-1'), 'energy (MeV)', 'spectrum (MeV-1 cm-2 s-1)')

            # Cylindrical case
            eng, spec = self.get_gamma_spectrum(energy, Rmax=Rmax, type_integral='cylindrical', NR500max=NR500max, Npt_los=Npt_los)
            tab['Spec_gamma_cyl'] = Column(spec.to_value('MeV-1 cm-2 s-1'), unit='MeV-1 cm-2 s-1',
                                           description='Cylindrically integrated gamma ray spectrum')
            self._save_txt_file(self._output_dir+'/SPECTRA_gamma_ray_cylindrical.txt',
                                eng.to_value('MeV'), spec.to_value('MeV-1 cm-2 s-1'), 'energy (MeV)', 'spectrum (MeV-1 cm-2 s-1)')
            
        # Save the data frame in a single file as well
        tab.meta['comments'] = ['Spectra are computed within '+str(Rmax.to_value('kpc'))+' kpc.',
                                'The projection of profiles stops at '+str(NR500max)+' R500.',
                                'The number of points for the line-of-sight integration is '+str(Npt_los)+'.']
        tab.write(self._output_dir+'/SPECTRA.fits', overwrite=True)
        
        
    #==================================================
    # Save map
    #==================================================
    
    def save_map(self, prod_list=['all'], NR500max=5.0, Npt_los=100):
        """
        Save the maps in a file
        
        Parameters
        ----------
        - prod_list (str): the outputs to produce. Can include: all, y_map, gamma_map

        Outputs
        ----------
        Files are saved

        """

        # Create the output directory if needed
        if not os.path.exists(self._output_dir): os.mkdir(self._output_dir)

        # Get the header of the maps 
        header = self.get_map_header()

        #---------- y map
        if 'all' in prod_list or 'y_map' in prod_list:
            image = self.get_ymap(NR500max=NR500max, Npt_los=Npt_los).to_value('adu')

            hdu = fits.PrimaryHDU(header=header)
            hdu.data = image
            hdu.writeto(self._output_dir+'/MAP_y_sz.fits', overwrite=True)

        #---------- gamma map
        if 'all' in prod_list or 'gamma_map' in prod_list: 
            image = self.get_gamma_template_map(NR500max=NR500max, Npt_los=Npt_los).to_value('sr-1')

            hdu = fits.PrimaryHDU(header=header)
            hdu.data = image
            hdu.writeto(self._output_dir+'/MAP_gamma_template.fits', overwrite=True)

       #---------- Sx map
        if 'all' in prod_list or 'sx_map' in prod_list:
            image = self.get_sxmap(NR500max=NR500max, Npt_los=Npt_los).to_value('erg s-1 cm-2 sr-1')

            hdu = fits.PrimaryHDU(header=header)
            hdu.data = image
            hdu.writeto(self._output_dir+'/MAP_Sx.fits', overwrite=True)

            
    #==================================================
    # Plots
    #==================================================
    
    def plot(self, prod_list=['all'],
             radius=np.logspace(0,4,1000)*u.kpc, energy=np.logspace(-2,6,1000)*u.GeV,
             NR500max=5.0, Npt_los=100, Rmax=None,
             Epmin=None, Epmax=None, Egmin=10.0*u.MeV, Egmax=1.0*u.PeV):
        """
        Plot what we want
        
        Parameters
        ----------
        - prod_list (str): the outputs to produce

        Outputs
        ----------
        Plots are saved as pdf.

        """
        
        if not os.path.exists(self._output_dir): os.mkdir(self._output_dir)
        model_plots.main(self, prod_list,
                         radius=radius, energy=energy,
                         NR500max=NR500max, Npt_los=Npt_los, Rmax=Rmax,
                         Epmin=Epmin, Epmax=Epmax, Egmin=Egmin, Egmax=Egmax)
