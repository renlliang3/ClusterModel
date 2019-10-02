"""
This file contains the Cluster class. It is dedicated to the construction of a 
Cluster object, definined by its physical properties and with  associated methods
to compute derived properties or observables. It focuses on the thermal and non-thermal 
component of the clusters ICM.

"""

#==================================================
# Requested imports
#==================================================

import numpy as np
import astropy.units as u
from astropy.io import fits
import astropy.cosmology
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS

from ClusterTools import cluster_global 
from ClusterTools import cluster_spectra 

from ClusterModel             import model_title
from ClusterModel.model_admin import Admin
from ClusterModel.model_phys  import Physics
from ClusterModel.model_obs   import Observables
from ClusterModel.model_plots import Plots


#==================================================
# Cluster class
#==================================================

class Cluster(Admin, Physics, Observables, Plots):
    """ Cluster class. 
    This class defines a cluster object. In addition to basic properties such as 
    mass and redshift, it includes the physical properties (e.g. pressure profile, 
    cosmic ray spectrum) from which derived properties can be obtained (e.g. 
    hydrostatic mass profile) as well as observables.
    
    To do list
    ----------  
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
    Methods are split in the respective files: 
    - model_admin.py
    - model_phys.py
    - model_obs.py
    - model_plots.py

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
        
        #---------- Print the code header at launch
        if not silent:
            model_title.show()

        #---------- Admin
        self._silent     = silent
        self._output_dir = './ClusterModel'

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

        # Cluster boundery
        self._R_truncation     = 3*self._R500
        self._theta_truncation = ((self._R_truncation / self._D_ang).to('') * u.rad).to('deg')
        
        # He fraction and metal abundances (in unit of Z_sun)
        self._helium_mass_fraction = 0.245
        self._abundance = 0.3

        # HSE bias
        self._hse_bias = 0.2

        #---------- Physical properties
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

        # Cosmic ray protons
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

        # Magnetic fields
        #self._mag_field_model = {"name"   : "SVM",
        #                         "B_0"    : 1e-6*u.Unit('G'),
        #                         "r_c"    : 500.0*u.kpc,
        #                         "beta"   : 0.75,
        #                         "r_s"    : 800.0*u.kpc,
        #                         "alpha"  : 0.6,
        #                         "gamma"  : 3.0,
        #                         "epsilon": 0.0}
        
        
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
            data_tpl = np.zeros((value['NAXIS2'], value['NAXIS1']))            
            header = w.to_header()
            hdu = fits.PrimaryHDU(header=header, data=data_tpl)
            header = hdu.header
        except:
            raise TypeError("It seems that the header you provided is not really a header, or does not contain NAXIS1,2.")
        
        # set the value
        self._map_header = header
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

        
