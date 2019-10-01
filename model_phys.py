import numpy as np
from scipy.misc import derivative
from scipy.optimize import brentq
import scipy.interpolate as interpolate
import scipy.ndimage as ndimage
import os

import astropy.units as u
from astropy.io import fits
import astropy.cosmology
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy import constants as const

import naima

from ClusterTools import cluster_global 
from ClusterTools import cluster_profile 
from ClusterTools import cluster_spectra 
from ClusterTools import cluster_xspec 
from ClusterTools import map_tools

from ClusterModel import model_title
from ClusterModel import model_plots


#==================================================
# Physics class
#==================================================

class Physics(object):
    """ Physics class
    This class searves as a parser to the main Cluster class, to 
    include the subclass Physics in this other file.

    Attributes
    ----------  
    The attributes are the same as the Cluster class, see model.py

    Methods
    ----------  
    - make_xspec_table(self, Emin=0.1*u.keV, Emax=2.4*u.keV,Tmin=0.1*u.keV, Tmax=50.0*u.keV, nbin=100,
    nH=0.0/u.cm**2, file_HI=None,Kcor=False): compute a temperature versus counts/Sx table to be interpolated
    when getting profiles and maps
    - itpl_xspec_table(self, xspecfile, Tinput): interpolate xspec tables to chosen temperature


    - set_pressure_gas_gNFW_param(self, pressure_model='P13UPP'): replace the electron gas 
    pressure profile GNFW parameters by the ones given by the user.

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

    """
    
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
