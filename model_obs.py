"""
This file contain a subclass of the model.py module and Cluster class. It
is dedicated to the computing of observables.

"""

#==================================================
# Requested imports
#==================================================

import numpy as np
import scipy.ndimage as ndimage
import astropy.units as u
from astropy.wcs import WCS
from astropy import constants as const

import naima

from ClusterModel import model_tools 

from ClusterTools import cluster_global 
from ClusterTools import cluster_profile 
from ClusterTools import cluster_spectra 
from ClusterTools import map_tools


#==================================================
# Observable class
#==================================================

class Observables(object):
    """ Observable class
    This class serves as a parser to the main Cluster class, to 
    include the subclass Observable in this other file.

    Attributes
    ----------  
    The attributes are the same as the Cluster class, see model.py

    Methods
    ----------  
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

    - get_ysph_profile(self, radius=np.logspace(0,4,1000)*u.kpc): compute the spherically 
    integrated compton parameter profile
    - get_ycyl_profile(self, radius=np.logspace(0,4,1000)*u.kpc): compute the cylindrincally 
    integrated Compton parameter profile
    - get_y_compton_profile(self, radius=np.logspace(0,4,1000)*u.kpc, NR500max=5.0, Npt_los=100):
    compute the Compton parameter profile
    - get_ymap(self, FWHM=None, NR500max=5.0, Npt_los=100): compute a Compton parameter map.

    - get_sx_profile(self, radius=np.logspace(0,4,1000)*u.kpc, NR500max=5.0, Npt_los=100,
    output_type='S'): compute the Xray surface brightness profile
    - get_fxsph_profile(self, radius=np.logspace(0,4,1000)*u.kpc, output_type='S'): compute the Xray 
    spherically integrated flux profile
    - get_fxcyl_profile(self, radius=np.logspace(0,4,1000)*u.kpc, NR500max=5.0, Npt_los=100,
    output_type='S'): compute the Xray cylindrically integrated flux profile
    - get_sxmap(self, FWHM=None, NR500max=5.0, Npt_los=100, output_type='S'): compute the Xray 
    surface brigthness map

    """
    
    #==================================================
    # Compute gamma ray spectrum
    #==================================================

    def get_gamma_spectrum2(self, energy=np.logspace(-2,6,100)*u.GeV,
                            Rmin=None, Rmax=None,
                            type_integral='spherical',
                            NR500max=5.0):

        # In case the input is not an array
        energy = model_tools.check_qarray(energy, unit='GeV')

        # Check the type of integral
        ok_list = ['spherical', 'cylindrical']
        if not type_integral in ok_list:
            raise ValueError("This requested integral type (type_integral) is not available")

        # Get the integration limits
        if Rmin is None:
            Rmin = self._Rmin
        if Rmax is None:
            Rmax = self._R500

        # Compute the integral
        if type_integral == 'spherical':
            rad = model_tools.sampling_array(Rmin, Rmax, NptPd=self._Npt_per_decade_integ, unit=True)
            dN_dEdVdt = self.get_rate_gamma_ray(energy, rad)
            dN_dEdt = model_tools.compute_spectrum_spherical(dN_dEdVdt, rad)
            
        # Compute the integral        
        if type_integral == 'cylindrical':
            los = model_tools.sampling_array(Rmin, NR500max*self._R500, NptPd=self._Npt_per_decade_integ, unit=True)
            r2d = model_tools.sampling_array(Rmin, Rmax, NptPd=self._Npt_per_decade_integ, unit=True)
            dN_dEdt = model_tools.compute_spectrum_cylindrical(self.get_rate_gamma_ray, energy, r2d, los)
            #dN_dEdt = model_tools.compute_spectrum_cylindrical_loop(self.get_rate_gamma_ray, energy, r2d, los)
        
        # From intrinsic luminosity to flux
        dN_dEdSdt = dN_dEdt / (4*np.pi * self._D_lum**2)
        
        return energy, dN_dEdSdt.to('GeV-1 cm-2 s-1')

    


    
    def get_gamma_spectrum(self, energy=np.logspace(-2,6,100)*u.GeV, Rmax=None,
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
        CRenergy_Rcut = self._X_cr_E['X'] * self.get_thermal_energy_profile(self._X_cr_E['R_norm'])[1][0]
        gamma.set_Wp(CRenergy_Rcut, Epmin=self._Epmin, Epmax=self._Epmax)

        # Compute the normalization volume and the integration cross density volume
        r3d1 = cluster_profile.define_safe_radius_array(np.array([self._X_cr_E['R_norm'].to_value('kpc')]), Rmin=1.0)*u.kpc
        radius1, f_crp_r1 = self.get_normed_density_crp_profile(r3d1)
        V_CRenergy = cluster_profile.get_volume_any_model(radius1.to_value('kpc'), f_crp_r1.to_value('adu'),
                                                          self._X_cr_E['R_norm'].to_value('kpc'))*u.kpc**3

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

        #---------- Apply EBL absorbtion
        if self._EBL_model != 'none':
            absorb = cluster_spectra.get_ebl_absorb(energy.to_value('GeV'), self._redshift, self._EBL_model)
            dN_dEdSdt = dN_dEdSdt * absorb
        
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
        CRenergy_Rcut = self._X_cr_E['X'] * self.get_thermal_energy_profile(self._X_cr_E['R_norm'])[1][0]
        gamma.set_Wp(CRenergy_Rcut, Epmin=self._Epmin, Epmax=self._Epmax)

        # Compute the normalization volume and the integration cross density volume
        r3d1 = cluster_profile.define_safe_radius_array(np.array([self._X_cr_E['R_norm'].to_value('kpc')]), Rmin=1.0)*u.kpc
        radius1, f_crp_r1 = self.get_normed_density_crp_profile(r3d1)
        V_CRenergy = cluster_profile.get_volume_any_model(radius1.to_value('kpc'), f_crp_r1, self._X_cr_E['R_norm'].to_value('kpc'))*u.kpc**3
        
        # Compute the spectral part
        energy = np.logspace(np.log10(Emin.to_value('GeV')), np.log10(Emax.to_value('GeV')), 1000)*u.GeV
        dN_dEdSdt = gamma.flux(energy, distance=self._D_lum).to('MeV-1 cm-2 s-1')

        # Apply EBL absorbtion
        if self._EBL_model != 'none':
            absorb = cluster_spectra.get_ebl_absorb(energy.to_value('GeV'), self._redshift, self._EBL_model)
            dN_dEdSdt = dN_dEdSdt * absorb

        # and integrate over energy
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
        radius = model_tools.check_qarray(radius)

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
        radius = model_tools.check_qarray(radius)

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
        - Rproj (quantity): the projected 2d radius in unit of kpc
        - y_r : the Compton parameter

        Note
        ----------
        The pressure profile is truncated at R500 along the line-of-sight.

        """
        
        # In case the input is not an array
        radius = model_tools.check_qarray(radius)

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
        radius = model_tools.check_qarray(radius)

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
        mu_gas, mu_e, mu_p, mu_alpha = cluster_global.mean_molecular_weight(Y=self._helium_mass_fraction,
                                                                            Z=self._metallicity_sol*self._abundance)
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
        radius = model_tools.check_qarray(radius)

        #---------- Define radius associated to the density/temperature
        press_radius = cluster_profile.define_safe_radius_array(radius.to_value('kpc'), Rmin=1.0, Nptmin=1000)*u.kpc
        n_radius = cluster_profile.define_safe_radius_array(radius.to_value('kpc'), Rmin=1.0, Nptmin=1000)*u.kpc
        
        #---------- Get the density profile and temperature
        rad, n_e  = self.get_density_gas_profile(radius=n_radius)
        rad, T_g  = self.get_temperature_gas_profile(radius=n_radius)

        #---------- Interpolate the differential surface brightness
        dC_xspec, dS_xspec, dR_xspec = self.itpl_xspec_table(self._output_dir+'/XSPEC_table.txt', T_g)
        
        #---------- Get the integrand
        mu_gas, mu_e, mu_p, mu_alpha = cluster_global.mean_molecular_weight(Y=self._helium_mass_fraction,
                                                                            Z=self._metallicity_sol*self._abundance)
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
        radius = model_tools.check_qarray(radius)

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

