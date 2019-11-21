"""
This file deals with 'administrative' issues regarding the Cluster Class (e.g. saving parameters etc)
"""

import os
import pprint
import numpy as np
import astropy.units as u
import pickle
from astropy.table import Table, Column
from astropy.io import fits


#==================================================
# Admin class
#==================================================

class Admin(object):
    """ Admin class
    This class searves as a parser to the main Cluster class, to 
    include the subclass Admin in this other file.

    Attributes
    ----------  
    The attributes are the same as the Cluster class, see model.py

    Methods
    ----------  
    - print_param(self): print the parameters.
    - save_param(self): save the current parameters describing the cluster object.
    - load_param(self, param_file): load a given pre-saved parameter file. The parameter
    file should contain the right parameters to avoid issues latter on.

    - _validate_profile_model_parameters(self, inpar, unit): dedicated to check and validate the parameters 
    of profile models
    - _validate_spectrum_model_parameters(self, inpar, unit): dedicated to check and validate the parameters 
    of spectral models

    - save_profile(self, radius=np.logspace(0,4,1000)*u.kpc, prod_list=['all'], NR500max=5.0, 
    Npt_los=100, Energy_density=False, Epmin=None, Epmax=None, Egmin=10.0*u.MeV, Egmax=1.0*u.PeV):
    Save the profiles as fits and txt files.
    - save_spectra(self, energy=np.logspace(-2,6,1000)*u.GeV, prod_list=['all'], Rmax=None,
    NR500max=5.0, Npt_los=100): save the spectra as fits and txt files
    - save_map(self, prod_list=['all'], NR500max=5.0, Npt_los=100): save the maps as fits files
    
    - _save_txt_file(self, filename, col1, col2, col1_name, col2_name, ndec=20): internal method 
    dedicated to save data in special format

    """
    
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
    # Validate profile model parameters
    #==================================================
    
    def _validate_profile_model_parameters(self, inpar, unit):
        """
        Check the profile parameters.
        
        Parameters
        ----------
        - inpar (dict): a dictionary containing the input parameters
        - unit (str): contain the unit of the profile, e.g. keV/cm-3 for pressure
        
        Outputs
        ----------
        - outpar (dict): a dictionary with output parameters

        """

        # List of available authorized models
        model_list = ['GNFW', 'SVM', 'beta', 'doublebeta']
        
        # Deal with unit
        if unit == '' or unit == None:
            hasunit = False
        else:
            hasunit = True

        # Check that the input is a dictionary
        if type(inpar) != dict :
            raise TypeError("The model should be a dictionary containing the name key and relevant parameters")
        
        # Check that input contains a name
        if 'name' not in inpar.keys() :
            raise ValueError("The model dictionary should contain a 'name' field")
            
        # Check that the name is in the acceptable name list        
        if not inpar['name'] in model_list:
            print('The profile model can be:')
            print(model_list)
            raise ValueError("The requested model is not available")
        
        #---------- Deal with the case of GNFW
        if inpar['name'] == 'GNFW':
            # Check the content of the dictionary
            cond1 = 'P_0' in inpar.keys() and 'a' in inpar.keys() and 'b' in inpar.keys() and 'c' in inpar.keys()
            cond2 = 'c500' in inpar.keys() or 'r_p' in inpar.keys()
            cond3 = not('c500' in inpar.keys() and 'r_p' in inpar.keys())

            if not (cond1 and cond2 and cond3):
                raise ValueError("The GNFW model should contain: {'P_0','c500' or 'r_p' (not both), 'a', 'b', 'c'}.")
            
            # Check units and values
            if hasunit:
                try:
                    test = inpar['P_0'].to(unit)
                except:
                    raise TypeError("P_0 should be homogeneous to "+unit)

            if inpar['P_0'] < 0:
                raise ValueError("P_0 should be >=0")

            if 'r_p' in inpar.keys():
                try:
                    test = inpar['r_p'].to('kpc')
                except:
                    raise TypeError("r_p should be homogeneous to kpc")
                
                if inpar['r_p'] <= 0:
                    raise ValueError("r_p should be >0")

            if 'c500' in inpar.keys():
                if inpar['c500'] <=0:
                    raise ValueError("c500 should be > 0")

            # All good at this stage, setting parameters
            if 'c500' in inpar.keys():
                c500 = inpar['c500']
                r_p = self._R500/inpar['c500']
            if 'r_p' in inpar.keys():
                c500 = (self._R500/inpar['r_p']).to_value('')
                r_p = inpar['r_p']
            if hasunit:
                P0 = inpar['P_0'].to(unit)
            else:
                P0 = inpar['P_0']*u.adu
                
            outpar = {"name": 'GNFW',
                      "P_0" : P0,
                      "c500": c500,
                      "r_p" : r_p.to('kpc'),
                      "a":inpar['a'],
                      "b":inpar['b'],
                      "c":inpar['c']}
            
        #---------- Deal with the case of SVM
        if inpar['name'] == 'SVM':
            # Check the content of the dictionary
            cond1 = 'n_0' in inpar.keys() and 'r_c' in inpar.keys() and 'beta' in inpar.keys()
            cond2 = 'r_s' in inpar.keys() and 'alpha' in inpar.keys() and 'epsilon' in inpar.keys() and 'gamma' in inpar.keys()
            if not (cond1 and cond2):
                raise ValueError("The SVM model should contain: {'n_0','beta','r_c','r_s', 'alpha', 'gamma', 'epsilon'}.")
 
            # Check units
            if hasunit:
                try:
                    test = inpar['n_0'].to(unit)
                except:
                    raise TypeError("n_0 should be homogeneous to "+unit)
            try:
                test = inpar['r_c'].to('kpc')
            except:
                raise TypeError("r_c should be homogeneous to kpc")
            try:
                test = inpar['r_s'].to('kpc')
            except:
                raise TypeError("r_s should be homogeneous to kpc")
            
            # Check values
            if inpar['n_0'] < 0:
                raise ValueError("n_0 should be >= 0")
            if inpar['r_c'] <= 0:
                raise ValueError("r_c should be larger than 0")
            if inpar['r_s'] <= 0:
                raise ValueError("r_s should be larger than 0")

            if hasunit:
                n0 = inpar['n_0'].to(unit)
            else:
                n0 = inpar['n_0']*u.adu
            
            # All good at this stage, setting parameters
            outpar = {"name"    : 'SVM',
                      "n_0"    : n0,
                      "r_c"    : inpar['r_c'].to('kpc'),
                      "r_s"    : inpar['r_s'].to('kpc'),
                      "alpha"  : inpar['alpha'],
                      "beta"   : inpar['beta'],
                      "gamma"  : inpar['gamma'],
                      "epsilon": inpar['epsilon']}
            
        #---------- Deal with the case of beta
        if inpar['name'] == 'beta':
            # Check the content of the dictionary
            cond1 = 'n_0' in inpar.keys() and 'r_c' in inpar.keys() and 'beta' in inpar.keys()
            if not cond1:
                raise ValueError("The beta model should contain: {'n_0','beta','r_c'}.")

            # Check units
            if hasunit:
                try:
                    test = inpar['n_0'].to(unit)
                except:
                    raise TypeError("n_0 should be homogeneous to "+unit)
            try:
                test = inpar['r_c'].to('kpc')
            except:
                raise TypeError("r_c should be homogeneous to kpc")

            # Check values
            if inpar['n_0'] < 0:
                raise ValueError("n_0 should be >= 0")
            if inpar['r_c'] <= 0:
                raise ValueError("r_c should be larger than 0")                   

            if hasunit:
                n0 = inpar['n_0'].to(unit)
            else:
                n0 = inpar['n_0']*u.adu
                
            # All good at this stage, setting parameters
            outpar = {"name"    : 'beta',
                      "n_0"    : n0,
                      "r_c"    : inpar['r_c'].to('kpc'),
                      "beta"   : inpar['beta']}
            
        #---------- Deal with the case of doublebeta
        if inpar['name'] == 'doublebeta':
            # Check the content of the dictionary
            cond1 = 'n_01' in inpar.keys() and 'r_c1' in inpar.keys() and 'beta1' in inpar.keys()
            cond2 = 'n_02' in inpar.keys() and 'r_c2' in inpar.keys() and 'beta2' in inpar.keys()
            if not (cond1 and cond2):
                raise ValueError("The double beta model should contain: {'n_01','beta1','r_c1','n_02','beta2','r_c2'}.")

            # Check units
            if hasunit:
                try:
                    test = inpar['n_01'].to(unit)
                except:
                    raise TypeError("n_01 should be homogeneous to "+unit)
                try:
                    test = inpar['n_02'].to(unit)
                except:
                    raise TypeError("n_02 should be homogeneous to "+unit)
            try:
                test = inpar['r_c1'].to('kpc')
            except:
                raise TypeError("r_c1 should be homogeneous to kpc")
            try:
                test = inpar['r_c2'].to('kpc')
            except:
                raise TypeError("r_c2 should be homogeneous to kpc")

            # Check values
            if inpar['n_01'] < 0:
                raise ValueError("n_01 should be >= 0")
            if inpar['r_c1'] <= 0:
                raise ValueError("r_c1 should be larger than 0")
            if inpar['n_02'] < 0:
                raise ValueError("n_02 should be >= 0")
            if inpar['r_c2'] <= 0:
                raise ValueError("r_c2 should be larger than 0")

            if hasunit:
                n01 = inpar['n_01'].to(unit)
                n02 = inpar['n_02'].to(unit)
            else:
                n01 = inpar['n_01']*u.adu
                n02 = inpar['n_02']*u.adu
            
            # All good at this stage, setting parameters
            outpar = {"name"    : 'doublebeta',
                      "n_01"    : n01,
                      "r_c1"    : inpar['r_c1'].to('kpc'),
                      "beta1"   : inpar['beta1'],
                      "n_02"    : n02,
                      "r_c2"    : inpar['r_c2'].to('kpc'),
                      "beta2"   : inpar['beta2']}

        return outpar


    #==================================================
    # Validate spectrum model parameters
    #==================================================
    
    def _validate_spectrum_model_parameters(self, inpar, unit):
        """
        Check the spectrum parameters.
        
        Parameters
        ----------
        - inpar (dict): a dictionary containing the input parameters
        - unit (str): contain the unit of the spectrum, e.g. GeV-1 cm-3 for proton
        
        Outputs
        ----------
        - outpar (dict): a dictionary with output parameters

        """

        # List of available authorized models
        model_list = ['PowerLaw', 'ExponentialCutoffPowerLaw']
        
        # Deal with unit
        if unit == '' or unit == None:
            hasunit = False
        else:
            hasunit = True

        # Check that the input is a dictionary
        if type(inpar) != dict :
            raise TypeError("The model should be a dictionary containing the name key and relevant parameters")
        
        # Check that input contains a name
        if 'name' not in inpar.keys() :
            raise ValueError("The model dictionary should contain a 'name' field")
            
        # Check that the name is in the acceptable name list        
        if not inpar['name'] in model_list:
            print('The spectrum model can be:')
            print(model_list)
            raise ValueError("The requested model is not available")
            
        #---------- Deal with the case of PowerLaw
        if inpar['name'] == 'PowerLaw':
            # Check the content of the dictionary
            cond1 = 'Index' in inpar.keys()            
            if not cond1:
                raise ValueError("The PowerLaw model should contain: {'Index'}.")

            # All good at this stage, setting parameters
            outpar = {"name" : 'PowerLaw',
                      "Index": inpar['Index']}
            
        #---------- Deal with the case of ExponentialCutoffPowerLaw
        if inpar['name'] == 'ExponentialCutoffPowerLaw':
            # Check the content of the dictionary
            cond1 = 'Index' in inpar.keys() and 'CutoffEnergy' in inpar.keys()
            if not cond1:
                raise ValueError("The ExponentialCutoffPowerLaw model should contain: {'Index', 'CutoffEnergy'}.")

            # Check units
            try:
                test = inpar['CutoffEnergy'].to('TeV')
            except:
                raise TypeError("CutoffEnergy should be homogeneous to TeV")
            
            # Check values
            if inpar['CutoffEnergy'] < 0:
                raise ValueError("CutoffEnergy should be <= 0")   
            
            # All good at this stage, setting parameters
            outpar = {"name"        : 'ExponentialCutoffPowerLaw',
                      "Index"       : inpar['Index'],
                      "CutoffEnergy": inpar['CutoffEnergy'].to('TeV')}
            
        return outpar

        
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
            tab['x_crp'] = Column(prof.to_value('adu'), unit='adu',
                                  description='Enclosed cosmic ray to thermal energy')
            self._save_txt_file(self._output_dir+'/PROFILE_fraction_energy_cosmic_to_thermal.txt',
                                radius.to_value('kpc'), prof.to_value('adu'),
                                'radius (kpc)', 'x')

        #---------- Gamma ray profile
        if 'all' in prod_list or 'gamma' in prod_list:
            rad, prof = self.get_gamma_profile(radius, Emin=Egmin, Emax=Egmax, Energy_density=Energy_density,
                                               NR500max=NR500max, Npt_los=Npt_los)
            if Energy_density:
                tab['Sgamma'] = Column(prof.to_value('GeV cm-2 s-1 sr-1'), unit='GeV cm-2 s-1 sr-1',
                                       description='Gamma ray surface brightness')
                self._save_txt_file(self._output_dir+'/PROFILE_gamma_ray_surface_brightness.txt',
                                    radius.to_value('kpc'), prof.to_value('GeV cm-2 s-1 sr-1'), 'radius (kpc)',
                                    'gamma SB (GeV cm-2 s-1 sr-1)')
            else:
                tab['Sgamma'] = Column(prof.to_value('cm-2 s-1 sr-1'), unit='cm-2 s-1 sr-1',
                                       description='Gamma ray surface brightness')
                self._save_txt_file(self._output_dir+'/PROFILE_gamma_ray_surface_brightness.txt',
                                    radius.to_value('kpc'), prof.to_value('cm-2 s-1 sr-1'), 'radius (kpc)',
                                    'gamma SB (cm-2 s-1 sr-1)')

        #---------- Spherically integrated X flux
        if 'all' in prod_list or 'fx_sph' in prod_list:
            if os.path.exists(self._output_dir+'/XSPEC_table.txt'):
                rad, prof = self.get_fxsph_profile(radius, output_type=Sx_type)
                tab['fx_sph'] = Column(prof.to_value('erg s-1 cm-2'), unit='erg s-1 cm-2',
                                       description='Spherically integrated Xray fux')
                self._save_txt_file(self._output_dir+'/PROFILE_xray_flux_spherical.txt',
                                    radius.to_value('kpc'), prof.to_value('erg s-1 cm-2'), 'radius (kpc)',
                                    'Fx sph (erg s-1 cm-2)')
            else:
                print('!!! WARNING: XSPEC_table.txt not generated, skip fx_sph')
                
        #---------- Cylindrically integrated X flux
        if 'all' in prod_list or 'fx_cyl' in prod_list:
            if os.path.exists(self._output_dir+'/XSPEC_table.txt'):
                rad, prof = self.get_fxcyl_profile(radius, NR500max=NR500max, Npt_los=Npt_los, output_type=Sx_type)
                tab['fx_cyl'] = Column(prof.to_value('erg s-1 cm-2'), unit='erg s-1 cm-2',
                                       description='Cylindrically integrated Xray flux')
                self._save_txt_file(self._output_dir+'/PROFILE_xray_flux_cylindrical.txt',
                                    radius.to_value('kpc'), prof.to_value('erg s-1 cm-2'), 'radius (kpc)',
                                    'Fx cyl (erg s-1 cm-2)')
            else:
                print('!!! WARNING: XSPEC_table.txt not generated, skip fx_cyl')
                
        #---------- Sx
        if 'all' in prod_list or 'sx' in prod_list:
            if os.path.exists(self._output_dir+'/XSPEC_table.txt'):
                rad, prof = self.get_sx_profile(radius, NR500max=NR500max, Npt_los=Npt_los, output_type=Sx_type)
                tab['sx'] = Column(prof.to_value('erg s-1 cm-2 sr-1'), unit='erg s-1 cm-2 sr-1',
                                   description='Xray surface brightness')
                self._save_txt_file(self._output_dir+'/PROFILE_xray_surface_brightness.txt',
                                    radius.to_value('kpc'), prof.to_value('erg s-1 cm-2 sr-1'), 'radius (kpc)',
                                    'Sx (erg s-1 cm-2 sr-1)')
            else:
                print('!!! WARNING: XSPEC_table.txt not generated, skip sx')
                
        # Save the data frame in a single file as well
        tab.meta['comments'] = ['Proton spectra are integrated within '+str(Epmin)+' and '+str(Epmax)+'.',
                                'Gamma ray spectra are integrated within '+str(Egmin)+' and '+str(Egmax)+'.',
                                'The projection of line-of-sight integrated profiles stops at '+str(NR500max)+' R500.',
                                'The number of points for the line-of-sight integration is '+str(Npt_los)+'.']
        tab.write(self._output_dir+'/PROFILE.fits', overwrite=True)
        
        
    #==================================================
    # Save spectra
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
            hdu.header.add_comment('Compton parameter ymap')
            hdu.header.add_comment('Adimensional unit')
            hdu.writeto(self._output_dir+'/MAP_y_sz.fits', overwrite=True)

        #---------- gamma map
        if 'all' in prod_list or 'gamma_map' in prod_list: 
            image = self.get_gamma_template_map(NR500max=NR500max, Npt_los=Npt_los).to_value('sr-1')

            hdu = fits.PrimaryHDU(header=header)
            hdu.data = image
            hdu.header.add_comment('Gamma-ray template map')
            hdu.header.add_comment('Unit = sr-1')
            hdu.writeto(self._output_dir+'/MAP_gamma_template.fits', overwrite=True)

       #---------- Sx map
        if 'all' in prod_list or 'sx_map' in prod_list:
            if os.path.exists(self._output_dir+'/XSPEC_table.txt'):
                image = self.get_sxmap(NR500max=NR500max, Npt_los=Npt_los).to_value('erg s-1 cm-2 sr-1')
                
                hdu = fits.PrimaryHDU(header=header)
                hdu.data = image
                hdu.header.add_comment('Xray surface brightness map')
                hdu.header.add_comment('Unit = erg s-1 cm-2 sr-1')
                hdu.writeto(self._output_dir+'/MAP_Sx.fits', overwrite=True)
            else:
                print('!!! WARNING: XSPEC_table.txt not generated, skip sx_map')
                
            
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
        
        Outputs
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
