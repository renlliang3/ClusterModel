"""
This file deals with 'model parameters' issues regarding the Cluster Class (e.g. GNFW parameteres)
"""
import astropy.units as u
import numpy as np

from ClusterModel.ClusterTools import cluster_global
from ClusterModel.ClusterTools import cluster_profile 
from ClusterModel.ClusterTools import cluster_spectra

#==================================================
# Admin class
#==================================================

class Modpar(object):
    """ Modpar class
    This class searves as a parser to the main Cluster class, to 
    include the subclass Modpar in this other file. All the definitions of the 
    model parameters should be here.

    Profile models are now:  ['GNFW', 'SVM', 'beta', 'doublebeta']
    Spectral models are now: ['PowerLaw', 'ExponentialCutoffPowerLaw', 'MomentumPowerLaw]

    Attributes
    ----------  
    The attributes are the same as the Cluster class, see model.py

    Methods
    ----------  
    - _validate_profile_model_parameters(self, inpar, unit): dedicated to check and validate the parameters 
    of profile models
    - _validate_spectrum_model_parameters(self, inpar, unit): dedicated to check and validate the parameters 
    of spectral models

    - set_pressure_gas_gNFW_param(self, pressure_model='P13UPP'): set the gas pressure profile parameters to the 
    universal value from different results
    - set_pressure_gas_isoT_param(self, kBT): set gas pressure profile parameters so that the cluster is isothermal
    - set_density_gas_isoT_param(self, kBT): set gas density profile parameters so that the cluster is isothermal
    - set_density_crp_isobaric_scal_param(self, scal=1.0): set CRp densitry profile parameters to have isobaric scaling
    - set_density_cre1_isobaric_scal_param(self, scal=1.0): set CRe1 densitry profile parameters to have isobaric scaling
    - set_density_crp_isodens_scal_param(self, scal=1.0): set CRp densitry profile parameters to have isodensity scaling
    - set_density_cre1_isodens_scal_param(self, scal=1.0): set CRe1 densitry profile parameters to have isodensity scaling
    - set_magfield_isobaric_scal_param(self, Bnorm, scal=0.5): set mag field profile parameters to have isobaric scaling
    - set_magfield_isodens_scal_param(self, Bnorm, scal=0.5): set mag field profile parameters to have isodensity scaling
    
    - _get_generic_profile(self, radius, model, derivative=False): get any profile base on model type

    """
    
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
                      "a"   : inpar['a'],
                      "b"   : inpar['b'],
                      "c"   : inpar['c']}
            
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
            outpar = {"name"   : 'SVM',
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
            outpar = {"name"   : 'beta',
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
            outpar = {"name"  : 'doublebeta',
                      "n_01"  : n01,
                      "r_c1"  : inpar['r_c1'].to('kpc'),
                      "beta1" : inpar['beta1'],
                      "n_02"  : n02,
                      "r_c2"  : inpar['r_c2'].to('kpc'),
                      "beta2" : inpar['beta2']}

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
        model_list = ['PowerLaw', 'ExponentialCutoffPowerLaw', 'MomentumPowerLaw',
                      'InitialInjection', 'ContinuousInjection']
        
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
                raise ValueError("CutoffEnergy should be >= 0")   
            
            # All good at this stage, setting parameters
            outpar = {"name"        : 'ExponentialCutoffPowerLaw',
                      "Index"       : inpar['Index'],
                      "CutoffEnergy": inpar['CutoffEnergy'].to('TeV')}

        #---------- Deal with the case of MomentumPowerLaw
        if inpar['name'] == 'MomentumPowerLaw':
            # Check the content of the dictionary
            cond1 = 'Index' in inpar.keys() and 'Mass' in inpar.keys()
            if not cond1:
                raise ValueError("The MomentumPowerLawModel model should contain: {'Index', 'Mass'}.")

            # The mass should be given in units homogeneous to GeV
            try:
                test = inpar['Mass'].to('GeV')
            except:
                raise TypeError("Mass should be homogeneous to GeV")
                            
            # All good at this stage, setting parameters
            outpar = {"name" : 'MomentumPowerLaw',
                      "Index": inpar['Index'],
                       "Mass": inpar['Mass']}

        #---------- Deal with the case of InitialInjection
        if inpar['name'] == 'InitialInjection':
            # Check the content of the dictionary
            cond1 = 'Index' in inpar.keys() and 'BreakEnergy' in inpar.keys()
            if not cond1:
                raise ValueError("The InitialInjection model should contain: {'Index', 'BreakEnergy'}.")

            # Check units
            try:
                test = inpar['BreakEnergy'].to('TeV')
            except:
                raise TypeError("BreakEnergy should be homogeneous to TeV")
            
            # Check values
            if inpar['BreakEnergy'] < 0:
                raise ValueError("BreakEnergy should be >= 0")   
            
            # All good at this stage, setting parameters
            outpar = {"name"       : 'InitialInjection',
                      "Index"      : inpar['Index'],
                      "BreakEnergy": inpar['BreakEnergy'].to('TeV')}

        #---------- Deal with the case of InitialInjection
        if inpar['name'] == 'ContinuousInjection':
            # Check the content of the dictionary
            cond1 = 'Index' in inpar.keys() and 'BreakEnergy' in inpar.keys()
            if not cond1:
                raise ValueError("The ContinuousInjection model should contain: {'Index', 'BreakEnergy'}.")

            # Check units
            try:
                test = inpar['BreakEnergy'].to('TeV')
            except:
                raise TypeError("BreakEnergy should be homogeneous to TeV")
            
            # Check values
            if inpar['BreakEnergy'] < 0:
                raise ValueError("BreakEnergy should be >= 0")   
            
            # All good at this stage, setting parameters
            outpar = {"name"       : 'ContinuousInjection',
                      "Index"      : inpar['Index'],
                      "BreakEnergy": inpar['BreakEnergy'].to('TeV')}
            
        return outpar

        
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
            if not self._silent: print('Setting gNFW Arnaud et al. (2010) UPP.')
            pppar = [8.403, 1.177, 0.3081, 1.0510, 5.4905]

        # Arnaud et al. (2010) : Cool-Core clusters parameters
        elif pressure_model == 'A10CC':
            if not self._silent: print('Setting gNFW Arnaud et al. (2010) cool-core.')
            pppar = [3.249, 1.128, 0.7736, 1.2223, 5.4905]

        # Arnaud et al. (2010) : Morphologically-Disturbed clusters parameters
        elif pressure_model == 'A10MD':
            if not self._silent: print('Setting gNFW Arnaud et al. (2010) morphologically disturbed.')
            pppar = [3.202, 1.083, 0.3798, 1.4063, 5.4905]

        # Planck Intermediate Paper V (2013) : Universal Pressure Profile parameters
        elif pressure_model == 'P13UPP':
            if not self._silent: print('Setting gNFW Planck coll. (2013) UPP.')
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
    # Set a given pressure isothermal profile
    #==================================================
    
    def set_pressure_gas_isoT_param(self, kBT):
        """
        Set the parameters of the pressure profile so that 
        the cluster is iso thermal
        
        Parameters
        ----------
        - kBT (quantity): isothermal temperature

        """

        # check type of temperature
        try:
            test = kBT.to('keV')
        except:
            raise TypeError("The temperature should be a quantity homogeneous to keV.")

        # Get the density parameters
        Ppar = self._density_gas_model.copy()

        # Modify the parameters depending on the model
        if self._density_gas_model['name'] == 'GNFW':
            Ppar['P_0'] = (Ppar['P_0'] * kBT).to('keV cm-3')
            
        elif self._density_gas_model['name'] == 'SVM':
            Ppar['n_0'] = (Ppar['n_0'] * kBT).to('keV cm-3')

        elif self._density_gas_model['name'] == 'beta':
            Ppar['n_0'] = (Ppar['n_0'] * kBT).to('keV cm-3')

        elif self._density_gas_model['name'] == 'doublebeta':
            Ppar['n_01'] = (Ppar['n_01'] * kBT).to('keV cm-3')
            Ppar['n_02'] = (Ppar['n_02'] * kBT).to('keV cm-3')

        else:
            raise ValueError('Problem with density model list.')

        self._pressure_gas_model = Ppar


    #==================================================
    # Set a given density isothermal profile
    #==================================================
    
    def set_density_gas_isoT_param(self, kBT):
        """
        Set the parameters of the density profile so that 
        the cluster is iso thermal
        
        Parameters
        ----------
        - kBT (quantity): isothermal temperature

        """

        # check type of temperature
        try:
            test = kBT.to('keV')
        except:
            raise TypeError("The temperature should be a quantity homogeneous to keV.")

        # Get the density parameters
        Ppar = self._pressure_gas_model.copy()

        # Modify the parameters depending on the model
        if self._pressure_gas_model['name'] == 'GNFW':
            Ppar['P_0'] = (Ppar['P_0'] / kBT).to('cm-3')
            
        elif self._pressure_gas_model['name'] == 'SVM':
            Ppar['n_0'] = (Ppar['n_0'] / kBT).to('cm-3')

        elif self._pressure_gas_model['name'] == 'beta':
            Ppar['n_0'] = (Ppar['n_0'] / kBT).to('cm-3')

        elif self._pressure_gas_model['name'] == 'doublebeta':
            Ppar['n_01'] = (Ppar['n_01'] / kBT).to('cm-3')
            Ppar['n_02'] = (Ppar['n_02'] / kBT).to('cm-3')

        else:
            raise ValueError('Problem with density model list.')

        self._density_gas_model = Ppar

        
    #==================================================
    # Set a given CRp density to isobaric profile
    #==================================================
    
    def set_density_crp_isobaric_scal_param(self, scal=1.0):
        """
        Set the parameters of the CRp density profile to 
        have isobaric conditions, i.e. CRp pressure over
        thermal pressure is constant.
        
        Parameters
        ----------
        scal (float): the scaling slope, n_CRp ~ P^scal

        """

        # Get the density parameters
        Ppar = self._pressure_gas_model.copy()

        # Modify the parameters depending on the model
        if self._pressure_gas_model['name'] == 'GNFW':
            Ppar['P_0'] = 1.0*u.adu
            Ppar['b'] *= scal
            Ppar['c'] *= scal
            
        elif self._pressure_gas_model['name'] == 'SVM':
            Ppar['n_0'] = 1.0*u.adu
            Ppar['beta'] *= scal
            Ppar['alpha'] *= scal
            Ppar['epsilon'] *= scal
            
        elif self._pressure_gas_model['name'] == 'beta':
            Ppar['n_0'] = 1.0*u.adu
            Ppar['beta'] *= scal
            
        elif self._pressure_gas_model['name'] == 'doublebeta':
            maxnorm = np.amax([Ppar['n_01'].to_value('keV cm-3'), Ppar['n_02'].to_value('keV cm-3')])
            Ppar['n_01'] = Ppar['n_01'].to_value('keV cm-3') / maxnorm *u.adu
            Ppar['n_02'] = Ppar['n_02'].to_value('keV cm-3') / maxnorm *u.adu

            if scal != 1.0:
                # In this case we have p = p1+p2 -> (p1+p2)^scal, so scal cannot be applied to individual profile
                raise ValueError('Transformation not available with doublebeta model for scal != 1.')
        else:
            raise ValueError('Problem with density model list.')

        self._density_crp_model = Ppar

    #==================================================
    # Set a given CRe1 density to isobaric profile
    #==================================================
    
    def set_density_cre1_isobaric_scal_param(self, scal=1.0):
        """
        Set the parameters of the CRe1 density profile to 
        have isobaric conditions, i.e. CRp pressure over
        thermal pressure is constant.
        
        Parameters
        ----------
        scal (float): the scaling slope, n_CRe1 ~ P^scal

        """

        # Get the density parameters
        Ppar = self._pressure_gas_model.copy()

        # Modify the parameters depending on the model
        if self._pressure_gas_model['name'] == 'GNFW':
            Ppar['P_0'] = 1.0*u.adu
            Ppar['b'] *= scal
            Ppar['c'] *= scal
            
        elif self._pressure_gas_model['name'] == 'SVM':
            Ppar['n_0'] = 1.0*u.adu
            Ppar['beta'] *= scal
            Ppar['alpha'] *= scal
            Ppar['epsilon'] *= scal
            
        elif self._pressure_gas_model['name'] == 'beta':
            Ppar['n_0'] = 1.0*u.adu
            Ppar['beta'] *= scal
            
        elif self._pressure_gas_model['name'] == 'doublebeta':
            maxnorm = np.amax([Ppar['n_01'].to_value('keV cm-3'), Ppar['n_02'].to_value('keV cm-3')])
            Ppar['n_01'] = Ppar['n_01'].to_value('keV cm-3') / maxnorm *u.adu
            Ppar['n_02'] = Ppar['n_02'].to_value('keV cm-3') / maxnorm *u.adu

            if scal != 1.0:
                # In this case we have p = p1+p2 -> (p1+p2)^scal, so scal cannot be applied to individual profile
                raise ValueError('Transformation not available with doublebeta model for scal != 1.')
        else:
            raise ValueError('Problem with density model list.')

        self._density_cre1_model = Ppar


    #==================================================
    # Set a given CRp density to isodensity profile
    #==================================================
    
    def set_density_crp_isodens_scal_param(self, scal=1.0):
        """
        Set the parameters of the CRp density profile to 
        have isodensity conditions, i.e. CRp density over
        thermal density is constant.
        
        Parameters
        ----------
        scal (float): the scaling slope, n_CRp ~ n_th^scal

        """

        # Get the density parameters
        Ppar = self._density_gas_model.copy()

        # Modify the parameters depending on the model
        if self._density_gas_model['name'] == 'GNFW':
            Ppar['P_0'] = 1.0*u.adu
            Ppar['b'] *= scal
            Ppar['c'] *= scal
            
        elif self._density_gas_model['name'] == 'SVM':
            Ppar['n_0'] = 1.0*u.adu
            Ppar['beta'] *= scal
            Ppar['alpha'] *= scal
            Ppar['epsilon'] *= scal
            
        elif self._density_gas_model['name'] == 'beta':
            Ppar['n_0'] = 1.0*u.adu
            Ppar['beta'] *= scal
            
        elif self._density_gas_model['name'] == 'doublebeta':
            maxnorm = np.amax([Ppar['n_01'].to_value('cm-3'), Ppar['n_02'].to_value('cm-3')])
            Ppar['n_01'] = Ppar['n_01'].to_value('cm-3') / maxnorm *u.adu
            Ppar['n_02'] = Ppar['n_02'].to_value('cm-3') / maxnorm *u.adu

            if scal != 1.0:
                # In this case we have p = p1+p2 -> (p1+p2)^scal, so scal cannot be applied to individual profile
                raise ValueError('Transformation not available with doublebeta model for scal != 1.')
        else:
            raise ValueError('Problem with density model list.')

        self._density_crp_model = Ppar


    #==================================================
    # Set a given CRe1 density to isodensity profile
    #==================================================
    
    def set_density_cre1_isodens_scal_param(self, scal=1.0):
        """
        Set the parameters of the CRe1 density profile to 
        have isodensity conditions, i.e. CRp density over
        thermal density is constant.
        
        Parameters
        ----------
        scal (float): the scaling slope, n_CRe1 ~ n_th^scal

        """

        # Get the density parameters
        Ppar = self._density_gas_model.copy()

        # Modify the parameters depending on the model
        if self._density_gas_model['name'] == 'GNFW':
            Ppar['P_0'] = 1.0*u.adu
            Ppar['b'] *= scal
            Ppar['c'] *= scal
            
        elif self._density_gas_model['name'] == 'SVM':
            Ppar['n_0'] = 1.0*u.adu
            Ppar['beta'] *= scal
            Ppar['alpha'] *= scal
            Ppar['epsilon'] *= scal
            
        elif self._density_gas_model['name'] == 'beta':
            Ppar['n_0'] = 1.0*u.adu
            Ppar['beta'] *= scal
            
        elif self._density_gas_model['name'] == 'doublebeta':
            maxnorm = np.amax([Ppar['n_01'].to_value('cm-3'), Ppar['n_02'].to_value('cm-3')])
            Ppar['n_01'] = Ppar['n_01'].to_value('cm-3') / maxnorm *u.adu
            Ppar['n_02'] = Ppar['n_02'].to_value('cm-3') / maxnorm *u.adu

            if scal != 1.0:
                # In this case we have p = p1+p2 -> (p1+p2)^scal, so scal cannot be applied to individual profile
                raise ValueError('Transformation not available with doublebeta model for scal != 1.')
        else:
            raise ValueError('Problem with density model list.')

        self._density_cre1_model = Ppar



    #==================================================
    # Set a given CRp density to isobaric profile
    #==================================================
    
    def set_magfield_isobaric_scal_param(self, Bnorm, scal=0.5):
        """
        Set the parameters of the magnetic field profile to 
        have isobaric conditions, i.e. magnetic pressure over
        thermal pressure is constant.
        
        Parameters
        ----------
        - Bnorm (quantity): the normalization of magnetic field
        homogeneous to micro Gauss.
        - scal (float): the scaling slope, B ~ P^scal. Default
        is 0.5 since magnetic energy scales as B^2

        """

        # Get the density parameters
        Ppar = self._pressure_gas_model.copy()

        # Modify the parameters depending on the model
        if self._pressure_gas_model['name'] == 'GNFW':
            Ppar['P_0'] = Bnorm
            Ppar['b'] *= scal
            Ppar['c'] *= scal
            
        elif self._pressure_gas_model['name'] == 'SVM':
            Ppar['n_0'] = Bnorm
            Ppar['beta'] *= scal
            Ppar['alpha'] *= scal
            Ppar['epsilon'] *= scal
            
        elif self._pressure_gas_model['name'] == 'beta':
            Ppar['n_0'] = Bnorm
            Ppar['beta'] *= scal
            
        elif self._pressure_gas_model['name'] == 'doublebeta':
            maxnorm = np.amax([Ppar['n_01'].to_value('keV cm-3'), Ppar['n_02'].to_value('keV cm-3')])
            Ppar['n_01'] = Ppar['n_01'].to_value('keV cm-3') / maxnorm * Bnorm
            Ppar['n_02'] = Ppar['n_02'].to_value('keV cm-3') / maxnorm * Bnorm

            if scal != 1.0:
                # In this case we have p = p1+p2 -> (p1+p2)^scal, so scal cannot be applied to individual profile
                raise ValueError('Transformation not available with doublebeta model for scal != 1.')
        else:
            raise ValueError('Problem with density model list.')

        self._magfield_model = Ppar


    #==================================================
    # Set a given CRp density to isodensity profile
    #==================================================
    
    def set_magfield_isodens_scal_param(self, Bnorm, scal=0.5):
        """
        Set the parameters of the magnetic field profile to 
        have isodensity conditions, i.e. mag field over
        thermal density is constant.
        
        Parameters
        ----------
        - Bnorm (quantity): the normalization of magnetic field
        homogeneous to micro Gauss.        
        - scal (float): the scaling slope, B ~ n_th^scal

        """

        # Get the density parameters
        Ppar = self._density_gas_model.copy()

        # Modify the parameters depending on the model
        if self._density_gas_model['name'] == 'GNFW':
            Ppar['P_0'] = Bnorm
            Ppar['b'] *= scal
            Ppar['c'] *= scal
            
        elif self._density_gas_model['name'] == 'SVM':
            Ppar['n_0'] = Bnorm
            Ppar['beta'] *= scal
            Ppar['alpha'] *= scal
            Ppar['epsilon'] *= scal
            
        elif self._density_gas_model['name'] == 'beta':
            Ppar['n_0'] = Bnorm
            Ppar['beta'] *= scal
            
        elif self._density_gas_model['name'] == 'doublebeta':
            maxnorm = np.amax([Ppar['n_01'].to_value('cm-3'), Ppar['n_02'].to_value('cm-3')])
            Ppar['n_01'] = Ppar['n_01'].to_value('cm-3') / maxnorm *Bnorm
            Ppar['n_02'] = Ppar['n_02'].to_value('cm-3') / maxnorm *Bnorm

            if scal != 1.0:
                # In this case we have p = p1+p2 -> (p1+p2)^scal, so scal cannot be applied to individual profile
                raise ValueError('Transformation not available with doublebeta model for scal != 1.')
        else:
            raise ValueError('Problem with density model list.')

        self._magfield_model = Ppar
        
        
    #==================================================
    # Get the generic model profile
    #==================================================

    def _get_generic_profile(self, radius, model, derivative=False):
        """
        Get the generic profile profile.
        
        Parameters
        ----------
        - radius (quantity) : the physical 3d radius in units homogeneous to kpc, as a 1d array
        - model (dict): dictionary containing the model parameters
        - derivative (bool): to get the derivative of the profile
        
        Outputs
        ----------
        - radius (quantity): the 3d radius in unit of kpc
        - p_r (quantity): the profile

        """

        model_list = ['GNFW', 'SVM', 'beta', 'doublebeta']

        if not model['name'] in model_list:
            print('The profile model can :')
            print(model_list)
            raise ValueError("The requested model has not been implemented")

        r3d_kpc = radius.to_value('kpc')

        #---------- Case of GNFW profile
        if model['name'] == 'GNFW':
            unit = model["P_0"].unit
            
            P0 = model["P_0"].to_value(unit)
            rp = model["r_p"].to_value('kpc')
            a  = model["a"]
            b  = model["b"]
            c  = model["c"]

            if derivative:
                prof_r = cluster_profile.gNFW_model_derivative(r3d_kpc, P0, rp,
                                                               slope_a=a, slope_b=b, slope_c=c) * unit*u.Unit('kpc-1')
            else:
                prof_r = cluster_profile.gNFW_model(r3d_kpc, P0, rp, slope_a=a, slope_b=b, slope_c=c)*unit

        #---------- Case of SVM model
        elif model['name'] == 'SVM':
            unit = model["n_0"].unit
            
            n0      = model["n_0"].to_value(unit)
            rc      = model["r_c"].to_value('kpc')
            rs      = model["r_s"].to_value('kpc')
            alpha   = model["alpha"]
            beta    = model["beta"]
            gamma   = model["gamma"]
            epsilon = model["epsilon"]

            if derivative:
                prof_r = cluster_profile.svm_model_derivative(r3d_kpc, n0, rc, beta,
                                                              rs, gamma, epsilon, alpha) * unit*u.Unit('kpc-1')
            else:
                prof_r = cluster_profile.svm_model(r3d_kpc, n0, rc, beta, rs, gamma, epsilon, alpha)*unit

        #---------- beta model
        elif model['name'] == 'beta':
            unit = model["n_0"].unit

            n0      = model["n_0"].to_value(unit)
            rc      = model["r_c"].to_value('kpc')
            beta    = model["beta"]

            if derivative:
                prof_r = cluster_profile.beta_model_derivative(r3d_kpc, n0, rc, beta) * unit*u.Unit('kpc-1')
            else:
                prof_r = cluster_profile.beta_model(r3d_kpc, n0, rc, beta)*unit
            
        #---------- double beta model
        elif model['name'] == 'doublebeta':
            unit1 = model["n_01"].unit
            unit2 = model["n_02"].unit

            n01      = model["n_01"].to_value(unit1)
            rc1      = model["r_c1"].to_value('kpc')
            beta1    = model["beta1"]
            n02      = model["n_02"].to_value(unit2)
            rc2      = model["r_c2"].to_value('kpc')
            beta2    = model["beta2"]

            if derivative:
                prof_r1 = cluster_profile.beta_model_derivative(r3d_kpc, n01, rc1, beta1) * unit1*u.Unit('kpc-1')
                prof_r2 = cluster_profile.beta_model_derivative(r3d_kpc, n02, rc2, beta2) * unit2*u.Unit('kpc-1')
                prof_r = prof_r1 + prof_r2
            else:
                prof_r1 = cluster_profile.beta_model(r3d_kpc, n01, rc1, beta1)*unit1
                prof_r2 = cluster_profile.beta_model(r3d_kpc, n02, rc2, beta2)*unit2
                prof_r = prof_r1 + prof_r2
            
        #---------- Otherwise nothing is done
        else :
            if not self._silent: print('The requested model has not been implemented.')

        return prof_r


    #==================================================
    # Get the generic model spectrum
    #==================================================

    def _get_generic_spectrum(self, energy, model):
        """
        Get the generic profile profile.
        
        Parameters
        ----------
        - energy (quantity) : the energy in units homogeneous to GeV, as a 1d array
        - model (dict): dictionary containing the model parameters
        
        Outputs
        ----------
        - energy (quantity): the energy in unit of GeV
        - S_E (quantity): the spectrum

        """

        model_list = ['PowerLaw', 'ExponentialCutoffPowerLaw', 'MomentumPowerLaw',
                      'InitialInjection', 'ContinuousInjection']

        if not model['name'] in model_list:
            print('The spectral model can :')
            print(model_list)
            raise ValueError("The requested model has not been implemented")

        eng_GeV = energy.to_value('GeV')

        #---------- Case of PowerLaw model
        if model['name'] == 'PowerLaw':
            index   = model["Index"]
            S_E = cluster_spectra.powerlaw_model(eng_GeV, 1.0, index)

        #---------- Case of ExponentialCutoffPowerLaw model
        elif model['name'] == 'ExponentialCutoffPowerLaw':
            index   = model["Index"]
            Ecut   = model["CutoffEnergy"].to_value('GeV')
            S_E = cluster_spectra.exponentialcutoffpowerlaw_model(eng_GeV, 1.0, index, Ecut)

        #---------- Case of MomentumPowerLaw model
        elif model['name'] == 'MomentumPowerLaw':
            index  = model["Index"]
            mass  = model["Mass"]
            S_E = cluster_spectra.momentumpowerlaw_model(eng_GeV, 1.0, index, mass=mass)

        #---------- Case of InitialInjection model
        elif model['name'] == 'InitialInjection':
            index  = model["Index"]
            Ebreak  = model["BreakEnergy"].to_value('GeV')
            S_E = cluster_spectra.initial_injection_model(eng_GeV, 1.0, index, Ebreak)
            
        #---------- Case of InitialInjection model
        elif model['name'] == 'ContinuousInjection':
            index  = model["Index"]
            Ebreak  = model["BreakEnergy"].to_value('GeV')
            S_E = cluster_spectra.continuous_injection_model(eng_GeV, 1.0, index, Ebreak)
            
        #---------- Otherwise nothing is done
        else :
            if not self._silent: print('The requested model has not been implemented.')

        return S_E

