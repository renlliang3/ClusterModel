"""
This script contains plotting tools associated with the clusterclass.

"""

#==================================================
# Requested imports and style
#==================================================

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import SymLogNorm
import astropy.units as u
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

cta_energy_range   = [0.02, 100.0]*u.TeV
fermi_energy_range = [0.1, 300.0]*u.GeV

def set_default_plot_param():
    
    dict_base = {'font.size':        16, 
                 'legend.fontsize':  16,
                 'xtick.labelsize':  16,
                 'ytick.labelsize':  16,
                 'axes.labelsize':   16,
                 'axes.titlesize':   16,
                 'figure.titlesize': 16,
                 'figure.figsize':[8.0, 6.0],
                 'figure.subplot.right':0.97,
                 'figure.subplot.left':0.18, # Ensure enough space on the left so that all plot can be aligned
                 'font.family':'serif',
                 'figure.facecolor': 'white',
                 'legend.frameon': True}

    plt.rcParams.update(dict_base)

#==================================================
# Plot radial profiles
#==================================================

def profile(radius, angle, prof, filename, label='Profile', R500=None):
    """
    Plot the profiles
    
    Parameters
    ----------
    - radius (quantity): homogeneous to kpc
    - angle (quantity): homogeneous to deg
    - prof (quantity): any profile
    - label (str): the full name of the profile
    - filename (str): the full path name of the profile
    - R500 (quantity): homogeneous to kpc

    """

    p_unit = prof.unit
    r_unit = radius.unit
    t_unit = angle.unit
    
    fig, ax1 = plt.subplots()
    ax1.plot(radius, prof, 'blue')
    if R500 != None:
        ax1.axvline(R500.to_value(r_unit), ymin=-1e300, ymax=1e300,
                    color='black', label='$R_{500}$', linestyle='--')
    ax1.set_xlabel('Radius ('+str(r_unit)+')')
    ax1.set_ylabel(label)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlim([np.amin(radius.to_value()), np.amax(radius.to_value())])
    ax1.legend()
    
    # Add extra projected radius axis
    ax2 = ax1.twiny()
    ax2.plot(angle, prof, 'blue')
    ax2.set_xlabel('Radius ('+str(t_unit)+')', color='k')
    ax2.set_xscale('log')
    ax2.set_xlim([np.amin(angle.to_value()),np.amax(angle.to_value())])
    fig.savefig(filename)
    plt.close()

    
#==================================================
# Plot spectra
#==================================================

def spectra(energy, spec, filename, label='Spectrum'):
    """
    Plot the profiles
    
    Parameters
    ----------
    - energy (quantity): homogeneous to GeV
    - sepc (quantity): any spectrum
    - sepc_label (str): the full name of the sepctrum
    - filename (str): the full path name of the profile

    """

    s_unit = spec.unit
    e_unit = energy.unit

    ymin = np.amin(spec[spec>0].to_value())*0.5
    ymax = np.amax(spec[spec>0].to_value())*2.0
    
    fig, ax = plt.subplots()
    ax.plot(energy, spec, 'black')
    ax.fill_between(cta_energy_range.to_value(e_unit), -1e300, 1e300,
                    facecolor='blue', alpha=0.2, label='CTA range')
    ax.fill_between(fermi_energy_range.to_value(e_unit), -1e300, 1e300,
                    facecolor='red', alpha=0.2, label='Fermi range')
    ax.set_xlabel('Energy ('+str(e_unit)+')')
    ax.set_ylabel(label)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim([ymin,ymax])
    ax.set_xlim([np.amin(energy.to_value()), np.amax(energy.to_value())])
    plt.legend()
    fig.savefig(filename)
    plt.close()


#==================================================
# Plot spectra
#==================================================

def maps(image, header, filename,
         label='Map', coord=None, theta_500=None,
         theta_trunc=None, logscale=False):
    """
    Plot the profiles
    
    Parameters
    ----------
    - image (np.2darray): the map
    - header (str): corresponding header
    - filename (str): the full path name of the profile
    - theta_500 (quantity): angle corresponding to R500
    - theta_trunc (quantity): angle corresponding to the truncation

    """

    plt.rcParams.update({'figure.subplot.right':0.90,
                         'figure.subplot.left':0.05})

    wcs_map = WCS(header)

    #---------- Check the map scale
    if np.amin(image) == np.amax(image):
        logscale = False
        print('WARNING: the image is empty. You may have set the map coordinates far away from the cluster center.')
    
    #---------- Plot the map
    fig = plt.figure()
    ax = plt.subplot(projection=wcs_map)
    if logscale:
        plt.imshow(image, origin='lower', cmap='magma', norm=SymLogNorm(1))
    else:
        plt.imshow(image, origin='lower', cmap='magma')
        
    if coord != None and theta_500 != None:
        circle = matplotlib.patches.Ellipse((coord.icrs.ra.deg, coord.icrs.dec.deg),
                                            2*theta_500.to_value('deg')/np.cos(coord.icrs.dec.rad),
                                            2*theta_500.to_value('deg'),
                                            linewidth=2, fill=False, zorder=2,
                                            edgecolor='white', linestyle='-.',
                                            facecolor='none', transform=ax.get_transform('fk5'))
        ax.add_patch(circle)
        txt = plt.text(coord.icrs.ra.deg - theta_500.to_value('deg'),
                       coord.icrs.dec.deg - theta_500.to_value('deg'),
                       '$R_{500}$',
                       transform=ax.get_transform('fk5'), fontsize=10, color='white',
                       horizontalalignment='center',verticalalignment='center')

    if coord != None and theta_trunc != None:
        circle = matplotlib.patches.Ellipse((coord.icrs.ra.deg, coord.icrs.dec.deg),
                                            2*theta_trunc.to_value('deg')/np.cos(coord.icrs.dec.rad),
                                            2*theta_trunc.to_value('deg'),
                                            linewidth=2, fill=False, zorder=2,
                                            edgecolor='white', linestyle='--',
                                            facecolor='none', transform=ax.get_transform('fk5'))
        ax.add_patch(circle)
        txt = plt.text(coord.icrs.ra.deg - theta_trunc.to_value('deg'),
                       coord.icrs.dec.deg - theta_trunc.to_value('deg'),
                       '$R_{trunc}$',
                       transform=ax.get_transform('fk5'), fontsize=10, color='white',
                       horizontalalignment='center',verticalalignment='center')
        
    ax.set_xlabel('R.A. (deg)')
    ax.set_ylabel('Dec. (deg)')
    cbar = plt.colorbar()
    cbar.set_label(label)
    fig.savefig(filename)
    plt.close()
    set_default_plot_param()

#==================================================
# Main function
#==================================================

def main(cluster, list_prod=['all'],
         radius=np.logspace(0,4,1000)*u.kpc, energy=np.logspace(-2,6,1000)*u.GeV,
         NR500max=5.0, Npt_los=100, Rmax=None,
         Epmin=None, Epmax=None, Egmin=10.0*u.MeV, Egmax=1.0*u.PeV):
    """
    Main function of the sub-module of the cluster class dedicated to plots.
    
    Parameters
    ----------
    - cluster: am object from the cluster class
    - list_prod (list): the list of what is required for production
    - radius (quantity) : the physical radius
    - energy (quantity) : the physical energy of CR protons
    - NR500max (float): the integration will stop at NR500max x R500
    Only used for projected profiles.
    - Npt_los (int): the number of points for line of sight integration
    Only used for projected profiles.
    - Epmin (quantity): the lower bound for energy proton integration
    - Epmax (quantity): the upper bound for energy proton integration
    - Egmin (quantity): the lower bound for energy gamma integration
    - Egmax (quantity): the upper bound for energy gamma integration
    - Rmax (quantity): the radius within with the spectrum is computed 
    (default is R500)

    """
    
    if Epmin == None:
        Epmin = cluster._Epmin
    if Epmax == None:
        Epmax = cluster._Epmax

    if Rmax == None:
        Rmax = cluster._R500

    
    outdir = cluster._output_dir

    set_default_plot_param()

    #---------- Profiles
    if 'all' in list_prod or 'profile' in list_prod:
        angle = (radius.to_value('kpc')/cluster._D_ang.to_value('kpc')*180.0/np.pi)*u.deg

        # Pressure
        rad, prof = cluster.get_pressure_gas_profile(radius)
        profile(radius, angle, prof.to('keV cm-3'), cluster._output_dir+'/PLOT_PROF_gas_pressure.pdf',
                label='Electron pressure (keV cm$^{-3}$)', R500=cluster._R500)

        # Density
        rad, prof = cluster.get_density_gas_profile(radius)
        profile(radius, angle, prof.to('cm-3'), cluster._output_dir+'/PLOT_PROF_gas_density.pdf',
                label='Electron density (cm$^{-3}$)', R500=cluster._R500)

        # temperature
        rad, prof = cluster.get_temperature_gas_profile(radius)
        profile(radius, angle, prof.to('keV'), cluster._output_dir+'/PLOT_PROF_gas_temperature.pdf',
                label='Temperature (keV)', R500=cluster._R500)

        # Entropy
        rad, prof = cluster.get_entropy_gas_profile(radius)
        profile(radius, angle, prof.to('keV cm2'), cluster._output_dir+'/PLOT_PROF_gas_entropy.pdf',
                label='Entropy (keV cm$^2$)', R500=cluster._R500)

        # Masse HSE
        rad, prof = cluster.get_hse_mass_profile(radius)
        profile(radius, angle, prof.to('Msun'), cluster._output_dir+'/PLOT_PROF_hse_mass.pdf',
                label='HSE mass (M$_{\\odot}$)', R500=cluster._R500)

        # Overdensity
        rad, prof = cluster.get_overdensity_contrast_profile(radius)
        profile(radius, angle, prof.to('adu'), cluster._output_dir+'/PLOT_PROF_overdensity.pdf',
                label='Overdensity $\\rho / \\rho_{c}$', R500=cluster._R500)

        # Gas mass
        rad, prof = cluster.get_gas_mass_profile(radius)
        profile(radius, angle, prof.to('Msun'), cluster._output_dir+'/PLOT_PROF_gas_mass.pdf',
                label='Gas mass (M$_{\\odot}$)', R500=cluster._R500)

        # fgas profile
        rad, prof = cluster.get_fgas_profile(radius)
        profile(radius, angle, prof.to('adu'), cluster._output_dir+'/PLOT_PROF_fgas.pdf',
                label='Gas fraction', R500=cluster._R500)

        # Thermal energy
        rad, prof = cluster.get_thermal_energy_profile(radius)
        profile(radius, angle, prof.to('erg'), cluster._output_dir+'/PLOT_PROF_thermal_energy.pdf',
                label='Thermal energy (erg)', R500=cluster._R500)

        # Spherically integrated Compton
        rad, prof = cluster.get_ysph_profile(radius)
        profile(radius, angle, prof.to('kpc2'), cluster._output_dir+'/PLOT_PROF_Ysph.pdf',
                label='Y spherical (kpc$^2$)', R500=cluster._R500)

        # Cylindrically integrated Compton
        rad, prof = cluster.get_ycyl_profile(radius, NR500max=NR500max, Npt_los=Npt_los)
        profile(radius, angle, prof.to('kpc2'), cluster._output_dir+'/PLOT_PROF_Ycyl.pdf',
                label='Y cylindrical (kpc$^2$)', R500=cluster._R500)

        # Compton parameter
        rad, prof = cluster.get_y_compton_profile(radius, NR500max=NR500max, Npt_los=Npt_los)
        profile(radius, angle, prof.to('adu'), cluster._output_dir+'/PLOT_PROF_ycompton.pdf',
                label='y Compton', R500=cluster._R500)

        # Cosmic ray proton
        rad, prof = cluster.get_density_crp_profile(radius, Emin=Epmin, Emax=Epmax, Energy_density=False)
        profile(radius, angle, prof.to('cm-3'), cluster._output_dir+'/PLOT_PROF_crp_density.pdf',
                label='CRp density (cm$^{-3}$)', R500=cluster._R500)
            
        # Cosmic ray to thermal energy
        rad, prof = cluster.get_crp_to_thermal_energy_profile(radius, Emin=Epmin, Emax=Epmax)
        profile(radius, angle, prof.to('adu'), cluster._output_dir+'/PLOT_PROF_crp_fraction.pdf',
                label='CRp to thermal energy $X_{CR}$', R500=cluster._R500)

        # Gamma ray profile
        rad, prof = cluster.get_gamma_profile(radius, Emin=Egmin, Emax=Egmax, Energy_density=False, NR500max=NR500max, Npt_los=Npt_los)
        profile(radius, angle, prof.to('cm-2 s-1 sr-1'), cluster._output_dir+'/PLOT_PROF_SBgamma.pdf',
                label='$\\gamma$-ray surface brightness (cm$^{-2}$ s$^{-1}$ sr$^{-1}$)', R500=cluster._R500)
        
    #---------- Spectra
    if 'all' in list_prod or 'spectra' in list_prod:
        # CR protons
        eng, spec = cluster.get_crp_spectrum(energy, Rmax=Rmax)
        spectra(energy, spec.to('GeV-1'), cluster._output_dir+'/PLOT_SPEC_CRproton.pdf', label='Volume integrated CRp (GeV$^{-1}$)')

        # Spherically integrated gamma ray
        eng, spec = cluster.get_gamma_spectrum(energy, Rmax=Rmax, type_integral='spherical', NR500max=NR500max, Npt_los=Npt_los)
        spectra(energy, (energy**2*spec).to('GeV cm-2 s-1'), cluster._output_dir+'/PLOT_SPEC_Fgamma_sph.pdf',
                label='Spherical $F_{\\gamma}$ (GeV cm$^{-2}$ s$^{-1}$)')

        # Cylindrically integrated gamma ray
        eng, spec = cluster.get_gamma_spectrum(energy, Rmax=Rmax, type_integral='cylindrical', NR500max=NR500max, Npt_los=Npt_los)
        spectra(energy, (energy**2*spec).to('GeV cm-2 s-1'), cluster._output_dir+'/PLOT_SPEC_Fgamma_cyl.pdf',
                label='Cylindrical $F_{\\gamma}$ (GeV cm$^{-2}$ s$^{-1}$)')

    #---------- Map
    if 'all' in list_prod or 'map' in list_prod:
        header = cluster.get_map_header()

        # ymap
        image = cluster.get_ymap().to_value('adu')
        maps(image*1e6, header, cluster._output_dir+'/PLOT_MAP_ycompon.pdf',
             label='Compton parameter $\\times 10^{6}$', coord=cluster._coord, theta_500=cluster._theta500, theta_trunc=cluster._theta_truncation, logscale=True)
        
        # ymap    
        image = cluster.get_gamma_template_map().to_value('sr-1')
        maps(image, header, cluster._output_dir+'/PLOT_MAP_gamma_template.pdf', label='$\\gamma$-ray template (sr$^{-1}$)',
             coord=cluster._coord, theta_500=cluster._theta500, theta_trunc=cluster._theta_truncation, logscale=True)

