import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import scipy.ndimage as ndimage
import scipy.interpolate as interpolate
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
import healpy

#===================================================
#========== EXTRACT AND REWRITE FITS MAP
#===================================================
def re_write_fits(in_file, out_file,
                  no_neg=False,
                  smooth_arcmin=0.0,
                  rm_median=False,
                  flag_dist_arcmin=0.0,
                  normalize=False) :
    """ 
    PURPOSE: This function extracts a map from a fits fil and modifies it 
    applying smoothing, baseline removal, flag of negative values, 
    and masking. The new image is then writtin in a new file.

    INPUT: - in_file (string): input file full name
           - out_file (string): (output file full name)
           - no_deg (bool): set negative values to zero (default is no)
           - smooth_arcmin (float): Gaussian smoothing FWHM, in arcmin, to apply
           - rm_median (bool): remove the median of the map (default is no)
           - flag_dist_arcmin (float): set to zero pixels beyond this limit, 
             from the center
           - normalize (bool): set this keyword to normalize the map to 1 (i.e.
             setting the integral sum(map)*reso^2 to 1)

    OUTPUT: - The new map is written in a new file
            - image (2d numpy array): the new map
            - header : the corresponding map header
    """

    #---------- Data extraction
    data = fits.open(in_file)[0]
    image = data.data
    wcs = WCS(data.header)
    reso_x = abs(wcs.wcs.cdelt[0])
    reso_y = abs(wcs.wcs.cdelt[1])
    Npixx = image.shape[0]
    Npixy = image.shape[1]
    fov_x = Npixx * reso_x
    fov_y = Npixy * reso_y
    
    #---------- Data modification
    if smooth_arcmin >= 0:
        sigma_sm = smooth_arcmin/60.0/np.array([reso_x, reso_x])/(2*np.sqrt(2*np.log(2)))
        image = ndimage.gaussian_filter(image, sigma=sigma_sm)
        
    if rm_median == True:
        image = image - np.median(image)

    if no_neg == True:
        image[image < 0] = 0.0

    if flag_dist_arcmin > 0:
        if Npixx/2.0 != int(Npixx/2.0): axisx = np.arange(-(Npixx-1.0)//2.0, ((Npixx-1.0)//2.0)+1.0)
        else: axisx = np.arange(-(Npixx-1.0)//2.0, ((Npixx-1.0)//2.0)+1.0) + 0.5
        if Npixy/2.0 != int(Npixy/2.0): axisy = np.arange(-(Npixy-1.0)//2.0, ((Npixy-1.0)//2.0)+1.0)
        else: axisy = np.arange(-(Npixy-1.0)//2.0, ((Npixy-1.0)//2.0)+1.0) + 0.5
        coord_y, coord_x = np.meshgrid(axisx, axisy, indexing='ij')
        radius_map = np.sqrt((coord_x * reso_x)**2 + (coord_y * reso_y)**2)
        image[radius_map > flag_dist_arcmin/60.0] = 0.0

    if normalize == True:
        norm = np.sum(image) * reso_x * reso_y * (np.pi/180.0)**2
        image = image/norm
        
    #---------- WCS construction
    w = WCS(naxis=2)
    w.wcs.crpix = wcs.wcs.crpix
    w.wcs.cdelt = wcs.wcs.cdelt
    w.wcs.crval = wcs.wcs.crval
    w.wcs.latpole = wcs.wcs.latpole
    w.wcs.lonpole = wcs.wcs.lonpole
    if (wcs.wcs.ctype[0] == "RA--TAN") or (wcs.wcs.ctype[1] == "DEC-TAN"):
        w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    else:
        w.wcs.ctype = wcs.wcs.ctype

    #---------- Write FITS
    header = w.to_header()
    hdu = fits.PrimaryHDU(header=header)
    hdu.data = image
    hdu.writeto(out_file, overwrite=True)
    
    return image, header

#===================================================
#========== Give the map normalization
#===================================================
def get_map_norm(image, header) :
    """
    Measure the normalization of a map

    Parameters
    ----------
    - image: input map
    - header: input header
    
    Outputs
    --------
    - norm ([image] * sr): the integral of the map over all solid angle
    """
    w = WCS(header)
    reso_x = np.abs(w.wcs.cdelt[0])
    reso_y = np.abs(w.wcs.cdelt[1])
    
    norm = np.sum(image) * reso_x * reso_y * (np.pi/180.0)**2

    return norm

#===================================================
#========== Build standard wcs and header
#===================================================
def define_std_header(ra_center, dec_center, FoV_x, FoV_y, reso) :
    """
    Build a header and wcs object for a standard map

    Parameters
    ----------
    - ra_center (deg): coordinate of the R.A. reference center
    - dec_center (deg): coordinate of the Dec. reference center
    - FoV_x (deg): size of the map along x axis
    - FoV_y (deg): size of the map along y axis

    Outputs
    --------
    - header

    """
        
    Naxisx = int(FoV_x/reso)
    Naxisy = int(FoV_y/reso)

    # Makes it odd to have one pixel at the center
    if Naxisx/2.0 == int(Naxisx/2.0): Naxisx += 1
    if Naxisy/2.0 == int(Naxisy/2.0): Naxisy += 1

    data_tpl = np.zeros((Naxisy, Naxisx))
    
    w = WCS(naxis=2)
    w.wcs.crpix = (np.array([Naxisx, Naxisy])-1)/2+1
    w.wcs.cdelt = np.array([-reso, reso])
    w.wcs.crval = [ra_center, dec_center]
    w.wcs.latpole = 90.0
    w.wcs.lonpole = 180.0
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    header = w.to_header()
    
    hdu = fits.PrimaryHDU(header=header, data=data_tpl)
    header = hdu.header
    w = WCS(header)

    return header

#===================================================
#========== CREATE R.A. and Dec. maps from wcs
#===================================================
def get_radec_map(header):
    """
    Extract a RA and Dec map from a map header and a reference coordinate

    Parameters
    ----------
    - image: grid containing the original data
    - header: header associated to the map

    Outputs
    --------
    - ra_map (deg): map of R.A. values
    - dec_map (deg): map of Dec. values

    Example
    -------
    hdu = fits.open(file)[0]
    ra_map, dec_map = get_radec_map(hdu)

    """

    w = WCS(header)
    Naxis1 = header['NAXIS1']
    Naxis2 = header['NAXIS2']
    
    axis1 = np.arange(0, Naxis1)
    axis2 = np.arange(0, Naxis2)
    coord_x, coord_y = np.meshgrid(axis1, axis2, indexing='xy')
    world = w.wcs_pix2world(coord_x, coord_y, 0)
    
    ra_map = world[0]
    dec_map = world[1]
    
    return ra_map, dec_map

#===================================================
#========== Compute great circle angles on the sky
#===================================================
def greatcircle(lon, lat, lon_ref, lat_ref):
    """
    Compute distances between points on a sphere.

    Parameters
    ----------
    - lon (deg): longitude (should be np array)
    - lat (deg): latitude (should be np array)
    - lon_ref (deg): reference longitude
    - lat_ref (deg): reference latitude

    Outputs
    --------
    - angle (deg): np array containing the angular distance

    """
    
    arg1 = 180.0/np.pi*2
    arg2 = (np.sin((lat_ref-lat)*np.pi/180.0/2.0))**2
    arg3 = np.cos(lat*np.pi/180.0) * np.cos(lat_ref*np.pi/180.0)
    arg4 = (np.sin((lon_ref-lon)*np.pi/180.0/2.0))**2
    
    wbad1 = arg2 + arg3 * arg4 < 0
    if np.sum(wbad1) > 0: print('WARNING : '+str(np.sum(wbad1))+' Bad coord')
    
    angle = arg1 * np.arcsin(np.sqrt(arg2 + arg3 * arg4))
                             
    return angle

#===================================================
#========== Interpolate profile onto map
#===================================================
def profile2map(profile_y, profile_r, map_r):
    """
    Interpolated a profile onto a map

    Parameters
    ----------
    - profile_y: amplitude value of the profile at a given radius
    - profile_r (deg): radius corresponding to profile_y
    - map_r (deg): radius map in 2d

    Outputs
    --------
    - map_y (in units of profile_y): interpolated map

    """

    map_r_flat = np.reshape(map_r, map_r.shape[0]*map_r.shape[1])
    itpl = interpolate.interp1d(profile_r, profile_y, kind='cubic', fill_value='extrapolate')
    map_y_flat = itpl(map_r_flat)
    map_y = np.reshape(map_y_flat, (map_r.shape[0],map_r.shape[1]))
    
    return map_y

#===================================================
#========== Extract ROI from Healpix maps
#===================================================
def roi_extract_healpix(file_name, ra, dec, reso_deg, FoV_deg, save_file=None, visu=True):
    """
    Extract a sky patch on the sky given a healpix fullsky map

    Parameters
    ----------
    - file_name (str): Healpix fits file in galactic coordinates
    - ra (deg)  : the R.A. coordinate of the center
    - dec (deg) : the Dec. coordinate of the center
    - reso_deg (deg): the map resolution in degrees
    - FoV_deg (deg): the field of view of the extracted map as 2d list (x,y)
    - save_file (str): the name of the file where to save the map
    - visu (bool): visualize the map

    Outputs
    --------
    - image: the extracted map
    - head: the corresponding header
    
    """

    #======== Preparation
    FoV_x = FoV_deg[0]
    FoV_y = FoV_deg[1]
    coord = SkyCoord(ra, dec, frame="icrs", unit="deg")
        
    #======== Map header and geometry
    head_roi = define_std_header(ra, dec, FoV_x, FoV_y, reso_deg) 
    
    #======== Read the healpix map
    image_hp, head_hp = healpy.fitsfunc.read_map(file_name, field=0, hdu=1, h=True, verbose=False)

    #======== Show the full sky map
    if visu:
        healpy.visufunc.mollview(map=image_hp,
                                 xsize=800,
                                 min=None, max=None,
                                 cmap='viridis',
                                 notext=False,
                                 norm='hist',
                                 hold=False,
                                 return_projected_map=False)
        healpy.visufunc.projscatter(coord.galactic.l.value, coord.galactic.b.value,
                                    lonlat=True,
                                    marker='o', s=80, facecolors='white', edgecolors='black')
        healpy.graticule()

    #======== Extract the gnomview
    if visu:
        image_roi = healpy.visufunc.gnomview(map=image_hp,
                                             coord=('G', 'C'),
                                             rot=(coord.ra.value, coord.dec.value, 0.0),
                                             xsize=head_roi['Naxis1'], ysize=head_roi['Naxis2'],
                                             reso=60.0*reso_deg,
                                             cmap='viridis',
                                             norm='hist',
                                             hold=False,
                                             return_projected_map=True,
                                             no_plot=False)
        healpy.graticule()
        
    else:
        image_roi = healpy.visufunc.gnomview(map=image_hp,
                                             coord=('G', 'C'),
                                             rot=(coord.ra.value, coord.dec.value, 0.0),
                                             xsize=head_roi['Naxis1'], ysize=head_roi['Naxis2'],
                                             reso=60.0*reso_deg,
                                             hold=False,
                                             return_projected_map=True,
                                             no_plot=True)

    #======== Save the data
    if save_file != '' and save_file != None :
        hdu = fits.PrimaryHDU(header=head_roi)
        hdu.data = image_roi.data
        hdu.writeto(save_file, overwrite=True)        

    #======== Print out the maps
    if visu:
        plt.show()        
        
    return image_roi, head_roi

#===================================================
#========== Extract ROI from Healpix maps
#===================================================
def get_healpix_dec_mask(dec_lim, nside):
    """
    Compute a Healpix binnary mask given a limit declinaison

    Parameters
    ----------
    - dec_lim (deg): the limit declinaison for the mask
    - nside : Healpix Nside argument

    Outputs
    --------
    - mask (int array): the binnary mask cut at dec_lim
    
    """

    # Compute longitude latitude maps
    ipix = np.linspace(0, healpy.nside2npix(nside), healpy.nside2npix(nside), dtype=int)
    angle = healpy.pix2ang(nside, ipix, lonlat=False)
    maplon = angle[1] * 180.0/np.pi
    maplat = 90.0 - angle[0] * 180.0/np.pi

    # Get the Dec map
    mapcoord = SkyCoord(maplon, maplat, frame="galactic", unit="deg")
    mapdec = mapcoord.icrs.dec.value

    # Compute a mask
    wmask = mapdec < dec_lim
    mask = np.zeros(healpy.nside2npix(nside))
    mask[wmask] = 1

    return mask


#===================================================
#========== Compute the radial profile of a map
#===================================================
def radial_profile(image, center, stddev=None, header=None, binsize=1.0, stat='GAUSSIAN'):
    """
    Compute the radial profile of an image

    Parameters
    ----------
    - image (2D array) : input map
    - center (tupple) : coord along x and y. In case a header is given, these
    are R.A. and Dec. in degrees, otherwise this is in pixel.
    - stddev (2D array) : the standard deviation map. In case of Gaussian statistics,
    this is the sigma of the gaussian noise distribution in each pixel. In case of 
    Poisson statistics, we have stddev = sqrt(expected counts)
    - header (string) : header that contains the astrometry
    - binsize (float): the radial bin size, in degree if the header is provided and 
    in pixel unit otherwise.
    - Stat (string): 'GAUSSIAN', 'POISSON'

    Outputs
    --------
    - r_ctr (1D array): the center of the radial bins
    - p (1D array): the profile
    - err (1D array): the uncertainty
    
    """

    #----- Use constant weight if no stddev given
    if stddev is None:
        stddev = image*0+1.0
        
    #------ Get the radius map
    if header is None:
        y, x = np.indices((image.shape))
        dist_map = np.sqrt((x-center[0])**2+(y-center[1])**2)  # in pixels
    else:
        ra_map, dec_map = get_radec_map(header)
        dist_map = greatcircle(ra_map, dec_map, center[0], center[1]) # in deg
    
    dist_max = np.max(dist_map)
    
    #----- Compute the binning
    Nbin = int(np.ceil(dist_max/binsize))
    r_in  = np.linspace(0, dist_max, Nbin+1)
    r_out = r_in + (np.roll(r_in, -1) - r_in)
    r_in  = r_in[0:-1]
    r_out = r_out[0:-1]

    #----- Compute the profile
    r_ctr = np.array([])
    p     = np.array([])
    err   = np.array([])
    for i in range(Nbin):
        r_ctr    = np.append(r_ctr, (r_in[i] + r_out[i])/2.0)
        w_ok_rad =  (dist_map < r_out[i])*(dist_map >= r_in[i])
        w_ok_val = (stddev > 0)*(np.isnan(stddev) == False)*(np.isnan(image) == False)
        w_ok     = w_ok_rad*w_ok_val
        Npix_ok  = np.sum(w_ok)

        if stat == 'GAUSSIAN':
            val     = np.sum((image/stddev**2)[w_ok]) / np.sum((1.0/stddev**2)[w_ok])
            val_err = 1.0 / np.sqrt(np.sum((1.0/stddev**2)[w_ok]))

        if stat == 'POISSON':
            cts     = np.sum(image[w_ok])
            cts_exp = np.sum((stddev**2)[w_ok]) # stddev**2 == model for poisson
            cts_dat = cts + cts_exp
            sig = np.sign(cts_dat-cts_exp)*np.sqrt(2*(cts_dat*np.log(cts_dat/cts_exp) + cts_exp - cts_dat))
            val = cts/float(Npix_ok)
            val_err = val/sig
            
        p       = np.append(p, val)
        err     = np.append(err, val_err)
        
    return r_ctr, p, err
