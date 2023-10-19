# *****************************************************************************
# Copyright (C) 2023 INAF
# This software is distributed under the terms of the BSD-3-Clause license
#
# Authors:
# Ambra Di Piano <ambra.dipiano@inaf.it>
# *****************************************************************************

import numpy as np
from astropy.wcs import WCS
from scipy.ndimage import gaussian_filter

def extract_heatmap(data, trange, smoothing, nbins):
    data = data[(data['TIME'] >= trange[0]) & (data['TIME'] <= trange[1])] 
    
    ra = data['RA'].to_numpy()
    dec = data['DEC'].to_numpy()
    
    heatmap, xe, ye = np.histogram2d(ra, dec, bins=nbins)
    heatmap = gaussian_filter(heatmap, sigma=smoothing)
    return heatmap.T

def set_wcs(point_ra, point_dec, point_ref, pixelsize):
    w = WCS(naxis=2)
    w.wcs.ctype = ['RA---CAR', 'DEC--CAR']
    w.wcs.cunit = ['deg', 'deg']
    w.wcs.crpix = [point_ref, point_ref]
    w.wcs.crval = [point_ra, point_dec]
    w.wcs.cdelt = [-pixelsize, pixelsize]
    #w.wcs.lonpole = 0.0
    #w.wcs.latpole = 67.49
    return w