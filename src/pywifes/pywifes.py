from __future__ import division, print_function
from astropy.coordinates import SkyCoord
from astropy.io import fits as pyfits
import astropy.units as u
import gc
from matplotlib import colormaps as cm, colors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy
import os
import pickle
import re
import scipy.interpolate as interp
import scipy.ndimage as ndimage
import scipy.signal as signal
import sys

# Pipeline imports
from pywifes.multiprocessing_utils import get_task, map_tasks
from pywifes.wifes_metadata import __version__, metadata_dir
from pywifes.wifes_imtrans import transform_data, detransform_data
from pywifes.wifes_wsol import fit_wsol_poly, evaluate_wsol_poly
from pywifes.wifes_adr import ha_degrees, dec_dms2dd, adr_x_y
from pywifes.wifes_utils import arguments, fits_scale_from_bitpix, is_halfframe, is_taros, nan_helper
from pywifes.mpfit import mpfit

# ------------------------------------------------------------------------
# NEED TO OPEN / ACCESS WIFES METADATA FILE!!
try:
    f0 = open(os.path.join(metadata_dir, "basic_wifes_metadata.pkl"), "rb")
    try:
        wifes_metadata = pickle.load(f0, fix_imports=True, encoding="latin")
    except Exception:
        wifes_metadata = pickle.load(f0)  # fix_imports doesn't work in python 2.7.
    f0.close()
except Exception as e:
    print(f"Failed to open or load wifes_metadata: {e}")
    raise

blue_slitlet_defs = wifes_metadata["blue_slitlet_defs"]
red_slitlet_defs = wifes_metadata["red_slitlet_defs"]
nslits = len(blue_slitlet_defs.keys())


# ------------------------------------------------------------------------
def cut_fits_to_half_frame(inimg_path, outimg_prefix="cut_", to_taros=False):
    """
    Cuts a FITS file to half-frame size and saves the output with a specified prefix.

    Parameters
    ----------
    inimg_path : str
        Path to the input FITS file.
    outimg_prefix : str, optional
        Prefix for the output FITS file. Default is "cut\_".
    to_taros : bool, optional
        Whether to extract the TAROS-style top half or the Automation-style middle
        half.

    Returns
    -------
    None

    Notes
    -----
    This function performs the following steps:

    1. Opens the input FITS file.
    2. Extracts the primary HDU (Header/Data Unit).
    3. Considers binning specified in the 'CCDSUM' header.
    4. Cuts the data to a section defined by the specified coordinates.
    5. Updates the 'DETSEC' keyword in the header.
    6. Writes the cut data to a new FITS file with the specified prefix.
    """

    # Open the input FITS file
    with pyfits.open(inimg_path) as hdul:
        # Get the primary HDU (Header/Data Unit)
        primary_hdu = hdul[0]

        # Extract the data and header
        data = primary_hdu.data
        header = primary_hdu.header

        # Consider binning
        bin_y = int(header["CCDSUM"].split()[1])

        # Cut the data according to the specified section
        if to_taros:
            cut_data = data[2056 // bin_y:4112 // bin_y, :]

            # Update the DETSEC in the header
            header["DETSEC"] = "[1:4202,2057:4112]"

        else:
            cut_data = data[1028 // bin_y:3084 // bin_y, :]

            # Update the DETSEC in the header
            header["DETSEC"] = "[1:4202,1029:3084]"

        # Create a new HDU with the cut data and the updated header
        cut_hdu = pyfits.PrimaryHDU(data=cut_data, header=header)

        # Generate the output filename

        dir_name, file_name = os.path.split(inimg_path)

        outimg_path = os.path.join(dir_name, outimg_prefix + file_name)
        print("Writing the cut data to the new FITS file:", outimg_path)

        # Write the cut data to the new FITS file
        cut_hdu.writeto(outimg_path, overwrite=True)


# ------------------------------------------------------------------------
def calib_to_half_frame(obs_metadata, temp_data_dir, to_taros=False):
    """
    Convert calibration files to half-frame format.

    This function takes the observation metadata and temporary data directory as input.
    It converts the calibration files to half-frame format by cutting them in half.
    The calibration types that can be converted are domeflat, twiflat, wire, and arc.
    The converted files are saved with a prefix 'cut\_' added to their names.

    Parameters
    ----------
    obs_metadata : dict
        The observation metadata containing calibration file information.
    temp_data_dir : str
        The temporary data directory where the calibration files are located.
    to_taros : bool, optional
        Whether the reference image (science, standard, or arc) needs the TAROS-style
        top half or the Automation-style middle half.

    Returns
    -------
    dict
        The updated observation metadata with the converted calibration file names.
    """

    # A list of calibration types that need to be half-frame for the pipeline to work
    # properly in the half-frame scenario.
    calib_types = ["domeflat", "twiflat", "wire", "arc", "bias", "dark"]
    prefix = "cut_"

    for calib_type in calib_types:
        file_list = obs_metadata[calib_type]
        for index, file_name in enumerate(file_list):
            calib_fits = os.path.join(temp_data_dir, file_name + ".fits")
            if not is_halfframe(calib_fits):

                cut_fits_to_half_frame(calib_fits, outimg_prefix=prefix,
                                       to_taros=to_taros)
                obs_metadata[calib_type][index] = prefix + file_name

    # Check the standard star separately because the dictionary has a slightly
    # different structure.
    if len(obs_metadata["std"]) == 0:
        return obs_metadata
    std_list = obs_metadata["std"][0]["sci"]
    print(f"Std list: {std_list}")
    for index, file_name in enumerate(std_list):
        calib_fits = os.path.join(temp_data_dir, file_name + ".fits")
        if not is_halfframe(calib_fits):
            cut_fits_to_half_frame(calib_fits, outimg_prefix=prefix, to_taros=to_taros)
            obs_metadata["std"][0]["sci"][index] = prefix + file_name
    return obs_metadata


# ------------------------------------------------------------------------
# functions that operate on data rather than images
def single_centroid_prof_fit(
    y, x=None, ctr_guess=None, width_guess=None, return_width=False
):
    """
    Fit a single Gaussian to a 1D profile to determine the centroid.

    Parameters
    ----------
    y : numpy.ndarray
        The 1D profile data.
    x : numpy.ndarray, optional
        The x-axis data. Default is None.
    ctr_guess : float, optional
        The initial guess for the centroid. Default is None.
    width_guess : float, optional
        The initial guess for the width. Default is None.
    return_width : bool, optional
        If True, the function returns the width of the Gaussian. Default is False.

    Returns
    -------
    float
        The centroid of the Gaussian.
    """
    N = len(y)
    if x is None:
        x = numpy.arange(N, dtype="d")
    # choose x,y subregions for this line
    if ctr_guess is not None and width_guess is not None:
        ifit_lo = ctr_guess - 5 * width_guess
        ifit_hi = ctr_guess + 5 * width_guess
    else:
        ifit_lo = 0
        ifit_hi = N
    xfit = x[ifit_lo:ifit_hi]
    yfit = y[ifit_lo:ifit_hi]
    new_xctr = numpy.nansum(xfit * (yfit**2)) / numpy.nansum((yfit**2))
    new_x2 = numpy.nansum(((xfit - new_xctr) ** 2) * (yfit**2)) / numpy.nansum((yfit**2))
    new_rms = new_x2**0.5
    new_sig = new_rms / 2.235
    if return_width:
        return new_xctr, new_sig
    else:
        return new_xctr


# ------------------------------------------------------------------------
def blockwise_mean_3D(A, S):
    """From https://stackoverflow.com/questions/37532184/downsize-3d-matrix-by-averaging-in-numpy-or-alike/73078468

    A is the 3D input array
    S is the blocksize on which averaging is to be performed (list or 1D array)
    """
    m, n, r = numpy.array(A.shape) // S
    return A.reshape(m, S[0], n, S[1], r, S[2]).mean((1, 3, 5))


# ------------------------------------------------------------------------
def imcombine(inimg_list, outimg, method="median", nonzero_thresh=100., scale=None,
              data_hdu=0, kwstring=None, commstring=None, outvarimg=None, sregion=None,
              plot=False, plot_dir='.', save_prefix='imcombine_inputs',
              interactive_plot=False, debug=False,
              ):
    """
    Combine multiple images into a single image using a specified method.

    Parameters
    ----------
    inimg_list : list
        A list of input image paths.
    outimg : str
        The path to the output image.
    method : str, optional
        The method 'median', 'sum', or 'mean' to be used for combining the images. Default is 'median'.
    scale : str, optional
        The scaling method to be used, 'median', 'median_nonzero', 'exptime' or None. Default is None.
    data_hdu : int, optional
        The HDU index for the data extension in the input images. Default is 0.
    kwstring : str, optional
        Header keyword to add to output as "PYW" + kwstring, containing number of inputs.
        Default: None.
    commstring : str, optional
        Header keyword comment to add to output.
        Default: None.
    outvarimg : str, optional
        Filename of variance image to output (if defined).
        Default: None.
    sregion : list, optional
        List defining image x-axis region in which to compute scaling (if a scaling method is defined).
        Example: [1000, 3000].
        Default: None.
    plot : bool, optional
        Whether to output a diagnostic plot.
        Default: False.
    plot_dir : str, optional
        Directory for output of plot (if requested).
        Default: '.'.
    save_prefix : str, optional
        Prefix for plot (if requested).
        Default: 'imcombine_inputs'.
    interactive_plot : bool, optional
        Whether to interrupt processing to provide interactive plot to user.
        Default: False.
    debug : bool, optional
        Whether to report the parameters used in this function call.
        Default: False.
    Returns
    -------
    None
    """
    if debug:
        print(arguments())
    inimg_list.sort()
    # read in data from inimg_list[0] to get image size
    f = pyfits.open(inimg_list[0])
    outfits = pyfits.HDUList(f)
    orig_data = f[data_hdu].data
    bin_x, bin_y = [int(b) for b in f[data_hdu].header["CCDSUM"].split()]

    nimg = len(inimg_list)
    midrow_shift = 80 // bin_y if is_taros(inimg_list[0]) and is_halfframe(inimg_list[0]) else 0
    try:
        scale_factor = numpy.ones(nimg)
        if scale is not None:
            # Do a loop over the images to determine scale factors
            if scale == "midrow_ratio":
                midrow = orig_data[orig_data.shape[0] // 2 - midrow_shift, :]
            for i in range(nimg):
                f = pyfits.open(inimg_list[i])
                new_data = f[data_hdu].data
                if sregion is None:
                    sreg_min = 0
                    sreg_max = new_data.shape[1]
                else:
                    sreg_min, sreg_max = [int(s) for s in sregion]
                if scale == "median":
                    scale_factor[i] = numpy.nanmedian(new_data[:, sreg_min:sreg_max], )
                elif scale == "median_nonzero":
                    nonzero_inds = numpy.nonzero(new_data[:, sreg_min:sreg_max] > nonzero_thresh)
                    scale_factor[i] = numpy.nanmedian(new_data[:, sreg_min:sreg_max][nonzero_inds])
                elif scale == "exptime":
                    scale_factor[i] = f[data_hdu].header["EXPTIME"]
                elif re.match("percentile", scale):
                    perc = float(scale.split("percentile")[1])
                    scale_factor[i] = numpy.nanpercentile(new_data[:, sreg_min:sreg_max], perc)
                elif scale == "midrow_ratio":
                    if i == 0:
                        scale_factor[i] = 1.0
                    else:
                        scale_factor[i] = numpy.nanmedian(new_data[new_data.shape[0] // 2 - midrow_shift, :] / midrow)
                else:
                    raise ValueError("scaling method not yet supported")
                if debug:
                    print(f"Scaling down image {inimg_list[i]} by {scale_factor[i]}")
        ny, nx = numpy.shape(orig_data)
        # chunk in x-axis if more than equivalent of 5 unbinned full-frame images
        chunks = int(numpy.ceil(nimg / (5. * bin_x * 4112. / ny)))
        coadd_data = numpy.zeros_like(orig_data)
        if outvarimg is not None:
            var_arr = numpy.zeros_like(orig_data)
        else:
            var_arr = None
        # gather data for all
        exptime_list = []
        airmass_list = []
        for ch in range(chunks):
            xmin = ch * nx // chunks
            xmax = min((ch + 1) * nx // chunks, nx)
            if debug:
                print(f"Chunk {ch+1}: xrange=[{xmin}:{xmax}]")
            coadd_arr = numpy.zeros([ny, xmax - xmin, nimg], dtype="d")
            for i in range(nimg):
                f = pyfits.open(inimg_list[i])
                new_data = f[data_hdu].data[:, xmin:xmax]
                if ch == 0:
                    exptime_list.append(f[data_hdu].header["EXPTIME"])
                    if f[data_hdu].header["IMAGETYP"].upper() in ["ARC", "BIAS", "FLAT", "SKYFLAT", "WIRE", "ZERO"]:
                        airmass_list.append(1.0)
                    else:
                        try:
                            airmass_list.append(f[data_hdu].header["AIRMASS"])
                        except Exception as air_err:
                            print(
                                f"Failed to get airmass for {f[data_hdu].header['IMAGETYP'].upper()} image {inimg_list[i]}: {air_err}"
                            )
                            airmass_list.append(1.0)
                f.close()
                coadd_arr[:, :, i] = new_data / scale_factor[i]
                gc.collect()
            if plot or interactive_plot:
                offset = 0.05
                fig = plt.figure()
                ax1 = fig.add_subplot(2, 1, 1)
                ax2 = fig.add_subplot(2, 1, 2)
                for i in range(coadd_arr.shape[2]):
                    ax1.plot(coadd_arr[ny // 2 - midrow_shift, :, i] + offset * i)
                    ax2.plot(coadd_arr[ny // 2 - midrow_shift, :, i])
                ax2.set_xlabel("pixel")
                ax1.set_ylabel(f"Counts with {offset}*i offsets")
                ax2.set_ylabel("Counts as scaled")
                ax1.set_title(f"{nimg} inputs ({scale}-scaled) for\n{os.path.basename(outimg)} ({method}-combined)")
                if interactive_plot:
                    plt.show()
                else:
                    ax1.set_xlim(0, coadd_arr.shape[1] // 4)
                    ax1.set_ylim(numpy.nanpercentile(coadd_arr[ny // 2 - midrow_shift, 0:coadd_arr.shape[1] // 4, :], 1),
                                 numpy.nanpercentile(coadd_arr[ny // 2 - midrow_shift, 0:coadd_arr.shape[1] // 4, :] + offset * nimg, 99))
                    ax2.set_xlim(int(0.75 * coadd_arr.shape[1]), coadd_arr.shape[1])
                    ax2.set_ylim(numpy.nanpercentile(coadd_arr[ny // 2 - midrow_shift, int(0.75 * coadd_arr.shape[1]):coadd_arr.shape[1], :], 1),
                                 numpy.nanpercentile(coadd_arr[ny // 2 - midrow_shift, int(0.75 * coadd_arr.shape[1]):coadd_arr.shape[1], :], 99))
                    plt.tight_layout()
                    if chunks > 1:
                        plot_name = f"{save_prefix}_chunk{ch+1}.png"
                    else:
                        plot_name = f"{save_prefix}.png"
                    plot_path = os.path.join(plot_dir, plot_name)
                    plt.savefig(plot_path, dpi=300)
                    plt.close()

            # now combine
            if method == "median":
                coadd_data[:, xmin:xmax] = numpy.nanmedian(coadd_arr, axis=2)
            elif method == "sum":
                coadd_data[:, xmin:xmax] = numpy.nansum(coadd_arr, axis=2)
            elif method == "mean":
                coadd_data[:, xmin:xmax] = numpy.nanmean(coadd_arr, axis=2)
            else:
                raise ValueError("combine method not yet supported")
            # Create the variance image, if requested
            if outvarimg is not None:
                var_arr[:, xmin:xmax] = numpy.nanvar(coadd_arr, axis=2, dtype="float64")

    except Exception as e:
        print(f"An error occurred in imcombine: {str(e)}")
        raise
    outfits[data_hdu].data = coadd_data.astype("float32", casting="same_kind")
    outfits[data_hdu].scale("float32")
    # fix ephemeris data if images are co-added!!!
    if method == "sum" and scale is None:
        f2 = pyfits.open(inimg_list[-1])
        last_hdr = f2[data_hdu].header
        f2.close()
        # HAEND, ZDEND, EXPTIME
        outfits[data_hdu].header.set("EXPTIME", numpy.sum(exptime_list))
        outfits[data_hdu].header.set("LSTEND", last_hdr["LSTEND"])
        outfits[data_hdu].header.set("UTCEND", last_hdr["UTCEND"])
        outfits[data_hdu].header.set("HAEND", last_hdr["HAEND"])
        outfits[data_hdu].header.set("ZDEND", last_hdr["ZDEND"])
        outfits[data_hdu].header.set("AIRMASS", numpy.nanmean(numpy.array(airmass_list)))
    # (5) write to outfile!
    outfits[data_hdu].header.set("PYWIFES", __version__, "PyWiFeS version")
    if kwstring is not None and commstring is not None:
        outfits[data_hdu].header.set(f"PYW{kwstring[:5].upper()}", nimg, f"PyWiFeS: number of {commstring[:30]} combined")
    outfits.writeto(outimg, overwrite=True)
    # write the variance image, if requested
    if outvarimg is not None:
        outfits[data_hdu].data = var_arr.astype("float64", casting="same_kind")
        outfits[data_hdu].scale("float64")
        outfits.writeto(outvarimg, overwrite=True)
    f.close()
    gc.collect()
    return


# ------------------------------------------------------------------------
def imcombine_mef(
    inimg_list,
    outimg,
    scale=None,
    method="median",
    debug=False,
):
    """
    Combine multiple images into a single image using a specified method.

    Parameters
    ----------
    inimg_list : list
        A list of input image paths.
    outimg : str
        The path to the output image.
    scale : str, optional
        The scaling method to be used. Options are 'exptime', 'per_slice_median'.
        Default: None.
    method : str, optional
        The method to be used for combining the images. Options are 'median', 'sum',
        'nansafesum'.
        Default: 'median'.
    debug : bool, optional
        Whether to report the parameters used in this function call.
        Default: False.
    Returns
    -------
    None
    """
    if debug:
        print(arguments())
    nimg = len(inimg_list)
    # read in data from inimg_list[0] to get image size
    f = pyfits.open(inimg_list[0])
    outfits = pyfits.HDUList(f)

    if is_halfframe(inimg_list[0]):
        if is_taros(inimg_list[0]):
            nslits = 12
        else:
            nslits = 13
    else:
        nslits = 25
    data_hdu_list = list(range(1, nslits + 1))
    var_hdu_list = list(range(nslits + 1, 2 * nslits + 1))
    dq_hdu_list = list(range(2 * nslits + 1, 3 * nslits + 1))
    n_ext = len(f)

    for data_hdu, hdu_type in list(zip(data_hdu_list, ('data' for _ in data_hdu_list))) \
            + list(zip(var_hdu_list, ('var' for _ in var_hdu_list))) \
            + list(zip(dq_hdu_list, ('dq' for _ in dq_hdu_list))):
        orig_data = f[data_hdu].data
        ny, nx = numpy.shape(orig_data)
        coadd_arr = numpy.zeros([ny, nx, nimg], dtype="d")
        # gather data for all
        for i in range(nimg):
            f2 = pyfits.open(inimg_list[i])
            new_data = f2[data_hdu].data
            exptime = f2[data_hdu].header["EXPTIME"]
            f2.close()
            if hdu_type == 'dq':
                scale_factor = 1
            else:
                if scale is None:
                    scale_factor = 1.0
                elif scale == "per_slice_median":
                    scale_factor = numpy.nanmedian(new_data)
                elif scale == "exptime":
                    scale_factor = exptime
                else:
                    raise ValueError(f"scaling method '{scale}' not yet supported")
            coadd_arr[:, :, i] = new_data / scale_factor
            gc.collect()
        # now combine
        if hdu_type == 'data':
            if method == "median":
                coadd_data = numpy.nanmedian(coadd_arr, axis=2)
            elif method == "sum":
                # Propagate any NaNs in the inputs into a NaN output
                coadd_data = numpy.sum(coadd_arr, axis=2)
            elif method == "nansafesum":
                # NaN-safe sum using the mean of the finite pixels and the number of inputs.
                # Only sensible to use if sources are very well aligned between exposures and
                # have consistent count levels (by scaling or otherwise)
                coadd_data = numpy.nanmean(coadd_arr, axis=2) * nimg
            else:
                raise ValueError(f"combine method '{method}' not yet supported")
            outfits[data_hdu].data = coadd_data.astype("float32", casting="same_kind")
            outfits[data_hdu].scale("float32")
        elif hdu_type == 'var':
            if method == "median":
                # Not formally correct, but perhaps not too bad
                coadd_data = numpy.nanmedian(coadd_arr, axis=2)
            elif method == "sum":
                # Propagate any NaNs in the inputs into a NaN output
                coadd_data = numpy.sum(coadd_arr, axis=2)
            elif method == "nansafesum":
                # NaN-safe sum using the mean of the finite pixels and the number of inputs.
                # Only sensible to use if sources are very well aligned between exposures and
                # have consistent count levels (by scaling or otherwise)
                coadd_data = numpy.nanmean(coadd_arr, axis=2) * nimg
            else:
                raise ValueError(f"combine method '{method}' not yet supported")
            outfits[data_hdu].data = coadd_data.astype("float64", casting="same_kind")
            outfits[data_hdu].scale("float64")
        elif hdu_type == 'dq':
            coadd_data = numpy.nansum(coadd_arr, axis=2)
            # trim data beyond range
            coadd_data[coadd_data > 32767] = 32767
            coadd_data[coadd_data < -32768] = -32768
            outfits[data_hdu].data = coadd_data.astype("int16", casting="unsafe")
            outfits[data_hdu].scale("int16")
        gc.collect()
    # fix ephemeris data if images are co-added!!!
    if method == "sum" and scale is None:
        mjdobs_list = []
        airmass_list = []
        exptime_list = []
        for i in range(nimg):
            f2 = pyfits.getheader(inimg_list[i], ext=0)
            if "AIRMASS" in f2:
                airmass_list.append(f2["AIRMASS"])
            if "EXPTIME" in f2:
                exptime_list.append(f2["EXPTIME"])
            if "MJD-OBS" in f2:
                mjdobs_list.append(f2["MJD-OBS"])
        last_hdr = pyfits.getheader(inimg_list[numpy.argsort(mjdobs_list)[-1]], ext=0)
        # Update HAEND, ZDEND, EXPTIME in each extension
        for i in range(n_ext):
            outfits[i].header.set("EXPTIME", numpy.sum(exptime_list))
            try:
                outfits[i].header.set("LSTEND", last_hdr["LSTEND"])
            except Exception:
                pass
            try:
                outfits[i].header.set("UTCEND", last_hdr["UTCEND"])
            except Exception:
                pass
            try:
                outfits[i].header.set("HAEND", last_hdr["HAEND"])
            except Exception:
                pass
            try:
                outfits[i].header.set("ZDEND", last_hdr["ZDEND"])
            except Exception:
                pass
            if len(airmass_list) > 0:
                outfits[i].header.set("AIRMASS", numpy.nanmean(numpy.array(airmass_list)))
    # Fix effective exposure time if scaled by EXPTIME
    if scale == "exptime":
        if method == "median":
            new_exptime = 1.0
        else:
            new_exptime = float(nimg)
        for i in range(n_ext):
            outfits[i].header.set("EXPTIME", new_exptime, "Effective exposure time after scaling")

    # (5) write to outfile!
    outfits[0].header.set("PYWIFES", __version__, "PyWiFeS version")
    outfits[0].header.set("PYWCONUM", nimg, "PyWiFeS: number of coadded images")
    outfits[0].header.set("PYWCOSCL", 'None' if scale is None else scale, "PyWiFeS: scaling prior to coadd")
    outfits[0].header.set("PYWCOMTH", method, "PyWiFeS: method for coadd")
    outfits.writeto(outimg, overwrite=True)
    f.close()
    gc.collect()
    return


# ------------------------------------------------------------------------
def imarith_mef(inimg1, operator, inimg2, outimg):
    """
    Performs arithmetic operations between two multi-extension images.

    Parameters
    ----------
    inimg1 : str
        The path to the first input image.
    operator : str
        The operator to be used for combining the images.
        Options: '+', '-', '*', '/'.
    inimg2 : str
        The path to the second input image.
    outimg : str
        The path to the output image.

    Returns
    -------
    None
    """
    if is_halfframe(inimg1):
        if is_taros(inimg1):
            nslits = 12
        else:
            nslits = 13
    else:
        nslits = 25
    data_hdu_list = range(1, nslits + 1)
    var_hdu_list = range(nslits + 1, 2 * nslits + 1)
    dq_hdu_list = range(2 * nslits + 1, 3 * nslits + 1)

    # read in data from the two images, set up the output hdu
    f1 = pyfits.open(inimg1)
    f2 = pyfits.open(inimg2)
    outfits = pyfits.HDUList(f1)

    # PART 1 - data HDUs
    for data_hdu in data_hdu_list:
        data1 = f1[data_hdu].data
        data2 = f2[data_hdu].data
        # do the desired operation
        if operator == "+":
            op_data = data1 + data2
        elif operator == "-":
            op_data = data1 - data2
        elif operator == "*":
            op_data = data1 * data2
        elif operator == "/":
            op_data = data1 / data2
        else:
            raise ValueError
        outfits[data_hdu].data = op_data.astype("float32", casting="same_kind")
        outfits[data_hdu].scale("float32")
        gc.collect()

    # PART 2 - var HDUs
    # NOTE: var_hdu_list must correspond directly with data_hdu_list!!
    for var_hdu, data_hdu in zip(var_hdu_list, data_hdu_list):
        try:
            var1 = f1[var_hdu].data
            var2 = f2[var_hdu].data
            data1 = f1[data_hdu].data
            data2 = f2[data_hdu].data
        except Exception:
            continue
        # do the desired operation
        if (operator == "+") or (operator == "-"):
            op_var = var1 + var2
        elif operator == "*":
            op_var = var1 * (data2**2) + var2 * (data1**2)
        elif operator == "/":
            op_var = var1 / (data2**2) + var2 * ((data1 / (data2**2)) ** 2)
        else:
            raise ValueError
        outfits[var_hdu].data = op_var.astype("float64", casting="same_kind")
        outfits[var_hdu].scale("float64")
        gc.collect()

    # PART 3 - dq HDUs
    for dq_hdu in dq_hdu_list:
        try:
            dq1 = f1[dq_hdu].data
            dq2 = f2[dq_hdu].data
        except Exception:
            continue
        # always add the DQ images!!
        op_dq = dq1 + dq2
        # trim data beyond range
        op_dq[op_dq > 32767] = 32767
        op_dq[op_dq < -32768] = -32768
        outfits[dq_hdu].data = op_dq.astype("int16", casting="unsafe")
        outfits[dq_hdu].scale("int16")
        gc.collect()

    # (5) write to outfile!
    outfits[0].header.set("PYWIFES", __version__, "PyWiFeS version")
    outfits.writeto(outimg, overwrite=True)
    f1.close()
    f2.close()
    gc.collect()
    return


def scaled_imarith_mef(inimg1, operator, inimg2, outimg, scale=None,
                       arg_scaled="second"):
    """
    Combine two images using a specified operator and a scaling factor.

    Parameters
    ----------
    inimg1 : str
        The path to the first input image.
    operator : str
        The operator to be used for combining the images.
        Options are '+', '-', '*', '/'.
    inimg2 : str
        The path to the second input image.
    outimg : str
        The path to the output image.
    scale : float or str
        The scaling factor to be used.
        Options: 'exptime', a float value.
        Default: None.
    arg_scaled : str, optional
        Which argument the scaling should be applied to.
        Options: "first", "second".
        Default: "second"
    Returns
    -------
    None
        This function does not return any value. It writes the combined image to the output file.
    """
    if arg_scaled not in ["first", "second"]:
        raise ValueError(f"Unknown arg_scaled value '{arg_scaled}'. Must be 'first' or 'second'.")
    # check if halfframe
    halfframe = is_halfframe(inimg1)
    if halfframe:
        if is_taros(inimg1):
            nslits = 12
        else:
            nslits = 13
    else:
        nslits = 25
    data_hdu_list = range(1, nslits + 1)
    var_hdu_list = range(nslits + 1, 2 * nslits + 1)
    dq_hdu_list = range(2 * nslits + 1, 3 * nslits + 1)
    # read in data from the two images, set up the output hdu
    f1 = pyfits.open(inimg1)
    f2 = pyfits.open(inimg2)
    outfits = pyfits.HDUList(f1)
    # calculate the scale factor!
    if isinstance(scale, (float, int)):
        scale_factor = scale
    elif scale == "exptime":
        exptime1 = f1[1].header["EXPTIME"]
        exptime2 = f2[1].header["EXPTIME"]
        scale_factor = exptime1 / exptime2

    else:
        scale_factor = 1.0
    # PART 1 - data HDUs
    for data_hdu in data_hdu_list:
        if arg_scaled == "first":
            data1 = f1[data_hdu].data / scale_factor
            data2 = f2[data_hdu].data
        elif arg_scaled == "second":
            data1 = f1[data_hdu].data
            data2 = scale_factor * (f2[data_hdu].data)
        # do the desired operation
        if operator == "+":
            op_data = data1 + data2
        elif operator == "-":
            op_data = data1 - data2
        elif operator == "*":
            op_data = data1 * data2
        elif operator == "/":
            op_data = data1 / data2
        else:
            raise ValueError
        # trim data beyond range
        outfits[data_hdu].data = op_data.astype("float32", casting="same_kind")
        outfits[data_hdu].scale("float32")
    # PART 2 - var HDUs
    # NOTE: var_hdu_list must correspond directly with data_hdu_list!!
    for i in range(len(var_hdu_list)):
        var_hdu = var_hdu_list[i]
        data_hdu = data_hdu_list[i]
        if arg_scaled == "first":
            var1 = f1[var_hdu].data / (scale_factor**2)
            var2 = f2[var_hdu].data
            data1 = f1[data_hdu].data / scale_factor
            data2 = f2[data_hdu].data
        elif arg_scaled == "second":
            var1 = f1[var_hdu].data
            var2 = (scale_factor**2) * (f2[var_hdu].data)
            data1 = f1[data_hdu].data
            data2 = scale_factor * (f2[data_hdu].data)
        # do the desired operation
        if (operator == "+") or (operator == "-"):
            op_var = var1 + var2
        elif operator == "*":
            op_var = var1 * (data2**2) + var2 * (data1**2)
        elif operator == "/":
            op_var = var1 / (data2**2) + var2 * ((data1 / (data2**2)) ** 2)
        else:
            raise ValueError
        outfits[var_hdu].data = op_var.astype("float64", casting="same_kind")
        outfits[var_hdu].scale("float64")
    # PART 3 - dq HDUs
    for dq_hdu in dq_hdu_list:
        dq1 = f1[dq_hdu].data
        dq2 = f2[dq_hdu].data
        # always add the DQ images!!
        op_dq = dq1 + dq2
        # trim data beyond range
        op_dq[op_dq > 32767] = 32767
        op_dq[op_dq < -32768] = -32768
        outfits[dq_hdu].data = op_dq.astype("int16", casting="unsafe")
        outfits[dq_hdu].scale("int16")
    # (5) write to outfile!
    outfits[0].header.set("PYWIFES", __version__, "PyWiFeS version")
    if scale is not None:
        outfits[0].header.set("PYWARSCA", scale, "PyWiFeS: scaling in MEF arithmetic")
        outfits[0].header.set("PYWARARG", arg_scaled, "PyWiFeS: argument scaled in MEF arithmetic")
    outfits.writeto(outimg, overwrite=True)
    f1.close()
    f2.close()
    return


def imarith(inimg1, operator, inimg2, outimg, data_hdu=0):
    """
    Performs arithmetic operations between two images.

    Parameters
    ----------
    inimg1 : str
        The path to the first input image.
    operator : str
        The operator to be used for combining the images.
        Options: '+', '-', '*', '/'.
    inimg2 : str
        The path to the second input image.
    outimg : str
        The path to the output image.
    data_hdu : int, optional
        The HDU index for the data extension in the input images. Default is 0.

    Returns
    -------
    None
    """
    f1 = pyfits.open(inimg1)
    f2 = pyfits.open(inimg2)
    outfits = pyfits.HDUList(f1)
    # determine datatype of output
    bitpix1 = f1[data_hdu].header['BITPIX']
    bitpix2 = f2[data_hdu].header['BITPIX']
    # write output as highest-bitcount floating point if either input is,
    # or hightest-bitcount integer if both integers
    bitpix = min(bitpix1, bitpix2) if bitpix1 < 0 or bitpix2 < 0 else max(bitpix1, bitpix2)
    orig_fits_scale = fits_scale_from_bitpix(bitpix)
    #
    data1 = f1[data_hdu].data
    # check if halfframe discrepancy
    halfframe1 = is_halfframe(inimg1)
    halfframe2 = is_halfframe(inimg2)
    if halfframe1 and not halfframe2:
        ymin, ymax = [int(b) for b in f1[data_hdu].header['DETSEC'].split(",")[1].rstrip(']').split(":")]
        # NB: this is not rebinning, just converting detector pixels to image pixels
        bin_y = int(f2[data_hdu].header["CCDSUM"].split()[1])
        data2 = f2[data_hdu].data[ymin // bin_y:ymax // bin_y, :]
    elif halfframe1 == halfframe2:
        data2 = f2[data_hdu].data
    else:
        raise ValueError(f"Cannot imarith when first image ({inimg1}) is full frame and second image ({inimg2}) is half frame")

    # do the desired operation
    if operator == "+":
        op_data = data1 + data2
    elif operator == "-":
        op_data = data1 - data2
    elif operator == "*":
        op_data = data1 * data2
    elif operator == "/":
        op_data = data1 / data2
    else:
        raise ValueError
    outfits[data_hdu].data = op_data.astype(orig_fits_scale, casting="same_kind")
    outfits[data_hdu].scale(orig_fits_scale)
    outfits[data_hdu].header.set("PYWIFES", __version__, "PyWiFeS version")
    outfits.writeto(outimg, overwrite=True)
    f1.close()
    f2.close()
    return


def imarith_float_mef(inimg1, operator, scale, outimg):
    """
    Performs arithmetic operations between two floating point multi-extension images.

    Parameters
    ----------
    inimg1 : str
        The path to the first input image.
    operator : str
        The operator to be used for combining the images.
        Options: '+', '-', '*', '/'.
    inimg2 : str
        The path to the second input image.
    outimg : str
        The path to the output image.

    Returns
    -------
    None
    """
    # check if halfframe
    halfframe = is_halfframe(inimg1)
    if halfframe:
        if is_taros(inimg1):
            nslits = 12
        else:
            nslits = 13
    else:
        nslits = 25
    data_hdu_list = range(1, nslits + 1)
    var_hdu_list = range(nslits + 1, 2 * nslits + 1)
    dq_hdu_list = range(2 * nslits + 1, 3 * nslits + 1)

    # read in data from the two images, set up the output hdu
    f1 = pyfits.open(inimg1)
    outfits = pyfits.HDUList(f1)
    # PART 1 - data HDUs
    for data_hdu in data_hdu_list:
        # determine datatype of output
        bitpix = f1[data_hdu].header['BITPIX']
        orig_fits_scale = fits_scale_from_bitpix(bitpix)
        #
        data1 = f1[data_hdu].data
        # do the desired operation
        if operator == "+":
            op_data = data1 + scale
        elif operator == "-":
            op_data = data1 - scale
        elif operator == "*":
            op_data = data1 * scale
        elif operator == "/":
            op_data = data1 / scale
        else:
            raise ValueError
        outfits[data_hdu].data = op_data.astype(orig_fits_scale, casting="same_kind")
        outfits[data_hdu.scale(orig_fits_scale)]
    # PART 2 - var HDUs
    # NOTE: var_hdu_list must correspond directly with data_hdu_list!!
    for i in range(len(var_hdu_list)):
        var_hdu = var_hdu_list[i]
        data_hdu = data_hdu_list[i]
        # determine datatype of output
        bitpix = f1[var_hdu].header['BITPIX']
        orig_fits_scale = fits_scale_from_bitpix(bitpix)
        #
        var1 = f1[var_hdu].data
        data1 = f1[data_hdu].data
        # do the desired operation
        if (operator == "+") or (operator == "-"):
            op_var = var1
        elif operator == "*":
            op_var = var1 * (scale**2)
        elif operator == "/":
            op_var = var1 / (scale**2)
        else:
            raise ValueError
        outfits[var_hdu].data = op_var.astype(orig_fits_scale, casting="same_kind")
        outfits[var_hdu].scale(orig_fits_scale)
    # PART 3 - dq HDUs
    for dq_hdu in dq_hdu_list:
        dq1 = f1[dq_hdu].data
        # No modification of DQ
        op_dq = dq1
        # trim data beyond range
        op_dq[op_dq > 32767] = 32767
        op_dq[op_dq < -32768] = -32768
        outfits[dq_hdu].data = op_dq.astype("int16", casting="unsafe")
        outfits[dq_hdu].scale("int16")
    # (5) write to outfile!
    outfits[0].header.set("PYWIFES", __version__, "PyWiFeS version")
    outfits.writeto(outimg, overwrite=True)
    f1.close()
    return


def imarith_float(inimg1, operator, scale, outimg, data_hdu=0):
    """
    Performs arithmetic operations between two floating point images.

    Parameters
    ----------
    inimg1 : str
        The path to the first input image.
    operator : str
        The operator to be used for combining the images.
        Options: '+', '-', '*', '/'.
    inimg2 : str
        The path to the second input image.
    outimg : str
        The path to the output image.
    data_hdu : int, optional
        The HDU index for the data extension in the input images. Default is 0.

    Returns
    -------
    None
    """
    return imarith_float_mef(
        inimg1,
        operator,
        scale,
        outimg,
        data_hdu_list=[data_hdu],
        var_hdu_list=[],
        dq_hdu_list=[],
    )


def imcopy(inimg, outimg):
    """
    Copy one image to a new filename.

    Parameters
    ----------
    inimg : str
        The path to the input image.
    outimg : str
        The path to the output image.

    Returns
    -------
    None
    """
    f = pyfits.open(inimg)
    outfits = pyfits.HDUList(f)
    outfits.writeto(outimg, overwrite=True)
    f.close()
    return


# ------------------------------------------------------------------------
# OVERSCAN SUBTRACTION
"""
Default values for 1x1 binning
default_det_reg_bin1x1 = [[1,4096,62,2109],
                          [1,4096,2115,4162]]
# default_ovs_reg_bin1x1 = [1,4096,48,61]
default_ovs_reg_bin1x1 = [[1,4096,48,61],
                          [1,4096,48,61]]
# Default values for 2x binning in Y
default_det_reg_bin1x2 = [[1,2048,62,2109],
                          [1,2048,2115,4162]]
# default_ovs_reg_bin1x2 = [1,2048,48,61]
default_ovs_reg_bin1x2 = [[1,2048,48,61],
                          [1,2048,48,61]]
"""
# These are [y_min, y_max, x_min, x_max] for each amplifier,
# in FITS (1-indexed) notation.
default_detector_values = {
    "B1": {
        "det_regs": [
            [1, 2048, 22, 2069],
            [1, 2048, 2070, 4117],
            [2049, 4096, 2070, 4117],
            [2049, 4096, 22, 2069],
        ],
        "ovs_regs": [
            [1, 2048, 8, 17],
            [1, 2048, 4122, 4131],
            [2049, 4096, 4122, 4131],
            [2049, 4096, 8, 17],
        ],
        "sci_regs": [
            [1, 2048, 1, 2048],
            [1, 2048, 2049, 4096],
            [2049, 4096, 2049, 4096],
            [2049, 4096, 1, 2048],
        ],
        "gain": [0.82, 0.92, 0.92, 0.96],
        "rdnoise": [7.9, 8.95, 4.4, 5.7],
    },
    "B2": {
        "det_regs": [
            [1, 2048, 19, 2066],
            [1, 2048, 2115, 4162],
            [2049, 4096, 2115, 4162],
            [2049, 4096, 19, 2066],
        ],
        "ovs_regs": [
            [1, 2048, 7, 14],
            [1, 2048, 4166, 4173],
            [2049, 4096, 4166, 4173],
            [2049, 4096, 7, 14],
        ],
        "sci_regs": [
            [1, 2048, 1, 2048],
            [1, 2048, 2049, 4096],
            [2049, 4096, 2049, 4096],
            [2049, 4096, 1, 2048],
        ],
        "gain": [0.82, 0.92, 0.92, 0.96],
        "rdnoise": [7.9, 8.95, 4.4, 5.7],
    },
    "B3": {
        "det_regs": [[1, 4096, 62, 2109], [1, 4096, 2115, 4162]],
        "ovs_regs": [[1, 4096, 14, 40], [1, 4096, 14, 40]],
        "sci_regs": [[1, 4096, 1, 2048], [1, 4096, 2049, 4096]],
        "gain": [1.0, 1.0],
        "rdnoise": [7.0, 7.0],
    },
    "B4": {
        "det_regs": [
            [1, 4112, 54, 4149],
        ],
        "ovs_regs": [
            [1, 4112, 6, 53],
        ],
        "sci_regs": [
            [1, 4112, 1, 4096],
        ],
        "gain": [1.0],
        "rdnoise": [7.0],
    },
    "B5": {
        "det_regs": [
            [1, 4112, 54, 4149],
        ],
        "ovs_regs": [
            [1, 4112, 4, 43],
        ],
        "sci_regs": [
            [1, 4112, 1, 4096],
        ],
        "gain": [1.47],  # e-/ADU
        "rdnoise": [3.5],  # e-
    },
    "R1": {
        "det_regs": [
            [1, 2048, 22, 2069],
            [1, 2048, 2070, 4117],
            [2049, 4096, 2070, 4117],
            [2049, 4096, 22, 2069],
        ],
        "ovs_regs": [
            [1, 2048, 8, 17],
            [1, 2048, 4122, 4131],
            [2049, 4096, 4122, 4131],
            [2049, 4096, 8, 17],
        ],
        "sci_regs": [
            [1, 2048, 1, 2048],
            [1, 2048, 2049, 4096],
            [2049, 4096, 2049, 4096],
            [2049, 4096, 1, 2048],
        ],
        "gain": [1.01, 0.98, 1.02, 0.84],
        "rdnoise": [4.4, 7.0, 5.2, 5.0],
    },
    "R2": {
        "det_regs": [
            [1, 2048, 19, 2066],
            [1, 2048, 2115, 4162],
            [2049, 4096, 2115, 4162],
            [2049, 4096, 19, 2066],
        ],
        "ovs_regs": [
            [1, 2048, 7, 14],
            [1, 2048, 4166, 4173],
            [2049, 4096, 4166, 4173],
            [2049, 4096, 7, 14],
        ],
        "sci_regs": [
            [1, 2048, 1, 2048],
            [1, 2048, 2049, 4096],
            [2049, 4096, 2049, 4096],
            [2049, 4096, 1, 2048],
        ],
        "gain": [1.01, 0.98, 1.02, 0.84],
        "rdnoise": [4.4, 7.0, 5.2, 5.0],
    },
    "R3": {
        "det_regs": [[1, 4096, 62, 2109], [1, 4096, 2115, 4162]],
        "ovs_regs": [[1, 4096, 14, 40], [1, 4096, 14, 40]],
        "sci_regs": [[1, 4096, 1, 2048], [1, 4096, 2049, 4096]],
        "gain": [1.0, 1.0],
        "rdnoise": [7.0, 7.0],
    },
    "R4a": {
        "det_regs": [
            [1, 4096, 64, 4159],
        ],
        "ovs_regs": [
            [1, 4096, 1, 63],
        ],
        "sci_regs": [
            [1, 4096, 1, 4096],
        ],
        "gain": [1.0],
        "rdnoise": [7.0],
    },
    "R4b": {
        "det_regs": [
            [1, 4096, 54, 4149],
        ],
        "ovs_regs": [
            [1, 4096, 6, 53],
        ],
        "sci_regs": [
            [1, 4096, 1, 4096],
        ],
        "gain": [1.0],
        "rdnoise": [7.0],
    },
    "R4": {
        "det_regs": [
            [1, 4112, 54, 4149],
        ],
        "ovs_regs": [
            [1, 4112, 6, 53],
        ],
        "sci_regs": [
            [1, 4112, 1, 4096],
        ],
        "gain": [1.0],
        "rdnoise": [7.0],
    },
    "R5": {
        "det_regs": [
            [1, 4112, 54, 4149],
        ],
        "ovs_regs": [
            [1, 4112, 4, 43],
        ],
        "sci_regs": [
            [1, 4112, 1, 4096],
        ],
        "gain": [1.39],  # e-/ADU
        "rdnoise": [3.5],  # e-
    },
}


def determine_detector_epoch(inimg, data_hdu=0):
    """
    Determine the observing epoch of the image, which defines detector
    properties like overscan regions, gain, and read noise.

    Parameters
    ----------
    inimg : str
        The path to the input image.
    data_hdu : int, optional
        The HDU index for the data extension in the input images. Default is 0.

    Returns
    -------
    epoch : str
        The code for the observing epoch.

    None
    """
    f = pyfits.open(inimg)
    orig_hdr = f[data_hdu].header
    f.close()
    utc_str = orig_hdr["DATE-OBS"].split("T")[0]
    utc_date = int(float(utc_str.replace("-", "")))
    camera = orig_hdr["CAMERA"]
    if camera == "WiFeSRed":
        if utc_date < 20100511:
            epoch = "R1"
        elif utc_date < 20100801:
            epoch = "R2"
        elif utc_date < 20130225:
            epoch = "R3"
        elif utc_date < 20130322:
            epoch = "R4a"
        elif utc_date < 20130503:
            epoch = "R4b"
        elif utc_date < 20230308:
            epoch = "R4"
        else:
            epoch = "R5"
    else:
        if utc_date < 20100511:
            epoch = "B1"
        elif utc_date < 20110616:
            epoch = "B2"
        elif utc_date < 20130522:
            epoch = "B3"
        elif utc_date < 20230308:
            epoch = "B4"
        else:
            epoch = "B5"
    return epoch


def convert_ccd_to_bindata_pix(pix_defs, bin_x, bin_y):
    mod_pix_defs = [
        (pix_defs[0] - 1) // bin_y,
        (pix_defs[1] - 1) // bin_y,
        (pix_defs[2] - 1) // bin_x,
        (pix_defs[3] - 1) // bin_x,
    ]
    return mod_pix_defs


def make_overscan_mask(dflat, omask, data_hdu=0, debug=False):
    """
    From a domeflat, get the slice y-axis locations from a median cut
    through the centre. Then use the mean of the illuminated and
    non-illuminated pixels to define a threshold. Rows above the
    threshold will be masked (value = 0) from the overscan calculation
    due to elevated counts in those rows. Note that the output FITS
    image has a single axis, so gets treated by many tools as having
    only an x-axis extent.

    Parameters
    ----------
    dflat : str
        The path to the input domeflat image.
    omask : str
        The path to the output image mask.
    data_hdu : int, optional
        The HDU index for the data extension in the input images. Default is 0.

    Returns
    -------
    None
    """
    if debug:
        print(arguments())
    fdata = pyfits.getdata(dflat, ext=data_hdu)
    ysize, xsize = fdata.shape
    fslice = numpy.nanmedian(fdata[:, int(0.33 * xsize):int(0.67 * xsize)], axis=1)
    flim = numpy.mean(numpy.nanpercentile(fslice, [33, 67]))
    mask_idx = (fslice < flim)
    outmask = numpy.zeros((ysize,), dtype=int)
    outmask[mask_idx] = 1
    # De-select edge pixels
    outmask[0] = 0
    outmask[-1] = 0
    # Shrink mask by five pixels from each side of each block of good pixels
    outmask = outmask * numpy.roll(outmask, 5) * numpy.roll(outmask, -5)

    hdu = pyfits.PrimaryHDU(data=outmask)
    hdu.scale('int16')
    hdu.writeto(omask, overwrite=True)


def correct_readout_shift(indata, verbose=False):
    """
    In the early period of Automated observations, the red arm readout was
    sometimes affected by a problem of extra pixels early in the data stream,
    which shifted the pixels positions throughout the recorded image.
    Per Ian Price (ANU):
        The amplifier is use is at the upper right of the image and you essentially
        need to chop the last N pixels from the image and insert N pixels at the
        start of the image to 'slide it right' and undo the wrap-around the original
        problem caused.

    Parameters
    ----------
    indata : numpy.ndarray
        Input data array.
    verbose : bool
        Whether to report when data has been shifted.

    Returns
    -------
    Corrected data array (or original, if unmodified).
    """
    redmin = 4199  # gives a good agreement with Taros wavelength solution
    redvalmin = 600
    redvalmax = 800
    orig_shape = indata.shape
    local_data = numpy.copy(indata)
    # Filter out bad pixels below the bias feature we want to isolate
    local_data[local_data < redvalmin] = redvalmax
    # Do not test with top and bottom rows. This avoid finding 0 shift for images
    # already fixed.
    rowmin = numpy.argmin(local_data[4:-4], axis=1)
    if numpy.any(rowmin != redmin):
        local_data = local_data.flatten(order='C')
        shift = redmin - numpy.argmin(local_data[:orig_shape[1]])
        if verbose:
            print(f"Shifting data by {shift} pixels")
        return numpy.roll(indata.flatten(order='C'), shift).reshape(orig_shape, order='C')
    return indata


def subtract_overscan(
    inimg,
    outimg,
    data_hdu=0,
    detector_regions=None,
    overscan_regions=None,
    science_regions=None,
    gain=None,
    rdnoise=None,
    omaskfile=None,
    omask_threshold=500.0,
    match_binning=None,
    interactive_plot=False,
    verbose=False,
    debug=False,
):
    """
    Subtract the overscan.

    Parameters
    ----------
    inimg : str
        The path to the input image.
    outimg : str
        The path to the output image.
    data_hdu : int, optional
        The HDU index for the data extension in the input images. Default is 0.
    detector_regions : list, optional
        List of full detector [ymin, ymax, xmin, xmax] pixel coordinates in FITS (1-indexed) format. Default is None (to determine from epoch).
    overscan_regions : list, optional
        List of overscan [ymin, ymax, xmin, xmax] pixel coordinates in FITS (1-indexed) format. Default is None (to determine from epoch).
    science_regions : list, optional
        List of science [ymin, ymax, xmin, xmax] pixel coordinates in FITS (1-indexed) format. Default is None (to determine from epoch).
    gain : float, optional
        CCD gain in e-/ADU. Default is None (to determine from epoch).
    rdnoise : float, optional
        CCD readnoise in e-. Default is None (to determine from epoch).
    omaskfile : str, optional
        The path to the maskfile defining the slice/interslice regions. Default is None.
    omask_threshold : float, optional
        Threshold above which to mask pixels from overscan fit. Threshold is per-row mean ADU relative to row with lowest mean. Default is 500.
    match_binning : str, optional
        If not None, will expand/contract binning to the specified 'x y' format.
    interactive_plot : bool, optional
        Whether to interrupt processing to provide interactive plot to user. Default is False.
    verbose : bool
        Whether to report when data has been checked for readout shifts. Default is False.
    debug : bool, optional
        Whether to report the parameters used in this function call. Default is False.

    Returns
    -------
    None
    """
    if debug:
        print(arguments())
    # (0) open file, initialize HDUList that will be saved at output
    f = pyfits.open(inimg)
    outfits = pyfits.HDUList(f)
    orig_data = f[data_hdu].data
    orig_hdr = f[data_hdu].header
    f.close()
    orig_y, orig_x = orig_data.shape
    # (1) format detector and overscan regions
    #     - if 'None' grab default values
    bin_x, bin_y = [int(b) for b in orig_hdr["CCDSUM"].split()]
    if interactive_plot:
        imagetype = orig_hdr["IMAGETYP"].upper()
    # default values if all of the values are not specified
    if (
        (detector_regions is None)
        or (overscan_regions is None)
        or (science_regions is None)
        or (gain is None)
        or (rdnoise is None)
    ):
        # determine the epoch
        epoch = determine_detector_epoch(inimg)
        # get detector characteristics
        init_det_reg = default_detector_values[epoch]["det_regs"]
        detector_regions = [
            convert_ccd_to_bindata_pix(x, bin_x, bin_y) for x in init_det_reg
        ]
        init_ovs_reg = default_detector_values[epoch]["ovs_regs"]
        overscan_regions = [
            convert_ccd_to_bindata_pix(x, bin_x, bin_y) for x in init_ovs_reg
        ]
        init_sci_reg = default_detector_values[epoch]["sci_regs"]
        science_regions = [
            convert_ccd_to_bindata_pix(x, bin_x, bin_y) for x in init_sci_reg
        ]
        gain = default_detector_values[epoch]["gain"]
        rdnoise = default_detector_values[epoch]["rdnoise"]

    # FORMAT REGION DEFINITIONS
    fmt_det_reg = [[x[0], (x[1] + 1), x[2], (x[3] + 1)] for x in detector_regions]
    fmt_ovs_reg = [[x[0], (x[1] + 1), x[2], (x[3] + 1)] for x in overscan_regions]
    fmt_sci_reg = [[x[0], (x[1] + 1), x[2], (x[3] + 1)] for x in science_regions]

    if omaskfile is not None:
        omask = pyfits.getdata(omaskfile)

    utc_date = int(orig_hdr["DATE-OBS"].split("T")[0].replace("-", ""))
    if (utc_date >= 20220613 and utc_date < 20230731
            and not is_taros(orig_hdr)
            and orig_hdr["CAMERA"] == "WiFeSRed"):
        # Check for red arm pixel shifts in early Automated era readouts. Correct, if present.
        if verbose:
            print(f"Checking {inimg} for pixel shift")
        orig_data = correct_readout_shift(orig_data, verbose=verbose)

    # (2) create data array - MUST QUERY FOR HALF-FRAME
    x_sci_size = numpy.sum([ff[3] - ff[2] if ff[0] == fmt_sci_reg[0][0] else 0 for ff in fmt_sci_reg])
    y_sci_size = numpy.sum([ff[1] - ff[0] if ff[2] == fmt_sci_reg[0][2] else 0 for ff in fmt_sci_reg])
    halfframe = is_halfframe(inimg, data_hdu=data_hdu)
    if halfframe:
        y_sci_size //= 2
    nx = min(x_sci_size, orig_x)
    ny = min(y_sci_size, orig_y)
    subbed_data = numpy.zeros([ny, nx], dtype=float)
    avg_oscan = []
    for i in range(len(fmt_det_reg)):
        if halfframe:
            init_det = fmt_det_reg[i]
            det = [init_det[0], init_det[0] + ny, init_det[2], init_det[3]]
            init_ovs = fmt_ovs_reg[i]
            ovs = [init_ovs[0], init_ovs[0] + ny, init_ovs[2], init_ovs[3]]
            init_sci = fmt_sci_reg[i]
            sci = [init_sci[0], init_sci[0] + ny, init_sci[2], init_sci[3]]
        else:
            det = fmt_det_reg[i]
            ovs = fmt_ovs_reg[i]
            sci = fmt_sci_reg[i]

        curr_data = orig_data[det[0]:det[1], det[2]:det[3]]
        # (3) determine overscan, subtract from data
        curr_ovs_data = orig_data[ovs[0]:ovs[1], ovs[2]:ovs[3]]

        if omaskfile is not None:
            curr_omask = omask[ovs[0]:ovs[1]]
            ovs_y = curr_ovs_data.shape[0]
            meancounts = numpy.nanmean(orig_data[sci[0]:sci[1], sci[2]:sci[3]], axis=1)
            if interactive_plot:
                plt.scatter(numpy.arange(meancounts.shape[0]), meancounts - numpy.nanmin(meancounts))
                plt.axhline(omask_threshold)
                plt.title(f"Imagetype {imagetype}")
                plt.ylabel('Science region mean counts per row (relative to smallest row)')
                plt.xlabel('Y pixel')
                plt.show()
            high_rows = (meancounts - numpy.nanmin(meancounts) > omask_threshold)
            nrows_high = numpy.count_nonzero(high_rows)
            if nrows_high > 0.5 * ovs_y:
                print(f"WARNING: Likely light leak in {inimg}. Not masking high rows in overscan.")
                omaskfile = None
            elif nrows_high > 0:
                masked_ovs_data = numpy.nanmedian(curr_ovs_data, axis=1)[numpy.nonzero(curr_omask * ~high_rows)]
                masked_ovs_rows = numpy.arange(ovs_y)[numpy.nonzero(curr_omask * ~high_rows)]
                ch_coeff = numpy.polynomial.chebyshev.chebfit(x=masked_ovs_rows, y=masked_ovs_data.T, deg=7)
                curr_ovs_model = numpy.polynomial.chebyshev.chebval(x=numpy.arange(curr_ovs_data.shape[0]), c=ch_coeff).T
                curr_ovs_median = numpy.nanmedian(curr_ovs_data, axis=1)
                curr_ovs_val = numpy.where(curr_omask + ~high_rows, curr_ovs_median, curr_ovs_model)
                if interactive_plot:
                    plt.scatter(numpy.arange(ovs_y), curr_ovs_median, c='k', label='Original data')
                    plt.plot(numpy.arange(ovs_y), curr_ovs_model.T, color='green', label='Chebyshev fit')
                    plt.scatter(numpy.arange(ovs_y), curr_ovs_val, c='b', label='Adopted overscan')
                    plt.title(f"{imagetype} - {os.path.basename(inimg)}")
                    plt.ylabel('Overscan value (ADU)')
                    plt.xlabel('Y pixel')
                    plt.ylim(numpy.nanmin(curr_ovs_val) - 10, numpy.nanmax(curr_ovs_val) + 10)
                    plt.legend()
                    plt.show()
            else:
                omaskfile = None
        if omaskfile is None:
            # Take mean of central 50% of values per row
            olim = [curr_ovs_data.shape[1] // 4, int(numpy.ceil(curr_ovs_data.shape[1] * 0.75)) + 1]
            curr_ovs_val = numpy.nanmean(numpy.sort(curr_ovs_data, axis=1)[:, olim[0]:olim[1]], axis=1)

        subbed_data[sci[0]:sci[1], sci[2]:sci[3]] = gain[i] * (
            curr_data - curr_ovs_val[:, numpy.newaxis]
        )
        avg_oscan.append(numpy.mean(curr_ovs_val))

    # (3a) - if there were any saturated pixels before overscan subtraction, set them to NaN
    subbed_data[curr_data == 65535] = numpy.nan

    # (3b) - if epoch R4a, flip data!
    if epoch == "R4a":
        temp_data = subbed_data[:, ::-1]
        next_data = temp_data[::-1, :]
        subbed_data = next_data

    # (3c) - if requested, ensure binning aligns with science data
    if match_binning is not None and match_binning != orig_hdr["CCDSUM"]:
        print(f"Adjusting binning of {os.path.basename(inimg)} from {orig_hdr['CCDSUM']} to {match_binning}")
        out_bin = [int(b) for b in match_binning.split()]
        if out_bin[0] < bin_x:
            # Stretch the data in x
            subbed_data = numpy.repeat(subbed_data, repeats=(bin_x // out_bin[0]), axis=1) / float(bin_x // out_bin[0])
            bin_x = out_bin[0]
        if out_bin[1] < bin_y:
            # Stretch the data in y
            subbed_data = numpy.repeat(subbed_data, repeats=(bin_y // out_bin[1]), axis=0) / float(bin_y // out_bin[1])
            bin_y = out_bin[1]
        elif out_bin[0] > bin_x or out_bin[1] > bin_y:
            # Block-sum the data.
            # Adapted from https://stackoverflow.com/a/37534242
            S = [out_bin[1] // bin_y, out_bin[0] // bin_x]
            m, n = numpy.array(subbed_data.shape) // S
            subbed_data = subbed_data.reshape(m, S[0], n, S[1]).sum((1, 3))

        ny, nx = subbed_data.shape
        outfits[data_hdu].header.set("CCDSUM", match_binning)

    # (4) only modify data part of data_hdu for output
    detsize_str = "[%d:%d,%d:%d]" % (1, nx, 1, ny)
    outfits[data_hdu].header.set("DETSIZE", detsize_str)
    outfits[data_hdu].header.set("CCDSIZE", detsize_str)
    outfits[data_hdu].header.set("DATASEC", detsize_str)
    outfits[data_hdu].header.set("TRIMSEC", detsize_str)
    outfits[data_hdu].header.set("RDNOISE", max(rdnoise), 'Read noise in electrons')
    outfits[data_hdu].header.set("GAIN", 1.0)
    outfits[data_hdu].header.set("PYWIFES", __version__, "PyWiFeS version")
    if omaskfile is not None:
        outfits[data_hdu].header.set('PYWOVERM', True, "PyWiFeS: used overscan mask")
        outfits[data_hdu].header.set('PYWOVTHR', omask_threshold, "PyWiFeS: overscan mask threshold")
    else:
        outfits[data_hdu].header.set('PYWOVERM', False, "PyWiFeS: used overscan mask")
    outfits[data_hdu].header.set('PYWOSUB', numpy.mean(avg_oscan), "PyWiFeS: mean overscan subtracted")
    outfits[data_hdu].data = subbed_data.astype("float32", casting="same_kind")
    outfits[data_hdu].scale("float32")
    # (5) write to outfile!
    outfits.writeto(outimg, overwrite=True)
    return


# ------------------------------------------------------------------------
def repair_bad_pix(inimg, outimg, arm, data_hdu=0, flat_littrow=False,
                   interp_buffer=3, interactive_plot=False, verbose=False, debug=False):
    """
    Handle bad pixels. Performs immediate linear x-interpolation across bad pixels in
    calibration frames, but sets bad pixels to NaN for STANDARD, OBJECT, and SKY frames.
    The NaN pixels are interpolated across in same way after the VAR and DQ extensions
    are created, so that the affected pixels are flagged appropriately.

    Parameters
    ----------
    inimg : str
        The path to the input image.
    outimg : str
        The path to the output image.
    arm : str
        Either 'blue' or 'red'.
    data_hdu : int, optional
        The HDU index for the data extension in the input images. Default is 0.
    flat_littrow : bool, optional
        Mask the Littrow ghosts in flats and skyflats. Applies linear interpolation, as with bad pixels in calibrations. Default is False.
    interp_buffer : int, optional
        Number of additional pixels to median combine with the first on each x-axis edge of a masking region. Default is 3.
    interactive_plot : bool, optional
        Whether to interrupt processing to provide interactive plot to user. Default is False.
    verbose : bool
        Whether to report method applied (determined by IMAGETYP header keyword). Default is False.
    debug : bool, optional
        Whether to report the parameters used in this function call. Default is False.

    Returns
    -------
    None
    """
    if debug:
        print(arguments())
    # bad pixel definitions only certain for epochs B/R4 and later, otherwise skip
    epoch = determine_detector_epoch(inimg)
    if float(epoch[1]) < 4:
        imcopy(inimg, outimg)
        return
    # get data and header
    f = pyfits.open(inimg)
    outfits = pyfits.HDUList(f)
    orig_data = f[data_hdu].data
    orig_hdr = f[data_hdu].header
    f.close()
    # figure out binning
    bin_x, bin_y = [int(b) for b in orig_hdr["CCDSUM"].split()]
    # image type determines method to use
    if orig_hdr['IMAGETYP'].upper() in ['OBJECT', 'STANDARD', 'SKY']:
        method = 'nan'
    else:
        method = 'interp'

    detsec = orig_hdr["DETSEC"]
    y_veryfirst, y_verylast = [int(pix) - 1 for pix in
                               detsec.split(",")[1].rstrip(']').split(":")]

    # Structure: bad_data = [[yfirst, ylast, xfirst, xlast], [...]].
    # Uses unbinned, full-frame, 0-indexed pixels after overscan trimming (p00.fits).
    # Limits are inclusive of the bad pixels on both ends of range.
    if arm == "blue":
        bad_data = [[746, 4111, 1525, 1531],
                    [2693, 3104, 3944, 3944],
                    # cold pixels
                    [3974, 4053, 900, 900],
                    [2387, 2388, 2197, 2197],
                    [909, 913, 1064, 1066],
                    ]
        # Mask the Littrow ghosts (one per slitlet) in flats, to be interpolated over.
        # Regions are generous to accommodate lamp vs sky and thermal shifts.
        if flat_littrow and orig_hdr['IMAGETYP'].upper() in ['FLAT', 'SKYFLAT']:
            littrow_data = [
                [3965, 4040, 3115, 3140],
                [3805, 3880, 3116, 3141],
                [3647, 3722, 3118, 3143],
                [3492, 3567, 3119, 3144],
                [3336, 3411, 3120, 3145],
                [3178, 3253, 3123, 3148],
                [3018, 3093, 3124, 3149],
                [2858, 2933, 3126, 3151],
                [2697, 2772, 3128, 3153],
                [2537, 2612, 3130, 3155],
                [2377, 2452, 3131, 3156],
                [2217, 2292, 3133, 3158],
                [2056, 2131, 3135, 3160],
                [1897, 1972, 3138, 3163],
                [1736, 1811, 3140, 3165],
                [1575, 1650, 3143, 3168],
                [1415, 1490, 3145, 3170],
                [1255, 1330, 3146, 3171],
                [1095, 1170, 3148, 3173],
                [934, 1009, 3150, 3175],
                [774, 849, 3153, 3178],
                [614, 689, 3157, 3182],
                [454, 529, 3159, 3184],
                [294, 369, 3162, 3187],
                [134, 209, 3164, 3189],
            ]
            # Choice of grating changes the ghost locations, but beam splitter
            # only shifts the position by a few pixels.
            if orig_hdr['GRATINGB'] == 'B7000':
                for ll in littrow_data:
                    ll[0] -= 50
                    ll[1] -= 40
                    ll[2] -= 1110
                    ll[3] -= 1090
            elif orig_hdr['GRATINGB'] == 'U7000':
                for ll in littrow_data:
                    ll[0] -= 35
                    ll[1] -= 25
                    ll[2] -= 1080
                    ll[3] -= 1060
            bad_data.extend(littrow_data)
    elif arm == "red":
        bad_data = [[0, 2707, 9, 11],
                    [0, 3280, 773, 775],
                    [0, 4111, 901, 904],
                    [3978, 3986, 897, 906],
                    [0, 3387, 939, 939],
                    [0, 1787, 2273, 2273],
                    # cold pixels
                    [3759, 3762, 257, 260],
                    [3373, 3376, 2382, 2385],
                    [3323, 3323, 2511, 2511],
                    [3319, 3319, 728, 728],
                    [3114, 3120, 1402, 1407],
                    [2944, 2949, 3702, 3706],
                    [2966, 2968, 3747, 3749],
                    [2684, 2685, 756, 757],
                    [2361, 2361, 1489, 1489],
                    [2249, 2251, 898, 899],
                    [2013, 2016, 1149, 1153],
                    [2017, 2017, 1151, 1153],
                    [2044, 2046, 1261, 1263],
                    [2037, 2039, 1825, 1826],
                    [2040, 2040, 1826, 1826],
                    [2036, 2036, 1826, 1826],
                    [1558, 1558, 2430, 2430],
                    [705, 708, 2184, 2189],
                    [632, 635, 905, 905],
                    [634, 636, 906, 906],
                    ]
        if is_taros(inimg):
            # TAROS uses a red amplifier on the bottom edge of the CCD rather than the
            # top, so bad columns extend in the opposite direction.
            for bb, (yfirst, ylast, xfirst, xlast) in enumerate(bad_data):
                if yfirst == 0:
                    bad_data[bb][0] = 0 if ylast == 4111 else ylast
                    bad_data[bb][1] = 4111

        # Mask the Littrow ghosts (one per slitlet) in flats, to be interpolated over.
        # Regions are generous to accommodate lamp vs sky and thermal shifts.
        if flat_littrow and orig_hdr['IMAGETYP'].upper() in ['FLAT', 'SKYFLAT']:
            littrow_data = [
                [3963, 4038, 1502, 1532],
                [3785, 3860, 1505, 1535],
                [3637, 3712, 1506, 1535],
                [3484, 3559, 1510, 1540],
                [3329, 3404, 1505, 1535],
                [3169, 3244, 1504, 1536],
                [3009, 3084, 1507, 1537],
                [2852, 2927, 1504, 1536],
                [2691, 2766, 1504, 1536],
                [2533, 2608, 1505, 1537],
                [2374, 2449, 1505, 1535],
                [2214, 2289, 1505, 1535],
                [2055, 2130, 1500, 1530],
                [1896, 1971, 1497, 1527],
                [1737, 1812, 1498, 1528],
                [1577, 1652, 1495, 1525],
                [1418, 1493, 1495, 1525],
                [1256, 1331, 1493, 1523],
                [1099, 1174, 1487, 1517],
                [938, 1013, 1486, 1516],
                [778, 853, 1480, 1510],
                [615, 690, 1478, 1508],
                [458, 533, 1473, 1503],
                [289, 364, 1470, 1500],
                [108, 183, 1465, 1495],
            ]
            # Choice of grating changes the ghost locations, but beam splitter
            # only shifts the position by a few pixels.
            if orig_hdr['GRATINGR'] == 'R7000':
                # Falls completely on science slits.
                for ll in littrow_data:
                    ll[0] -= 30
                    ll[1] -= 10
                    ll[2] += 480
                    ll[3] += 500
            elif orig_hdr['GRATINGR'] == 'I7000':
                # Falls completely on science slits in region of high fringing.
                # Better to omit masking.
                littrow_data = []
            bad_data.extend(littrow_data)

    else:
        # Unknown arm
        raise ValueError(f"Arm must be 'blue' or 'red'. Received '{arm}'.")

    interp_data = 1.0 * orig_data
    for yfirst, ylast, xfirst, xlast in bad_data:
        yfirst = max(min(yfirst - y_veryfirst, y_verylast - y_veryfirst), 0)
        ylast = min(max(ylast - y_veryfirst, 0), y_verylast - y_veryfirst)
        if yfirst == (y_verylast - y_veryfirst) or ylast == 0:
            continue
        yfirst //= bin_y
        ylast //= bin_y
        xfirst //= bin_x
        xlast //= bin_x
        if method == 'interp':
            slice_lo = numpy.nanmedian(orig_data[yfirst:ylast + 1, xfirst - 1 - interp_buffer:xfirst], axis=1)
            slice_hi = numpy.nanmedian(orig_data[yfirst:ylast + 1, xlast + 1:xlast + 2 + interp_buffer], axis=1)
            if verbose:
                print(f"Interpolating from ({yfirst}:{ylast + 1}, {xfirst - 1}) to ({yfirst}:{ylast + 1}, {xlast + 1})")
            for this_x in numpy.arange(xfirst, xlast + 1):
                interp_data[yfirst:ylast + 1, this_x] = (slice_hi - slice_lo) / ((xlast + 1.) - (xfirst - 1.)) * (this_x - (xfirst - 1.)) + slice_lo
        elif method == 'nan':
            if verbose:
                print(f"NaN-ing ({yfirst}:{ylast + 1},{xfirst}:{xlast + 1})")
            interp_data[yfirst:ylast + 1, xfirst:xlast + 1] = numpy.nan
    # Interpolate over any NaN pixels arising from saturation (excluding science images)
    if method == 'interp':
        nan_rows = numpy.nonzero(numpy.isnan(interp_data))[0]
        if len(nan_rows) > 0:
            # loop over each row with at least one bad pixel and linearly interpolate across the gaps
            for row in sorted(set(nan_rows)):
                interp_data[row, :][numpy.isnan(interp_data[row, :])] = \
                    numpy.interp(numpy.where(numpy.isnan(interp_data[row, :]))[0],
                                 numpy.where(numpy.isfinite(interp_data[row, :]))[0],
                                 interp_data[row, :][numpy.isfinite(interp_data[row, :])])
    if interactive_plot:
        fig, axs = plt.subplots(2, 1, figsize=(8, 6))
        axs[0].imshow(orig_data, norm=colors.LogNorm(), cmap=cm['gist_ncar'])
        axs[0].set_title(f"Original - {orig_hdr['IMAGETYP'].upper()}")
        axs[1].imshow(interp_data, norm=colors.LogNorm(), cmap=cm['gist_ncar'])
        axs[1].set_title('Repaired')
        plt.tight_layout()
        plt.show()
    # save it!
    outfits[data_hdu].data = interp_data
    outfits.writeto(outimg, overwrite=True)
    return


# ------------------------------------------------------------------------
def fit_wifes_interslit_bias(
    inimg,
    data_hdu=0,
    slitlet_def_file=None,
    method="row_med",
    x_polydeg=1,
    y_polydeg=1,
    interactive_plot=False,
):
    """
    Inter-slitlet bias determination.

    Parameters
    ----------
    inimg : str
        The path to the input image.
    data_hdu : int, optional
        The HDU index for the data extension in the input images. Default is 0.
    slitlet_def_file : str, optional
        The path to the pickle file defining the slitlet boundaries. Default is None.
    method : str, optional
        Method to fit interslit bias level. Options are "row_med" (per-column mean of
        points within 20 counts of the column's median), "surface" (fit 2D surface to
        bias level), "median" (median of the interslice regions from the entire frame).
        Default is "row_med".
    x_polydeg : int, optional
        For method="surface", polynomial degree of the fit in the x-axis. Default is 1.
    y_polydeg : int, optional
        For method="surface", polynomial degree of the fit in the y-axis. Default is 1.
    interactive_plot : bool, optional
        Whether to interrupt processing to provide interactive plot to user. Default is False.

    Returns
    -------
    Numpy ndarray of the bias level.
    """
    # ---------------------------
    # (0) open file, initialize HDUList that will be saved at output
    f = pyfits.open(inimg)
    orig_data = f[data_hdu].data
    orig_hdr = f[data_hdu].header
    f.close()
    # check if halfframe
    halfframe = is_halfframe(inimg)
    # grab necessary info from header
    bin_x, bin_y = [int(b) for b in orig_hdr["CCDSUM"].split()]
    utc_str = orig_hdr["DATE-OBS"].split("T")[0]
    utc_date = int(float(utc_str.replace("-", "")))
    camera = orig_hdr["CAMERA"]
    # ---------------------------
    # (1) make a mask of inter-slitlet zones
    interstice_map = numpy.ones(numpy.shape(orig_data))
    interstice_mask = numpy.ones(numpy.shape(orig_data)[0])
    if slitlet_def_file is not None:
        f2 = open(slitlet_def_file, "rb")
        slitlet_defs = pickle.load(f2)
        f2.close()
    elif camera == "WiFeSRed":
        slitlet_defs = red_slitlet_defs
    else:
        slitlet_defs = blue_slitlet_defs
    if halfframe:
        if is_taros(inimg):
            nslits = 12
            first = 1
            offset = 2056 // bin_y
        else:
            nslits = 13
            first = 7
            offset = 1028 // bin_y
    else:
        nslits = 25
        first = 1
        offset = 0
    for i in range(nslits):
        init_curr_defs = slitlet_defs[str(i + 1)]
        # convert to binned values
        curr_defs = [
            ((init_curr_defs[0] - 1) // bin_x) + 1,
            ((init_curr_defs[1] - 1) // bin_x) + 1,
            ((init_curr_defs[2] - 1) // bin_y) + 1,
            ((init_curr_defs[3] - 1) // bin_y) + 1,
        ]
        # horrible kluge to make sure everything has the same dimensions!
        if (curr_defs[1] - curr_defs[0] + 1) != (4096 // bin_x):
            curr_defs[1] -= 1
        if (curr_defs[3] - curr_defs[2] + 1) != (86 // bin_y):
            curr_defs[3] -= 1
        # define a buffer zone
        ybuff = 6 // bin_y
        mod_defs = [
            curr_defs[0] - 1,
            curr_defs[1],
            curr_defs[2] - 1 - ybuff - offset,
            curr_defs[3] + ybuff - offset,
        ]
        # everything in slitlet zone gets set to zero
        interstice_map[mod_defs[2]:mod_defs[3], mod_defs[0]:mod_defs[1]] = 0
        interstice_mask[mod_defs[2]:mod_defs[3]] = 0
    # ---------------------------
    # (2) from its epoch, determine if is 1-amp or 4-amp readout
    if camera == "WiFeSRed":
        if utc_date < 20110616:
            # namps=4
            sci_regs = [
                [0, 2047 // bin_y, 0, 2047 // bin_x],
                [0, 2047 // bin_y, 2048 // bin_x, 4095 // bin_x],
                [2048 // bin_y, 4095 // bin_y, 0, 2047 // bin_x],
                [2048 // bin_y, 4095 // bin_y, 2048 // bin_x, 4095 // bin_x],
            ]
        else:
            # namps=1
            sci_regs = [[0, 4095 // bin_y, 0, 4095 // bin_x]]
    else:
        if utc_date < 20100801:
            # namps=4
            sci_regs = [
                [0, 2047 // bin_y, 0, 2047 // bin_x],
                [0, 2047 // bin_y, 2048 // bin_x, 4095 // bin_x],
                [2048 // bin_y, 4095 // bin_y, 0, 2047 // bin_x],
                [2048 // bin_y, 4095 // bin_y, 2048 // bin_x, 4095 // bin_x],
            ]
        else:
            # namps=1
            sci_regs = [[0, 4095 // bin_y, 0, 4095 // bin_x]]
    # ---------------------------
    # (3) for each sci region, get interstitial bias level
    out_data = numpy.zeros(numpy.shape(orig_data), dtype="d")
    for i in range(len(sci_regs)):
        init_reg = sci_regs[i]
        if halfframe:
            ny = 2056 // bin_y
            reg = [init_reg[0], init_reg[1] - ny, init_reg[2], init_reg[3]]
        else:
            reg = init_reg
        curr_data = orig_data[reg[0]:reg[1] + 1, reg[2]:reg[3] + 1]
        curr_map = interstice_map[reg[0]:reg[1] + 1, reg[2]:reg[3] + 1]
        curr_mask = interstice_mask[reg[0]:reg[1] + 1]
        ny, nx = numpy.shape(curr_data)
        linx = numpy.arange(nx, dtype="d")
        liny = numpy.arange(ny, dtype="d")
        full_x, full_y = numpy.meshgrid(linx, liny)
        if method == "row_med":
            row_med = numpy.zeros(nx, dtype="d")
            # iterate over columns to excise CRs... bleh
            for i in range(nx):
                curr_col = curr_data[:, i]
                curr_med = numpy.nanmedian(curr_col[numpy.nonzero(curr_mask)[0]])
                good_inds = numpy.nonzero(
                    (numpy.abs(curr_col - curr_med) < 20.0) * (curr_mask)
                )[0]
                if interactive_plot:
                    plt.hist(curr_col[good_inds], bins=50)
                    plt.xlabel("Good curr_col values")
                    plt.ylabel("Number")
                    plt.title(f"Slitlet {i + first}")
                    plt.show()
                if good_inds.size > 0:
                    row_med[i] = numpy.nanmean(curr_col[good_inds])
            # row_med = numpy.nanmedian(curr_data, axis=0)
            bias_sub = row_med ** numpy.ones(numpy.shape(curr_data), dtype="d")
            out_data[reg[0]:reg[1] + 1, reg[2]:reg[3] + 1] = bias_sub
        elif method == "surface":
            curr_inds = numpy.nonzero(curr_map)
            fit_x = full_x[curr_inds].flatten()
            fit_y = full_y[curr_inds].flatten()
            fit_b = curr_data[curr_inds].flatten()
            init_x_poly, init_y_poly = fit_wsol_poly(
                fit_x, fit_y, fit_b, x_polydeg, y_polydeg
            )
            init_bias_fvals = evaluate_wsol_poly(fit_x, fit_y, init_x_poly, init_y_poly)
            resids = fit_b - init_bias_fvals
            resids_rms = numpy.nanmedian(resids**2) ** 0.5
            good_inds = numpy.nonzero(numpy.abs(resids) < 3.0 * (resids_rms))
            x_poly, y_poly = fit_wsol_poly(
                fit_x[good_inds],
                fit_y[good_inds],
                fit_b[good_inds],
                x_polydeg,
                y_polydeg,
            )
            bias_fvals = evaluate_wsol_poly(full_x, full_y, x_poly, y_poly)
            out_data[reg[0]:reg[1] + 1, reg[2]:reg[3] + 1] = bias_fvals
            # PLOT IT!!!
            if interactive_plot:
                randplot_data = numpy.random.permutation(len(fit_x))[:1000]
                randplot_inds = numpy.random.permutation(len(full_x.flatten()))[:1000]
                randplot_lin_mask = numpy.zeros(len(full_x.flatten()))
                randplot_lin_mask[randplot_inds] = 1
                # randplot_mask = numpy.reshape(randplot_lin_mask, numpy.shape(full_x))
                # randplot_fit = numpy.nonzero(randplot_mask)
                fig = plt.figure()
                sp = fig.gca(projection="3d")
                sp.plot(
                    fit_x[randplot_data],
                    fit_y[randplot_data],
                    fit_b[randplot_data],
                    "b.",
                )
                sp.plot_surface(
                    full_x,
                    full_y,
                    bias_fvals,
                    alpha=0.3,
                    rstride=200,
                    cstride=200,
                    color="r",
                )
                fval_med = numpy.nanmedian(bias_fvals)
                sp.set_zlim([fval_med - 5.0 * resids_rms, fval_med + 5.0 * resids_rms])
                plt.show()
        elif method == "median":
            bias_val = numpy.nanmedian(curr_data[numpy.nonzero(curr_map)])
            out_data[reg[0]:reg[1] + 1, reg[2]:reg[3] + 1] = bias_val
        else:
            print("only row_med, surface, and median methods currently supported")
    # ---------------------------
    # (4) return it!
    return out_data


def save_wifes_interslit_bias(
    inimg,
    outimg,
    data_hdu=0,
    slitlet_def_file=None,
    method="surface",
    x_polydeg=1,
    y_polydeg=1,
    interactive_plot=False,
):
    """
    Save the inter-slitlet bias level.

    Parameters
    ----------
    inimg : str
        The path to the input image.
    outimg : str
        The path to the output bias level image.
    data_hdu : int, optional
        The HDU index for the data extension in the input images. Default is 0.
    slitlet_def_file : str, optional
        The path to the pickle file defining the slitlet boundaries. Default is None.
    method : str, optional
        Method to fit interslit bias level. Options are "row_med" (per-column mean of
        points within 20 counts of the column's median), "surface" (fit 2D surface to
        bias level), "median" (median of the interslice regions from the entire frame).
        Default is "surface".
    x_polydeg : int, optional
        For method="surface", polynomial degree of the fit in the x-axis. Default is 1.
    y_polydeg : int, optional
        For method="surface", polynomial degree of the fit in the y-axis. Default is 1.
    interactive_plot : bool, optional
        Whether to interrupt processing to provide interactive plot to user. Default is False.

    Returns
    -------
    None.
    """
    # (0) open file, initialize HDUList that will be saved at output
    f = pyfits.open(inimg)
    outfits = pyfits.HDUList(f)
    # (1) fit bias with requested settings
    out_data = fit_wifes_interslit_bias(
        inimg,
        data_hdu=data_hdu,
        slitlet_def_file=slitlet_def_file,
        method=method,
        x_polydeg=x_polydeg,
        y_polydeg=y_polydeg,
        interactive_plot=interactive_plot,
    )
    # (2) save it!
    outfits[data_hdu].data = out_data
    outfits[data_hdu].header.set("PYWIFES", __version__, "PyWiFeS version")
    outfits.writeto(outimg, overwrite=True)
    f.close()
    return


def subtract_wifes_interslit_bias(
    inimg,
    outimg,
    data_hdu=0,
    slitlet_def_file=None,
    method="surface",
    x_polydeg=1,
    y_polydeg=1,
    interactive_plot=False,
):
    """
    Subtract the inter-slitlet bias level.

    Parameters
    ----------
    inimg : str
        The path to the input image.
    outimg : str
        The path to the output bias level image.
    data_hdu : int, optional
        The HDU index for the data extension in the input images. Default is 0.
    slitlet_def_file : str, optional
        The path to the pickle file defining the slitlet boundaries. Default is None.
    method : str, optional
        Method to fit interslit bias level. Options are "row_med" (per-column mean of
        points within 20 counts of the column's median), "surface" (fit 2D surface to
        bias level), "median" (median of the interslice regions from the entire frame).
        Default is "surface".
    x_polydeg : int, optional
        For method="surface", polynomial degree of the fit in the x-axis. Default is 1.
    y_polydeg : int, optional
        For method="surface", polynomial degree of the fit in the y-axis. Default is 1.
    interactive_plot : bool, optional
        Whether to interrupt processing to provide interactive plot to user. Default is False.

    Returns
    -------
    None.
    """
    # (0) open file, initialize HDUList that will be saved at output
    f = pyfits.open(inimg)
    outfits = pyfits.HDUList(f)
    orig_data = f[data_hdu].data
    # (1) fit bias with requested settings
    out_data = fit_wifes_interslit_bias(
        inimg,
        data_hdu=data_hdu,
        slitlet_def_file=slitlet_def_file,
        method=method,
        x_polydeg=x_polydeg,
        y_polydeg=y_polydeg,
        interactive_plot=interactive_plot,
    )
    # (2) save it!
    outfits[data_hdu].data = orig_data - out_data
    outfits[data_hdu].header.set("PYWIFES", __version__, "PyWiFeS version")
    outfits.writeto(outimg, overwrite=True)
    return


def wifes_bias_model(p, x, camera):
    """
    Generate bias model from parametrised fit. Model includes constant, linear term, and
    two exponentials (with an additional Gaussian term for the blue arm).

    Parameters
    ----------
    p : list or 1D array
        Parameter values of fit.
    x : Numpy array
        Pixel number.
    camera : str
        Label of camera/arm ("WiFeSRed" or not).

    Returns
    -------
    Numpy ndarray of the bias model.
    """
    if camera == "WiFeSRed":
        model = p[0]
        model += p[1] * numpy.exp(p[2] / numpy.abs(x - p[3]))
        model += p[4] * numpy.exp(p[5] / numpy.abs(x - p[6]))
        model += p[7] * x
    else:
        model = p[0]
        model += p[1] * numpy.exp(p[2] / numpy.abs(x - p[3]))
        model += p[4] * numpy.exp(p[5] / numpy.abs(x - p[6]))
        model += p[7] * x
        model += p[8] * numpy.exp(-((p[9] * (x - p[10])) ** 2))
    return model


def error_wifes_bias_model(p, x, z, err, camera):
    """
    Generate bias model from parametrised fit. Model includes constant, linear term, and
    two exponentials (with an additional Gaussian term for the blue arm).

    Parameters
    ----------
    p : list or 1D array
        Parameter values of fit.
    x : Numpy array
        Pixel number.
    z : Numpy array
        Pixel value.
    err : Numpy array
        Array of pixel value errors.
    camera : str
        Label of camera/arm ("WiFeSRed" or not).

    Returns
    -------
    List of status value and residual array.
    """
    status = 0
    residual = (wifes_bias_model(p, x, camera) - z) / err
    return [status, residual]


def generate_wifes_bias_fit(
    bias_img,
    outimg,
    method="row_med",
    data_hdu=0,
    plot=True,
    plot_dir=".",
    save_prefix='bias',
    verbose=False,
):
    """
    Fit the bias level.

    Parameters
    ----------
    bias_img : str
        The path to the input bias image.
    outimg : str
        The path to the output bias level image.
    method : str, optional
        Method to fit bias level. Options are "row_med" (per-column mean of points
        within 20 counts of the column's median), "fit" (collapse along y-axis and
        fit a parametrised shape). Default is "row_med".
    data_hdu : int, optional
        The HDU index for the data extension in the input images. Default is 0.
    plot : bool, optional
        Whether to output a diagnostic plot. Default is False.
    plot_dir : str, optional
        Directory for output of plot (if requested). Default is '.'.
    save_prefix : str, optional
        Prefix for plot (if requested). Default is 'bias'.
    verbose : bool, optional
        If method="fit", whether to report progress of MPFit. Default is False.

    Returns
    -------
    None.
    """
    # get object and bias data
    f1 = pyfits.open(bias_img)
    orig_data = f1[data_hdu].data
    orig_hdr = f1[data_hdu].header
    # from its epoch, determine if is 1-amp or 4-amp readout
    bin_x, bin_y = [int(b) for b in orig_hdr["CCDSUM"].split()]
    utc_str = orig_hdr["DATE-OBS"].split("T")[0]
    utc_date = int(float(utc_str.replace("-", "")))
    camera = orig_hdr["CAMERA"]
    if camera == "WiFeSRed":
        if utc_date < 20110616:
            # namps=4
            sci_regs = [
                [0, 2047 // bin_y, 0, 2047 // bin_x],
                [0, 2047 // bin_y, 2048 // bin_x, 4095 // bin_x],
                [2048 // bin_y, 4095 // bin_y, 0, 2047 // bin_x],
                [2048 // bin_y, 4095 // bin_y, 2048 // bin_x, 4095 // bin_x],
            ]
            # Currently not supported !
            if method == "fit":
                raise ValueError("Bias frame not compatible with surface fitting (4 amps) !")
        else:
            # namps=1
            sci_regs = [[0, 4095 // bin_y, 0, 4095 // bin_x]]
    else:
        if utc_date < 20100801:
            # namps=4
            sci_regs = [
                [0, 2047 // bin_y, 0, 2047 // bin_x],
                [0, 2047 // bin_y, 2048 // bin_x, 4095 // bin_x],
                [2048 // bin_y, 4095 // bin_y, 0, 2047 // bin_x],
                [2048 // bin_y, 4095 // bin_y, 2048 // bin_x, 4095 // bin_x],
            ]
            # Currently not supported !
            if method == "fit":
                raise ValueError("Bias frame not compatible with surface fitting (4 amps) !")

        else:
            # namps=1
            sci_regs = [[0, 4095 // bin_y, 0, 4095 // bin_x]]

    # method == fit :
    # for each region, fit the bias structure
    # It is time consuming to fit the whole 2D structure of the bias.
    # Instead, I collapse it along the y-direction and fit it.
    # It's (much) faster, and accurate enough.
    # method == row_med :
    # This is faster still, and as accurate. No fitting required, we just collapsed
    # the bias, and take the mean after rejecting outliers. Should be the default,
    # unless you know what you are doing ...

    out_data = numpy.zeros(numpy.shape(orig_data), dtype="float32")
    for i in range(len(sci_regs)):
        reg = sci_regs[i]
        curr_data = orig_data[reg[0]:reg[1] + 1, reg[2]:reg[3] + 1]

        ny, nx = numpy.shape(curr_data)
        linx = numpy.arange(nx, dtype="float32")
        liny = numpy.arange(ny, dtype="float32")
        full_x = numpy.meshgrid(linx, liny)[0]

        if method == "row_med":
            row_med = numpy.zeros(nx, dtype="float32")
            # iterate over columns to excise CRs... bleh
            for i in range(nx):
                curr_col = curr_data[:, i]
                curr_med = numpy.nanmedian(curr_col)
                good_inds = numpy.nonzero(numpy.abs(curr_col - curr_med) < 20.0)[0]
                if good_inds.size > 0:
                    row_med[i] = numpy.nanmean(curr_col[good_inds])
            bias_sub = row_med ** numpy.ones(numpy.shape(curr_data), dtype="float32")
            # 's update (bias fit) ------
            # To remove the variations (some at least) along the
            # y-direction, let's smooth the residuals !
            residual = curr_data - bias_sub
            residual[numpy.where(residual > 150)] = 0.0  # remove 'CR'
            residual_blur = ndimage.gaussian_filter(residual, sigma=[50, 0])
            bias_sub += residual_blur
            # -----
            out_data[reg[0]:reg[1] + 1, reg[2]:reg[3] + 1] = bias_sub

        if method == "fit":
            # Fit the bias with an appropriate model
            # Initial conditions differ depending on the camera

            if camera == "WiFeSRed":
                p0 = [0.0, 1.0, -200.0, 4100.0, 1.0, 20.0, -800.0, 0.0001]
                fa = {
                    "x": linx,
                    "z": numpy.nanmean(curr_data, axis=0),
                    "err": numpy.nanstd(curr_data, axis=0),
                    "camera": camera,
                }
                constraints = [
                    {"limited": [0, 0]},
                    {"limited": [0, 0]},
                    {"limited": [0, 0]},
                    {"limited": [1, 0], "limits": [numpy.nanmax(full_x[0, :]), 0]},
                    {"limited": [0, 0]},
                    {"limited": [0, 0]},
                    {"limited": [0, 1], "limits": [0, numpy.nanmin(full_x[0, :])]},
                    {"limited": [0, 0]},
                ]

            else:
                p0 = [
                    -3.0,
                    5.0,
                    -500.0,
                    4100.0,
                    0.0001,
                    200.0,
                    -800.0,
                    0.0001,
                    1,
                    0.01,
                    4000.0,
                ]
                fa = {
                    "x": linx,
                    "z": numpy.nanmean(curr_data, axis=0),
                    "err": numpy.nanstd(curr_data, axis=0),
                    "camera": camera,
                }
                constraints = [
                    {"limited": [0, 0]},
                    {"limited": [0, 0]},
                    {"limited": [0, 0]},
                    {"limited": [1, 0], "limits": [numpy.nanmax(full_x[0, :]), 0]},
                    {"limited": [0, 0]},
                    {"limited": [0, 0]},
                    {"limited": [0, 1], "limits": [0, numpy.nanmin(full_x[0, :])]},
                    {"limited": [0, 0]},
                    {"limited": [1, 0], "limits": [0, 0]},
                    {"limited": [0, 0]},
                    {"limited": [0, 0]},
                ]

            print(" Fitting bias frame %s" % bias_img.split("/")[-1])
            fit_result = mpfit(
                error_wifes_bias_model,
                p0,
                functkw=fa,
                parinfo=constraints,
                quiet=not verbose,
            )
            p1 = fit_result.params
            if fit_result.status <= 0 or fit_result.status == 5:
                print(" Fit may have failed : mpfit status:", fit_result.status)
                print("I'll plot this one for sanity check...")
                plot = True

            out_data[reg[0]:reg[1] + 1, reg[2]:reg[3] + 1] = wifes_bias_model(
                p1, full_x, camera
            )
        # Plot for test purposes ...
        if plot:
            plt.figure(1)
            row_bias = numpy.nanmean(curr_data, axis=0)
            plt.plot(linx, row_bias, "k-", label="raw bias", lw=2)

            # Avoid ploting outlier peaks
            lower_limit = numpy.nanpercentile(row_bias, 0.2)
            upper_limit = numpy.nanpercentile(row_bias, 99.8)

            if method == "row_med":
                row_med_bias = numpy.nanmean(out_data, axis=0)
                plt.plot(linx, row_med_bias, "r-", label="row_med bias")
                resid = numpy.nanmean(curr_data, axis=0) - numpy.nanmean(out_data, axis=0)
                plt.plot(
                    linx,
                    resid,
                    "g-",
                    label="residual",
                )
                lower_limit = numpy.nanmin([lower_limit, numpy.nanpercentile(row_med_bias, 0.2), numpy.nanpercentile(resid, 0.2)])
                upper_limit = numpy.nanmax([upper_limit, numpy.nanpercentile(row_med_bias, 99.8), numpy.nanpercentile(resid, 99.8)])
            elif method == "fit":
                resid = numpy.nanmean(curr_data, axis=0) - wifes_bias_model(p1, linx, camera)
                plt.plot(
                    linx,
                    resid,
                    "g",
                    label="residual",
                )
                model_fit = wifes_bias_model(p1, linx, camera)
                plt.plot(
                    linx, model_fit, "r", label="model fit"
                )
                lower_limit = numpy.nanmin([lower_limit, numpy.nanpercentile(resid, 0.2), numpy.nanpercentile(model_fit, 0.2)])
                upper_limit = numpy.nanmax([upper_limit, numpy.nanpercentile(resid, 99.8), numpy.nanpercentile(model_fit, 99.8)])

            plt.axhline(0, numpy.nanmin(linx), numpy.nanmax(linx), color="k", ls='--')
            plt.xlabel("x [pixels]")
            plt.ylabel(" bias signal collapsed along y")
            plt.legend()
            plt.xlim([numpy.nanmin(linx), numpy.nanmax(linx)])
            plt.ylim(lower_limit, upper_limit)
            plt.title("Fitting bias frame %s" % bias_img.split("/")[-1])
            plot_path = os.path.join(plot_dir, f"{save_prefix}.png")
            plt.savefig(plot_path, dpi=300)
            plt.close()

    # ----------------------------------------------------
    # save it!
    outfits = pyfits.HDUList(f1)
    outfits[data_hdu].data = out_data.astype("float32", casting="same_kind")
    outfits[data_hdu].scale("float32")
    outfits[data_hdu].header.set("PYWIFES", __version__, "PyWiFeS version")
    outfits[data_hdu].header.set("PYWBMTHD", method, "PyWiFeS: bias fit method")
    outfits.writeto(outimg, overwrite=True)
    f1.close()
    return


# ------------------------------------------------------------------------
# WIFES specific tasks
def derive_slitlet_profiles(
    flatfield_fn,
    output_fn,
    data_hdu=0,
    verbose=False,
    shift_global=True,
    interactive_plot=False,
    bin_x=None,
    bin_y=None,
    debug=False,
):
    """
    From input flatfield exposure, determine the locations of each slit.

    Parameters
    ----------
    flatfield_fn : str
        The path to the input flatfield image.
    output_fn : str
        The path to the output file with slitlet positions.
    data_hdu : int, optional
        The HDU index for the data extension in the input images. Default is 0.
    verbose : bool, optional
        Whether to report shifts relative to default slit positions. Default is False.
    shift_global : bool, optional
        Whether to shift all slits by the mean shift value (versus shifting slits individually).
        Default is True.
    interactive_plot : bool, optional
        Whether to interrupt processing to provide interactive plot to user. Default is False.
    bin_x : int, optional
        If specified, override the CCD binning indicated in the header with this value for the x axis.
        Default is None.
    bin_y : int, optional
        If specified, override the CCD binning indicated in the header with this value for the y axis.
        Default is None.
    debug : bool, optional
        Whether to report the parameters used in this function call. Default is False.

    Returns
    -------
    None.

    """
    if debug:
        print(arguments())
    f = pyfits.open(flatfield_fn)
    flat_data = f[data_hdu].data
    orig_hdr = f[data_hdu].header
    f.close()
    # check if halfframe
    halfframe = is_halfframe(flatfield_fn)
    # check for binning, if not specified read from header
    try:
        default_bin_x, default_bin_y = [int(b) for b in orig_hdr["CCDSUM"].split()]
    except Exception:
        default_bin_x = 1
        default_bin_y = 1
    if bin_x is None:
        bin_x = default_bin_x
    if bin_y is None:
        bin_y = default_bin_y
    # first check which camera it is
    if orig_hdr["CAMERA"] == "WiFeSRed":
        baseline_defs = red_slitlet_defs
    else:
        baseline_defs = blue_slitlet_defs
    # now fit slitlet profiles!!
    new_slitlet_defs = {}
    y_shift_vals = []
    if halfframe:
        if is_taros(flatfield_fn):
            nslits = 13  # Only using 12, but need slit definitions for half-slit in interslice cleanup
            first_slit = 1
            offset = 2056 // bin_y
        else:
            nslits = 13
            first_slit = 7
            offset = 1028 // bin_y
    else:
        nslits = 25
        first_slit = 1
        offset = 0
    for i in range(first_slit, first_slit + nslits):
        init_curr_defs = baseline_defs[str(i)]
        # convert to binned values
        curr_defs = [
            ((init_curr_defs[0] - 1) // bin_x) + 1,
            ((init_curr_defs[1] - 1) // bin_x) + 1,
            ((init_curr_defs[2] - 1) // bin_y) + 1,
            ((init_curr_defs[3] - 1) // bin_y) + 1,
        ]
        # check data outside these regions by 20/bin_y pixels
        y_buff = 20 // bin_y
        mod_defs = [
            curr_defs[0] - 1,
            curr_defs[1],
            max(0, curr_defs[2] - 1 - y_buff - offset),
            min(curr_defs[3] + y_buff - offset, flat_data.shape[0]),
        ]
        expanded_data = flat_data[mod_defs[2]:mod_defs[3], mod_defs[0]:mod_defs[1]]
        # ------------------
        # fit for best new center!
        init_yprof = numpy.nansum(expanded_data, axis=1)

        y_prof = ((init_yprof - numpy.nanmin(init_yprof)) /
                  (numpy.nanmax(init_yprof) - numpy.nanmin(init_yprof)))

        # center = halfway between edges where it drops below 10 percent of peak
        bright_inds = numpy.nonzero(y_prof > 0.1)[0]
        new_ymin = bright_inds[0] - 1
        new_ymax = bright_inds[-1] + 1
        orig_ctr = 0.5 * float(len(y_prof))
        new_ctr = 0.5 * (new_ymin + new_ymax)
        # now adjust the slitlet definitions!
        y_shift = int(bin_y * (new_ctr - orig_ctr))
        if interactive_plot:
            plt.figure()
            plt.plot(y_prof, color="b")
            plt.axvline(orig_ctr, color="r")
            plt.axvline(new_ctr, color="g")
            plt.ylabel("Relative counts")
            plt.xlabel("y pixel")
            plt.title(f"Slitlet {i} (red=orig, green=new)")
            plt.show()
        if verbose:
            print(
                "Fitted shift of %d (unbinned) pixels for slitlet %d" % (y_shift, i)
            )
        y_shift_vals.append(y_shift)
        final_defs = [
            init_curr_defs[0],
            init_curr_defs[1],
            init_curr_defs[2] + y_shift,
            init_curr_defs[3] + y_shift,
        ]
        new_slitlet_defs[str(i)] = final_defs
    if interactive_plot:
        plt.imshow(flat_data, norm=colors.LogNorm(), origin="lower")
        # More elements added in loop below

    # finally, use a single global shift if requested
    if shift_global:
        best_shift = int(numpy.nanmean(numpy.array(y_shift_vals)))
        if verbose:
            print("Best global shift is %d (unbinned) pixels" % best_shift)
        final_slitlet_defs = {}
        for i in range(first_slit, first_slit + nslits):
            init_curr_defs = baseline_defs[str(i)]
            final_defs = [
                init_curr_defs[0],
                init_curr_defs[1],
                init_curr_defs[2] + best_shift,
                init_curr_defs[3] + best_shift,
            ]
            final_slitlet_defs[str(i)] = final_defs
    else:
        final_slitlet_defs = new_slitlet_defs
    if interactive_plot:
        for i in range(first_slit, first_slit + nslits):
            plt.axhline(final_slitlet_defs[str(i)][2] // bin_y - offset, color="k", lw=2)
            plt.axhline(final_slitlet_defs[str(i)][3] // bin_y - offset, color="k", lw=2)
        plt.title("Flat - black lines show slit boundaries")
        plt.xlabel("x pixel")
        plt.ylabel("y pixel")
        plt.show()
    # save it!
    f3 = open(output_fn, "wb")
    pickle.dump(final_slitlet_defs, f3)
    f3.close()
    return


def interslice_cleanup(
    input_fn,
    output_fn,
    slitlet_def_file=None,
    bin_x=None,
    bin_y=None,
    data_hdu=0,
    offset=0.4,
    buffer=0,
    radius=10,
    nsig_lim=5.0,
    verbose=False,
    plot=True,
    plot_dir=".",
    save_prefix="cleanup_",
    method="2D",
    debug=False,
    interactive_plot=False,
):
    """
    Uses the dark interslice regions of the detector to interpolate the
    scattered light over the entire detector.

    Note that setting nsig_lim to 3 (as it historically has been) can cause
    problems by sigma clipping the scattered light, not just cosmic rays. For
    a halframe detector with y binning 2, there are 1024 x 4096 pixels.
    With sigma values as follows, this is how many pixels lie above:
    - 3.0 sigma (99.73%): ~11,323 px
    - 4.0 sigma (99.994%): ~265 px
    - 4.5 sigma (99.9993%): ~28 px
    - 5.0 sigma (99.99994%): ~2 px

    (This however assumes a Gaussian distribution, which isn't at all the case
    but still useful as a guide/metric.)

    Parameters
    ----------
    input_fn : str
        The path to the input flatfield image.
    output_fn : str
        The path to the output file with slitlet positions.
    slitlet_def_file : str, optional
        Path to a file containing slitlet definitions. If not specified, baseline values will be used.
    bin_x : int, optional
        If specified, override the CCD binning indicated in the header with this value for the x axis.
        Default is None.
    bin_y : int, optional
        If specified, override the CCD binning indicated in the header with this value for the y axis.
        Default is None.
    data_hdu : int, optional
        The HDU index for the data extension in the input images. Default is 0.
    offset : float, optional
        Fraction of median whole-frame background level to add back to flat to avoid negative values.
        Default is 0.4.
    buffer : int, optional
        Number of y-axis pixels on either side of each slitlet in which to avoid background fitting.
        Default is 0.
    radius : float, optional
        Sigma for 2D Gaussian filtering (with "method" = "2D"). Equal in both axes. Default is 10.
    nsig_lim : float, optional
        Clipping threshold in number of standard deviations above median. Computed independently for
        each slitlet. Default is 5.0.
    verbose : bool, optional
        Whether to report progress. Default is False.
    plot : bool, optional
        Whether to output a diagnostic plot. Default is True.
    plot_dir : str, optional
        Directory for output of plot (if requested). Default is '.'.
    save_prefix : str, optional
        Prefix for plot (if requested). Default is 'cleanup\_'.
    method : str, optional
        Method to fit interslice region. Options are "2D" (fit 2D Gaussian-smoothed shape
        to interslice segments), "1D" (use median value of each cosmic-ray-filtered
        interslice segment). Default is "2D".
    debug : bool, optional
        Whether to report the parameters used in this function call. Default is False.
    interactive_plot : bool, optional
        Whether to interrupt processing to provide interactive plot to user. Default is False.

    Returns
    -------
    None
    """
    if debug:
        print(arguments())

    # ------------------------------------
    # 1) Open the flat field
    f = pyfits.open(input_fn)
    header = f[data_hdu].header
    data = f[data_hdu].data
    f.close()

    # Check if half-frame
    halfframe = is_halfframe(input_fn)
    taros = is_taros(header)
    orig_y = data.shape[0]

    # check which channel (blue / red) it is!!
    camera = header["CAMERA"]

    # ------------------------------------
    # 2) Get the slitlets boundaries
    if slitlet_def_file is not None:
        f2 = open(slitlet_def_file, "rb")
        init_slitlet_defs = pickle.load(f2)
        f2.close()
    elif camera == "WiFeSRed":
        init_slitlet_defs = red_slitlet_defs
    else:
        init_slitlet_defs = blue_slitlet_defs

    # check for binning, if no specified read from header
    try:
        default_bin_x, default_bin_y = [int(b) for b in header["CCDSUM"].split()]
    except Exception:
        default_bin_x = 1
        default_bin_y = 1
    if bin_x is None:
        bin_x = default_bin_x
    if bin_y is None:
        bin_y = default_bin_y

    # If halfframe we use slits 7-19 (1-12 under Taros since slit 13 is cut in half),
    # and if fullframe we use slits 1-25. When in halfframe we need to consider an
    # offset, as we're using the middle of the data array. The frame_offset is in
    # unbinned pixels.
    if halfframe:
        if taros:
            nslits = 13  # only keeping 12 but want the definitions for the half-slit
            first_slit = 1
            frame_offset = 2056
        else:
            nslits = 13
            first_slit = 7
            frame_offset = 1028

    else:
        nslits = 25
        first_slit = 1
        frame_offset = 0

    last_slit = first_slit + nslits - 1
    slitlets_n = numpy.arange(first_slit, last_slit + 1)

    slitlet_defs = {}
    for i in slitlets_n:
        slit_num = str(i)
        init_slitdefs = init_slitlet_defs[slit_num]
        slitlet_defs[slit_num] = [
            init_slitdefs[0],
            init_slitdefs[1],
            max(init_slitdefs[2] - buffer - frame_offset, 0),
            min(init_slitdefs[3] + buffer - frame_offset, orig_y * bin_y),
        ]

    # ------------------------------------
    # 3) Create temporary storage structure
    inter_smooth = numpy.zeros_like(data)
    fitted = numpy.zeros_like(data)

    # ------------------------------------
    # 4) Get rid of the 'science-full region' and leave only the interslice
    # (and smooth it as well)
    for slit in slitlets_n:
        if verbose:
            print(f"Blanking and smoothing slit: {slit}")
        # Get the slit boundaries
        [xmin, xmax, ymin, ymax] = slitlet_defs[str(slit)]
        # Account for binning
        xmin = numpy.round(xmin // bin_x)
        xmax = numpy.round(xmax // bin_x)
        ymin = numpy.round(ymin // bin_y)
        ymax = numpy.round(ymax // bin_y)

        # Clear the appropriate region in temporary structure
        inter_smooth[ymin:ymax, xmin:xmax] = numpy.nan
        # Now, smooth the remaining interslice region
        symin = ymax
        if slit == first_slit:
            symax = data.shape[0]
        else:
            symax = numpy.round(slitlet_defs[str(slit - 1)][2] // bin_y)

        # Need to get rid of cosmic rays
        # here's a quick and dirty way of doing it ...
        tmp = numpy.zeros_like(data[symin:symax, xmin:xmax])
        median = numpy.nanmedian(data[symin:symax, xmin:xmax])
        std = numpy.nanstd(data[symin:symax, xmin:xmax])
        tmp[data[symin:symax, xmin:xmax] < (median + nsig_lim * std)] = data[
            symin:symax, xmin:xmax
        ][data[symin:symax, xmin:xmax] < (median + nsig_lim * std)]

        if interactive_plot:
            plt.imshow(tmp, aspect='auto', origin='lower')
            plt.title(f"Slitlet {slit}: y=({symin}:{symax}), data < {median:.2f} + {nsig_lim} * {std:.2f}")
            plt.show()
        # Perform smoothing
        if method == "2D":
            inter_smooth[symin:symax, xmin:xmax] = ndimage.gaussian_filter(
                tmp, sigma=[radius, radius]
            )
        elif method == "1D":
            inter_smooth[symin:symax, xmin:xmax] += numpy.nanmedian(tmp)
        if interactive_plot:
            plt.imshow(inter_smooth[symin:symax, xmin:xmax], aspect='auto', origin='lower')
            plt.title(f"Slitlet {slit}: inter_smooth y=({symin}:{symax})")
            plt.show()

        # And don't forget below the last slice
        if slit == last_slit:
            symin = 1
            symax = ymin
            if method == "2D":
                inter_smooth[symin:symax, xmin:xmax] = ndimage.gaussian_filter(
                    data[symin:symax, xmin:xmax], sigma=[radius, radius]
                )
            elif method == "1D":
                inter_smooth[symin:symax, xmin:xmax] += numpy.nanmedian(tmp)
            if interactive_plot:
                plt.imshow(inter_smooth[symin:symax, xmin:xmax], aspect='auto', origin='lower')
                plt.title(f"Slitlet {slit}: extra inter_smooth[{symin}:{symax}, {xmin}:{xmax}]")
                plt.show()

    # ------------------------------------
    # 5) Great, now we can interpolate this and reconstruct the contamination
    # Do each slitlet individually to avoid overloading the memory
    # Sampling
    dx = 10
    dy = 3
    for slit in slitlets_n:
        if verbose:
            print(f"Interpolating slitlet {slit}")
        [xmin, xmax, y2, y3] = slitlet_defs[str(slit)]
        # Account for binning
        xmin = numpy.round(xmin // bin_x)
        xmax = numpy.round(xmax // bin_x)
        y2 = numpy.round(y2 // bin_y)
        y3 = numpy.round(y3 // bin_y)

        if slit == first_slit:
            y4 = numpy.shape(data)[0]
            y1 = numpy.round(slitlet_defs[str(slit + 1)][3] // bin_y)
        elif slit == last_slit:
            # Special case we have to extrapolate for the half slitlet with Taros
            # rather than interpolate as there is not dark interslit region on the
            # other side.
            if taros and halfframe:
                y4 = numpy.round(slitlet_defs[str(slit - 1)][2] // bin_y)
                y1 = y2
            else:
                y4 = numpy.round(slitlet_defs[str(slit - 1)][2] // bin_y)
                y1 = 1
        else:
            y4 = numpy.round(slitlet_defs[str(slit - 1)][2] // bin_y)
            y1 = numpy.round(slitlet_defs[str(slit + 1)][3] // bin_y)

        # Select a subsample of point to do the integration
        x = numpy.arange(xmin + 3, xmax - 3, dx)

        if taros and halfframe and slit == last_slit:
            y = numpy.arange(y3 + 1, y4 - 1, dy)
        else:
            y = numpy.append(
                numpy.arange(y1 + 1, y2 - 1, dy), numpy.arange(y3 + 1, y4 - 1, dy)
            )
        grid = numpy.zeros((len(y), len(x)))
        for i in range(len(x)):
            for j in range(len(y)):
                grid[j, i] = inter_smooth[y[j], x[i]]

        # ------------------------------------
        # 6) Actually perform the interpolation

        # Note : because of the large gap to fill, kx and ky have little effect
        # reconstruct missing slice
        xall = numpy.arange(xmin, xmax, 1)
        yall = numpy.arange(y1, y4, 1)

        func = interp.RectBivariateSpline(y, x, grid, kx=1, ky=1)
        fitted[y1:y4, xmin:xmax] = func(yall, xall)

        if interactive_plot:
            plt.imshow(grid, aspect='auto', origin='lower')
            plt.title(f"Slitlet {slit} - grid[{y1}:{y4}, {xmin}:{xmax}]")
            plt.show()
            plt.imshow(fitted[y1:y4, xmin:xmax], aspect='auto', origin='lower')
            plt.title(f"Slitlet {slit} - fitted[{y1}:{y4}, {xmin}:{xmax}]")
            plt.show()

    # ------------------------------------
    # 7) All done ! Let's save it all ...
    f = pyfits.open(input_fn)
    f[0].data = fitted
    # Save the corretion image separately just in case
    f.writeto(
        input_fn[:-5] + "_corr.fits", overwrite=True
    )
    # Add back in a small offset to avoid the flat approaching zero
    applied_offset = numpy.round(offset * numpy.nanmedian(fitted), 1)
    f[0].data = data - fitted + applied_offset
    f[0].header.set("PYWIFES", __version__, "PyWiFeS version")
    f[0].header.set("PYWICOFF", applied_offset, "PyWiFeS: interslice cleanup additive offset")
    f[0].header.set("PYWICBUF", buffer, "PyWiFeS: interslice clenaup buffer")
    if method == "2D":
        f[0].header.set("PYWICRAD", radius, "PyWiFeS: interslice cleanup radius")
        f[0].header.set("PYWICSLM", nsig_lim, "PyWiFeS: interslice cleanup nsig_lim")
    f.writeto(output_fn, overwrite=True)
    f.close()
    if verbose:
        print(" Additive offset:", applied_offset)
    # 8) Plot anything ?
    if plot:
        myvmax = numpy.nanmax(fitted)

        # Create a figure with subplots
        fig, axs = plt.subplots(3, 1, figsize=(11, 10))  # 3 row, 1 columns
        text_size = 18
        # Plot and save the first subplot
        axs[0].imshow(data, vmin=0, vmax=myvmax, cmap="nipy_spectral", origin="lower", aspect='auto')
        axs[0].set_title("Pre-corrected " + os.path.basename(input_fn), size=text_size)

        # Plot and save the second subplot
        axs[1].imshow(fitted, vmin=0, vmax=myvmax, cmap="nipy_spectral", origin="lower", aspect='auto')
        axs[1].set_title("Fitted contamination for " + input_fn.split("/")[-1], size=text_size)
        axs[1].set_ylabel('Y-axis [pixel]', size=text_size)

        # Plot and save the third subplot
        axs[2].imshow(data - fitted + applied_offset, vmin=0, vmax=myvmax, cmap="nipy_spectral", origin="lower", aspect='auto')
        axs[2].set_title("Corrected " + os.path.basename(output_fn), size=text_size)
        axs[2].set_xlabel('X-axis [pixel]', size=text_size)

        # Adjust layout to prevent overlapping of titles
        plt.tight_layout()
        fn_no_extension = os.path.splitext(os.path.basename(output_fn))[0]
        plot_name = save_prefix + fn_no_extension + ".png"
        plot_path = os.path.join(plot_dir, plot_name)
        plt.savefig(plot_path, dpi=300)
        plt.close()

    return


def wifes_slitlet_mef(
    inimg, outimg, data_hdu=0, bin_x=None, bin_y=None, slitlet_def_file=None,
    nan_method="interp", repl_val=0.0, debug=False,
):
    """
    Create multi-extension FITS file from a single input image.

    Parameters
    ----------
    inimg : str
        The path to the input image.
    outimg : str
        The path to the output multi-extension image.
    data_hdu : int, optional
        The HDU index for the data extension in the input images. Default is 0.
    bin_x : int, optional
        If specified, override the CCD binning indicated in the header with this value for the x axis.
        Default is None.
    bin_y : int, optional
        If specified, override the CCD binning indicated in the header with this value for the y axis.
        Default is None.
    slitlet_def_file : str, optional
        Path to a file containing slitlet definitions. If not specified, baseline values will be used.
    nan_method : str, optional
        Method to treat NaN pixels. Options are "interp" (apply linear interpolation in x-direction),
        "replace" (replace with constant value). Default is "interp".
    repl_val : float, optional
        Constant to replace NaN values with if "method" = "replace". Default is 0.0.
    debug : bool, optional
        Whether to report the parameters used in this function call. Default is False.

    Returns
    -------
    None.
    """
    if debug:
        print(arguments())
    f = pyfits.open(inimg)
    outfits = pyfits.HDUList([pyfits.PrimaryHDU(header=f[0].header)])
    old_hdr = f[data_hdu].header
    full_data = f[data_hdu].data
    f.close()

    # check if halfframe
    halfframe = is_halfframe(inimg)
    # check which channel (blue / red) it is!!
    camera = old_hdr["CAMERA"]
    rdnoise = old_hdr["RDNOISE"]

    # get slitlet definitions!
    # new ones if defined, otherwise use baseline values!
    if slitlet_def_file is not None:
        f2 = open(slitlet_def_file, "rb")
        try:
            slitlet_defs = pickle.load(f2, fix_imports=True, encoding="latin")
        except Exception:
            slitlet_defs = pickle.load(f2)  # for python 2.7
        f2.close()
    elif camera == "WiFeSRed":
        slitlet_defs = red_slitlet_defs
    else:
        slitlet_defs = blue_slitlet_defs

    # check for binning, if no specified read from header
    try:
        default_bin_x, default_bin_y = [int(b) for b in old_hdr["CCDSUM"].split()]
    except Exception:
        default_bin_x = 1
        default_bin_y = 1
    if bin_x is None:
        bin_x = default_bin_x
    if bin_y is None:
        bin_y = default_bin_y

    dq_img = numpy.zeros(numpy.shape(full_data), dtype="int16")
    var_img = full_data + rdnoise**2

    # flag bad pixels on new detectors
    epoch = determine_detector_epoch(inimg)
    if int(float(epoch[1])) > 3:
        var_img[numpy.isnan(full_data)] = 65535.**2
        dq_img[numpy.isnan(full_data)] = 1

        # Linear row-by-row x-interpolation over NaNs from bad pixel mask
        # (after adjusting VAR and DQ)
        if nan_method == "interp":
            for row in numpy.arange(full_data.shape[0]):
                if numpy.any(numpy.isnan(full_data[row, :])):
                    nans, x = nan_helper(full_data[row, :])
                    full_data[row, :][nans] = numpy.interp(x(nans), x(~nans),
                                                           full_data[row, :][~nans])
        # Replace NaNs with indicated value
        elif nan_method == "replace":
            full_data[numpy.isnan(full_data)] = repl_val
        else:
            raise ValueError(f"Unknown nan_method '{nan_method}' in wifes_slitlet_mef "
                             f"for creation of {os.path.basename(outimg)}")

    # ---------------------------
    # for each slitlet, save it to a single header extension
    if halfframe:
        if is_taros(inimg):
            nslits = 12
            first_slit = 1
            offset = 2056 // bin_y
        else:
            nslits = 13
            first_slit = 7
            offset = 1028 // bin_y
    else:
        nslits = 25
        first_slit = 1
        offset = 0

    for i in range(first_slit, first_slit + nslits):
        init_curr_defs = slitlet_defs[str(i)]
        # convert to binned values
        curr_defs = [
            ((init_curr_defs[0] - 1) // bin_x) + 1,
            ((init_curr_defs[1] - 1) // bin_x) + 1,
            ((init_curr_defs[2] - 1) // bin_y) + 1,
            ((init_curr_defs[3] - 1) // bin_y) + 1,
        ]
        # horrible kluge to make sure everything has the same dimensions!
        if (curr_defs[1] - curr_defs[0] + 1) != (4096 // bin_x):
            curr_defs[1] -= 1
        if (curr_defs[3] - curr_defs[2] + 1) != (86 // bin_y):
            curr_defs[3] -= 1
        # get the data
        dim_str = "[%d:%d,%d:%d]" % (
            curr_defs[0],
            curr_defs[1],
            curr_defs[2],
            curr_defs[3],
        )
        mod_defs = [
            curr_defs[0] - 1,
            curr_defs[1],
            curr_defs[2] - 1 - offset,
            curr_defs[3] - offset,
        ]
        new_data = full_data[mod_defs[2]:mod_defs[3], mod_defs[0]:mod_defs[1]]
        # create fits hdu
        hdu_name = "SCI%d" % i
        new_hdu = pyfits.ImageHDU(new_data.astype("float32", casting="same_kind"), old_hdr, name=hdu_name)
        new_hdu.header.set("DETSEC", dim_str)
        new_hdu.header.set("DATASEC", dim_str)
        new_hdu.header.set("TRIMSEC", dim_str)
        new_hdu.scale("float32")
        outfits.append(new_hdu)
        gc.collect()
    # VARIANCE EXTENSIONS
    for i in range(first_slit, first_slit + nslits):
        init_curr_defs = slitlet_defs[str(i)]
        # convert to binned values
        curr_defs = [
            ((init_curr_defs[0] - 1) // bin_x) + 1,
            ((init_curr_defs[1] - 1) // bin_x) + 1,
            ((init_curr_defs[2] - 1) // bin_y) + 1,
            ((init_curr_defs[3] - 1) // bin_y) + 1,
        ]
        # horrible kluge to make sure everything has the same dimensions!
        if (curr_defs[1] - curr_defs[0] + 1) != (4096 // bin_x):
            curr_defs[1] -= 1
        if (curr_defs[3] - curr_defs[2] + 1) != (86 // bin_y):
            curr_defs[3] -= 1
        # get the data
        dim_str = "[%d:%d,%d:%d]" % (
            curr_defs[0],
            curr_defs[1],
            curr_defs[2],
            curr_defs[3],
        )
        mod_defs = [
            curr_defs[0] - 1,
            curr_defs[1],
            curr_defs[2] - 1 - offset,
            curr_defs[3] - offset,
        ]
        var_data = var_img[mod_defs[2]:mod_defs[3], mod_defs[0]:mod_defs[1]]
        # create fits hdu
        hdu_name = "VAR%d" % i
        new_hdu = pyfits.ImageHDU(var_data.astype("float64", casting="same_kind"), old_hdr, name=hdu_name)
        new_hdu.header.set("DETSEC", dim_str)
        new_hdu.header.set("DATASEC", dim_str)
        new_hdu.header.set("TRIMSEC", dim_str)
        new_hdu.scale('float64')
        outfits.append(new_hdu)
        gc.collect()
    # DATA QUALITY EXTENSIONS
    for i in range(first_slit, first_slit + nslits):
        init_curr_defs = slitlet_defs[str(i)]
        # convert to binned values
        curr_defs = [
            ((init_curr_defs[0] - 1) // bin_x) + 1,
            ((init_curr_defs[1] - 1) // bin_x) + 1,
            ((init_curr_defs[2] - 1) // bin_y) + 1,
            ((init_curr_defs[3] - 1) // bin_y) + 1,
        ]
        # horrible kluge to make sure everything has the same dimensions!
        if (curr_defs[1] - curr_defs[0] + 1) != (4096 // bin_x):
            curr_defs[1] -= 1
        if (curr_defs[3] - curr_defs[2] + 1) != (86 // bin_y):
            curr_defs[3] -= 1
        # get the data
        dim_str = "[%d:%d,%d:%d]" % (
            curr_defs[0],
            curr_defs[1],
            curr_defs[2],
            curr_defs[3],
        )
        mod_defs = [
            curr_defs[0] - 1,
            curr_defs[1],
            curr_defs[2] - 1 - offset,
            curr_defs[3] - offset,
        ]
        dq_data = dq_img[mod_defs[2]:mod_defs[3], mod_defs[0]:mod_defs[1]]
        # trim data beyond range
        dq_data[dq_data > 32767] = 32767
        dq_data[dq_data < -32768] = -32768
        hdu_name = "DQ%d" % i
        new_hdu = pyfits.ImageHDU(dq_data.astype("int16", casting="unsafe"), old_hdr, name=hdu_name)
        new_hdu.header.set("DETSEC", dim_str)
        new_hdu.header.set("DATASEC", dim_str)
        new_hdu.header.set("TRIMSEC", dim_str)
        new_hdu.scale('int16')
        outfits.append(new_hdu)
        gc.collect()
    outfits[0].header.set("PYWIFES", __version__, "PyWiFeS version")
    outfits[0].header.set("PYWSMINT", nan_method,
                          "PyWiFeS: method for bad pixels at MEF creation")
    outfits.writeto(outimg, overwrite=True)
    return


def wifes_slitlet_mef_ns(
    inimg,
    outimg_obj,
    outimg_sky,
    data_hdu=0,
    bin_x=None,
    bin_y=None,
    nod_dy=80,
    slitlet_def_file=None,
    nan_method="interp",
    repl_val=0.0,
    debug=False,
):
    """
    Create multi-extension FITS files for object and sky data from a single input image.

    Parameters
    ----------
    inimg : str
        Path to the input FITS image.
    outimg_obj : str
        Path to the output FITS file for object data.
    outimg_sky : str
        Path to the output FITS file for sky data.
    data_hdu : int, optional
        Index of the HDU containing the data in the input FITS image. Default is 0.
    bin_x : int, optional
        Binning factor in the x-direction. If not specified, it will be read from the header.
    bin_y : int, optional
        Binning factor in the y-direction. If not specified, it will be read from the header.
    nod_dy : int, optional
        Offset in pixels between the object and sky slitlets in the y-direction. Default is 80.
    slitlet_def_file : str, optional
        Path to a file containing slitlet definitions. If not specified, baseline values will be used.
    nan_method : str, optional
        Method for handling bad (NaN) pixels. Options are 'interp', 'replace'.
        Default: "interp".
    repl_val : float, optional
        If 'nan_method'='replace', replace bad (NaN) pixels with the specified value.
        Default: 0.0.
    debug : bool, optional
        Whether to report the parameters used in this function call.
        Default: False.

    Returns
    -------
    None
    """
    if debug:
        print(arguments())
    f = pyfits.open(inimg)
    outfits_obj = pyfits.HDUList([pyfits.PrimaryHDU(header=f[0].header)])
    outfits_sky = pyfits.HDUList([pyfits.PrimaryHDU(header=f[0].header)])
    old_hdr = f[data_hdu].header
    full_data = f[data_hdu].data
    f.close()
    rdnoise = old_hdr["RDNOISE"]
    # check which channel (blue / red) it is!!
    camera = old_hdr["CAMERA"]
    # get slitlet definitions!
    # new ones if defined, otherwise use baseline values!
    if slitlet_def_file is not None:
        f2 = open(slitlet_def_file, "rb")
        slitlet_defs = pickle.load(f2, fix_imports=True, encoding="latin")
        f2.close()
    elif camera == "WiFeSRed":
        slitlet_defs = red_slitlet_defs
    else:
        slitlet_defs = blue_slitlet_defs
    # check for binning, if no specified read from header
    try:
        default_bin_x, default_bin_y = [int(b) for b in old_hdr["CCDSUM"].split()]
    except Exception:
        default_bin_x = 1
        default_bin_y = 1
    if bin_x is None:
        bin_x = default_bin_x
    if bin_y is None:
        bin_y = default_bin_y

    # Set up baseline arrays
    dq_img = numpy.zeros(numpy.shape(full_data), dtype="int16")
    var_img = full_data + rdnoise**2

    # flag bad pixels on new detectors
    epoch = determine_detector_epoch(inimg)
    if int(float(epoch[1])) > 3:
        var_img[numpy.isnan(full_data)] = 65535.**2
        dq_img[numpy.isnan(full_data)] = 1

        # Linear row-by-row x-interpolation over NaNs from bad pixel mask (after adjusting VAR and DQ)
        if nan_method == "interp":
            for row in numpy.arange(full_data.shape[0]):
                if numpy.any(numpy.isnan(full_data[row, :])):
                    nans, x = nan_helper(full_data[row, :])
                    full_data[row, :][nans] = numpy.interp(x(nans), x(~nans), full_data[row, :][~nans])
        # Replace NaNs with indicated value
        elif nan_method == "replace":
            full_data[numpy.isnan(full_data)] = repl_val

    # ------------------------------------
    # for each slitlet, save it to a single header extension
    for i in range(nslits):
        init_curr_defs = slitlet_defs[str(i + 1)]
        # ------------------
        # convert to binned values
        obj_defs = [
            ((init_curr_defs[0] - 1) // bin_x) + 1,
            ((init_curr_defs[1] - 1) // bin_x) + 1,
            ((init_curr_defs[2] - 1) // bin_y) + 1,
            ((init_curr_defs[3] - 1) // bin_y) + 1,
        ]
        sky_defs = [
            ((init_curr_defs[0] - 1) // bin_x) + 1,
            ((init_curr_defs[1] - 1) // bin_x) + 1,
            ((init_curr_defs[2] - 1 + nod_dy) // bin_y) + 1,
            ((init_curr_defs[3] - 1 + nod_dy) // bin_y) + 1,
        ]
        # ------------------
        # horrible kluge to make sure everything has the same dimensions!
        if (obj_defs[1] - obj_defs[0] + 1) != (4096 // bin_x):
            obj_defs[1] -= 1
        if (obj_defs[3] - obj_defs[2] + 1) != (86 // bin_y):
            obj_defs[3] -= 1
        if (sky_defs[1] - sky_defs[0] + 1) != (4096 // bin_x):
            sky_defs[1] -= 1
        if (sky_defs[3] - sky_defs[2] + 1) != (86 // bin_y):
            sky_defs[3] -= (sky_defs[3] - sky_defs[2] + 1) - (86 // bin_y)
            # sky_defs[3] -= 1
        # ------------------
        # get the object data
        obj_dim_str = "[%d:%d,%d:%d]" % (
            obj_defs[0],
            obj_defs[1],
            obj_defs[2],
            obj_defs[3],
        )
        obj_mod_defs = [obj_defs[0] - 1, obj_defs[1], obj_defs[2] - 1, obj_defs[3]]
        obj_data = full_data[
            obj_mod_defs[2]:obj_mod_defs[3], obj_mod_defs[0]:obj_mod_defs[1]
        ]
        # kill outer 3//bin_y pixels!!
        ykill = 4 // bin_y
        obj_data[:ykill, :] *= 0.0
        obj_data[-ykill:, :] *= 0.0
        # create fits hdu for object
        hdu_name = "SCI%d" % (i + 1)
        obj_hdu = pyfits.ImageHDU(obj_data.astype("float32", casting="same_kind"), old_hdr, name=hdu_name)
        obj_hdu.header.set("DETSEC", obj_dim_str)
        obj_hdu.header.set("DATASEC", obj_dim_str)
        obj_hdu.header.set("TRIMSEC", obj_dim_str)
        # fix the exposure time!!
        exptime_true = float(old_hdr["EXPTIME"])
        obj_hdu.header.set("EXPTIME", exptime_true, comment="Total NS exposure time")
        obj_hdu.scale("float32")
        outfits_obj.append(obj_hdu)
        # ------------------
        # and the sky data
        sky_dim_str = "[%d:%d,%d:%d]" % (
            sky_defs[0],
            sky_defs[1],
            sky_defs[2],
            sky_defs[3],
        )
        sky_mod_defs = [sky_defs[0] - 1, sky_defs[1], sky_defs[2] - 1, sky_defs[3]]
        nsx = sky_mod_defs[1] - sky_mod_defs[0]
        nsy = sky_mod_defs[3] - sky_mod_defs[2]
        # horrible fix to include bad NS regions definition for slitlet 1
        sky_data = numpy.zeros([nsy, nsx])
        true_sky_data = full_data[
            sky_mod_defs[2]:sky_mod_defs[3], sky_mod_defs[0]:sky_mod_defs[1]
        ]
        tsy, tsx = numpy.shape(true_sky_data)
        sky_data[:tsy, :tsx] = true_sky_data
        # kill outer 3//bin_y pixels!!
        ykill = 4 // bin_y
        sky_data[:ykill, :] *= 0.0
        sky_data[-ykill:, :] *= 0.0
        # create fits hdu for sky
        hdu_name = "SCI%d" % (i + 1)
        sky_hdu = pyfits.ImageHDU(sky_data.astype("float32", casting="same_kind"), old_hdr, name=hdu_name)
        sky_hdu.header.set("DETSEC", sky_dim_str)
        sky_hdu.header.set("DATASEC", sky_dim_str)
        sky_hdu.header.set("TRIMSEC", sky_dim_str)
        # fix the exposure time!!
        exptime_true = float(old_hdr["EXPTIME"])
        sky_hdu.header.set("EXPTIME", exptime_true, comment="Total NS exposure time")
        sky_hdu.scale("float32")
        outfits_sky.append(sky_hdu)
        gc.collect()
    # ------------------------------------
    # VARIANCE EXTENSIONS
    for i in range(nslits):
        init_curr_defs = slitlet_defs[str(i + 1)]
        # ------------------
        # convert to binned values
        obj_defs = [
            ((init_curr_defs[0] - 1) // bin_x) + 1,
            ((init_curr_defs[1] - 1) // bin_x) + 1,
            ((init_curr_defs[2] - 1) // bin_y) + 1,
            ((init_curr_defs[3] - 1) // bin_y) + 1,
        ]
        sky_defs = [
            ((init_curr_defs[0] - 1) // bin_x) + 1,
            ((init_curr_defs[1] - 1) // bin_x) + 1,
            ((init_curr_defs[2] - 1 + nod_dy) // bin_y) + 1,
            ((init_curr_defs[3] - 1 + nod_dy) // bin_y) + 1,
        ]
        # ------------------
        # horrible kluge to make sure everything has the same dimensions!
        if (obj_defs[1] - obj_defs[0] + 1) != (4096 // bin_x):
            obj_defs[1] -= 1
        if (obj_defs[3] - obj_defs[2] + 1) != (86 // bin_y):
            obj_defs[3] -= 1
        if (sky_defs[1] - sky_defs[0] + 1) != (4096 // bin_x):
            sky_defs[1] -= 1
        if (sky_defs[3] - sky_defs[2] + 1) != (86 // bin_y):
            sky_defs[3] -= (sky_defs[3] - sky_defs[2] + 1) - (86 // bin_y)
            # sky_defs[3] -= 1
        # ------------------
        # get the object data
        obj_dim_str = "[%d:%d,%d:%d]" % (
            obj_defs[0],
            obj_defs[1],
            obj_defs[2],
            obj_defs[3],
        )
        obj_mod_defs = [obj_defs[0] - 1, obj_defs[1], obj_defs[2] - 1, obj_defs[3]]
        obj_var = var_img[
            obj_mod_defs[2]:obj_mod_defs[3], obj_mod_defs[0]:obj_mod_defs[1]
        ]
        # kill outer 3//bin_y pixels!!
        ykill = 4 // bin_y
        obj_var[:ykill, :] *= 0.0
        obj_var[-ykill:, :] *= 0.0
        # create fits hdu for object variance
        hdu_name = "VAR%d" % (i + 1)
        obj_hdu = pyfits.ImageHDU(obj_var.astype("float64", casting="same_kind"), old_hdr, name=hdu_name)
        obj_hdu.header.set("DETSEC", obj_dim_str)
        obj_hdu.header.set("DATASEC", obj_dim_str)
        obj_hdu.header.set("TRIMSEC", obj_dim_str)
        # fix the exposure time!!
        exptime_true = float(old_hdr["EXPTIME"])
        obj_hdu.header.set("EXPTIME", exptime_true, comment="Total NS exposure time")
        obj_hdu.scale("float64")
        outfits_obj.append(obj_hdu)
        # ------------------
        # get the sky data
        sky_dim_str = "[%d:%d,%d:%d]" % (
            sky_defs[0],
            sky_defs[1],
            sky_defs[2],
            sky_defs[3],
        )
        sky_mod_defs = [sky_defs[0] - 1, sky_defs[1], sky_defs[2] - 1, sky_defs[3]]
        nsx = sky_mod_defs[1] - sky_mod_defs[0]
        nsy = sky_mod_defs[3] - sky_mod_defs[2]
        # horrible fix to include bad NS regions definition for slitlet 1
        sky_var = numpy.zeros([nsy, nsx])
        true_sky_data = var_img[
            sky_mod_defs[2]:sky_mod_defs[3], sky_mod_defs[0]:sky_mod_defs[1]
        ]
        tsy, tsx = numpy.shape(true_sky_data)
        sky_var[:tsy, :tsx] = true_sky_data
        # kill outer 3//bin_y pixels!!
        ykill = 4 // bin_y
        sky_var[:ykill, :] *= 0.0
        sky_var[-ykill:, :] *= 0.0
        # create fits hdu for sky variance
        hdu_name = "VAR%d" % (i + 1)
        sky_hdu = pyfits.ImageHDU(sky_var.astype("float64", casting="same_kind"), old_hdr, name=hdu_name)
        sky_hdu.header.set("DETSEC", sky_dim_str)
        sky_hdu.header.set("DATASEC", sky_dim_str)
        sky_hdu.header.set("TRIMSEC", sky_dim_str)
        # fix the exposure time!!
        exptime_true = float(old_hdr["EXPTIME"])
        sky_hdu.header.set("EXPTIME", exptime_true, comment="Total NS exposure time")
        sky_hdu.scale("float64")
        outfits_sky.append(sky_hdu)
        gc.collect()
    # ------------------------------------
    # DATA QUALITY EXTENSIONS
    for i in range(nslits):
        init_curr_defs = slitlet_defs[str(i + 1)]
        # ------------------
        # convert to binned values
        obj_defs = [
            ((init_curr_defs[0] - 1) // bin_x) + 1,
            ((init_curr_defs[1] - 1) // bin_x) + 1,
            ((init_curr_defs[2] - 1) // bin_y) + 1,
            ((init_curr_defs[3] - 1) // bin_y) + 1,
        ]
        sky_defs = [
            ((init_curr_defs[0] - 1) // bin_x) + 1,
            ((init_curr_defs[1] - 1) // bin_x) + 1,
            ((init_curr_defs[2] - 1 + nod_dy) // bin_y) + 1,
            ((init_curr_defs[3] - 1 + nod_dy) // bin_y) + 1,
        ]
        # ------------------
        # horrible kluge to make sure everything has the same dimensions!
        if (obj_defs[1] - obj_defs[0] + 1) != (4096 // bin_x):
            obj_defs[1] -= 1
        if (obj_defs[3] - obj_defs[2] + 1) != (86 // bin_y):
            obj_defs[3] -= 1
        if (sky_defs[1] - sky_defs[0] + 1) != (4096 // bin_x):
            sky_defs[1] -= 1
        if (sky_defs[3] - sky_defs[2] + 1) != (86 // bin_y):
            sky_defs[3] -= (sky_defs[3] - sky_defs[2] + 1) - (86 // bin_y)
            # sky_defs[3] -= 1
        # ------------------
        # get the object data
        obj_dim_str = "[%d:%d,%d:%d]" % (
            obj_defs[0],
            obj_defs[1],
            obj_defs[2],
            obj_defs[3],
        )
        obj_mod_defs = [obj_defs[0] - 1, obj_defs[1], obj_defs[2] - 1, obj_defs[3]]
        obj_dq = dq_img[
            obj_mod_defs[2]:obj_mod_defs[3], obj_mod_defs[0]:obj_mod_defs[1]
        ]
        # kill outer 3//bin_y pixels!!
        ykill = 4 // bin_y
        obj_dq[:ykill, :] = 1
        obj_dq[-ykill:, :] = 1
        # trim data beyond range
        obj_dq[obj_dq > 32767] = 32767
        obj_dq[obj_dq < -32768] = -32768
        # ------------------
        # create fits hdu for object DQ
        hdu_name = "DQ%d" % (i + 1)
        obj_hdu = pyfits.ImageHDU(obj_dq.astype("int16", casting="unsafe"), old_hdr, name=hdu_name)
        obj_hdu.header.set("DETSEC", obj_dim_str)
        obj_hdu.header.set("DATASEC", obj_dim_str)
        obj_hdu.header.set("TRIMSEC", obj_dim_str)
        # fix the exposure time!!
        exptime_true = float(old_hdr["EXPTIME"])
        obj_hdu.header.set("EXPTIME", exptime_true, comment="Total NS exposure time")
        obj_hdu.scale("int16")
        outfits_obj.append(obj_hdu)
        # ------------------
        # get the sky data
        sky_dim_str = "[%d:%d,%d:%d]" % (
            sky_defs[0],
            sky_defs[1],
            sky_defs[2],
            sky_defs[3],
        )
        sky_mod_defs = [sky_defs[0] - 1, sky_defs[1], sky_defs[2] - 1, sky_defs[3]]
        nsx = sky_mod_defs[1] - sky_mod_defs[0]
        nsy = sky_mod_defs[3] - sky_mod_defs[2]
        # horrible fix to include bad NS regions definition for slitlet 1
        sky_dq = numpy.zeros([nsy, nsx], dtype="int16")
        true_sky_data = dq_img[
            sky_mod_defs[2]:sky_mod_defs[3], sky_mod_defs[0]:sky_mod_defs[1]
        ]
        tsy, tsx = numpy.shape(true_sky_data)
        sky_dq[:tsy, :tsx] = true_sky_data
        # kill outer 3//bin_y pixels!!
        ykill = 4 // bin_y
        sky_dq[:ykill, :] = 1
        sky_dq[-ykill:, :] = 1
        # trim data beyond range
        sky_dq[sky_dq > 32767] = 32767
        sky_dq[sky_dq < -32768] = -32768
        # create fits hdu for sky DQ
        hdu_name = "DQ%d" % (i + 1)
        sky_hdu = pyfits.ImageHDU(sky_dq.astype("int16", casting="unsafe"), old_hdr, name=hdu_name)
        sky_hdu.header.set("DETSEC", sky_dim_str)
        sky_hdu.header.set("DATASEC", sky_dim_str)
        sky_hdu.header.set("TRIMSEC", sky_dim_str)
        # fix the exposure time!!
        exptime_true = float(old_hdr["EXPTIME"])
        sky_hdu.header.set("EXPTIME", exptime_true, comment="Total NS exposure time")
        sky_hdu.scale("int16")
        outfits_sky.append(sky_hdu)
        gc.collect()
    # ------------------------------------
    outfits_obj[0].header.set("PYWIFES", __version__, "PyWiFeS version")
    outfits_obj[0].header.set("PYWSMDEF", slitlet_def_file.split('/')[-1],
                              "PyWiFeS: slitlet MEF definition file")
    outfits_obj[0].header.set("PYWSMINT", nan_method,
                              "PyWiFeS: method for bad pixels at MEF creation")
    outfits_obj.writeto(outimg_obj, overwrite=True)
    outfits_sky[0].header.set("PYWIFES", __version__, "PyWiFeS version")
    outfits_sky[0].header.set("PYWSMDEF", slitlet_def_file.split('/')[-1],
                              "PyWiFeS: slitlet MEF definition file")
    outfits_sky[0].header.set("PYWSMINT", nan_method,
                              "PyWiFeS: method for bad pixels at MEF creation")
    outfits_sky.writeto(outimg_sky, overwrite=True)
    return


# ------------------------------------------------------------------------
def wifes_response_pixel(inimg, outimg, wsol_fn=None, debug=False):
    """
    Generate response array by dividing flatfield by the median along each axis.
    Operates separately on each slitlet.

    Parameters
    ----------
    inimg : str
        The file path to the input image.
    outimg : str
        The file path to the output image.
    wsol_fn : str, optional
        The file path to the wavelength solution. Default is None.
    debug : bool, optional
        Whether to report the parameters used in this function call.
        Default is False.

    Returns
    -------
    None.
    """
    if debug:
        print(arguments())
    # check if halfframe
    halfframe = is_halfframe(inimg)
    if halfframe:
        if is_taros(inimg):
            first = 1
            nslits = 12
        else:
            first = 7
            nslits = 13
    else:
        first = 1
        nslits = 25
    # now open and operate on data
    f = pyfits.open(inimg)
    outfits = pyfits.HDUList(f)
    for i in range(nslits):
        curr_hdu = i + 1

        orig_data = f[curr_hdu].data
        # rectify data!
        if wsol_fn is not None:
            f3 = pyfits.open(wsol_fn)
            wave = f3[curr_hdu].data
            f3.close()
            print("Transforming data for Slitlet %d" % (curr_hdu + first))
            rect_data = transform_data(orig_data, wave)
            curr_ff_rowwise_ave = numpy.nanmedian(rect_data, axis=0)
            curr_ff_illum = numpy.nanmedian(rect_data / curr_ff_rowwise_ave, axis=1)
            curr_model = (
                (numpy.ones(numpy.shape(rect_data)) * curr_ff_rowwise_ave)
                * curr_ff_illum.T[:, numpy.newaxis]
            )
            orig_model = detransform_data(curr_model, orig_data, wave)
            normed_data = orig_data / orig_model
        else:
            curr_ff_rowwise_ave = numpy.nanmedian(orig_data, axis=0)
            curr_ff_illum = numpy.nanmedian(orig_data / curr_ff_rowwise_ave, axis=1)
            curr_model = (
                (numpy.ones(numpy.shape(orig_data)) * curr_ff_rowwise_ave)
                * curr_ff_illum.T[:, numpy.newaxis]
            )
            normed_data = orig_data / curr_model
        outfits[curr_hdu].data = normed_data.astype("float32", casting="same_kind")
        outfits[curr_hdu].scale("float32")
        # need to fit this for each slitlet
    outfits[0].header.set("PYWIFES", __version__, "PyWiFeS version")
    outfits.writeto(outimg, overwrite=True)
    f.close()
    return


def wifes_response_poly(
    inimg,
    outimg,
    wsol_fn=None,
    zero_var=True,
    polydeg=7,
    shape_fn=None,
    plot=True,
    plot_dir='.',
    save_prefix="flat_response",
    debug=False,
):
    """
    Generate response array by fitting dome flatfield with polynomials in log(count)-space.
    Operates separately on each slitlet.

    Parameters
    ----------
    inimg : str
        The file path to the input image.
    outimg : str
        The file path to the output image.
    wsol_fn : str, optional
        The file path to the wavelength solution. Default is None.
    zero_var : bool, optional
        Whether to set the VAR extensions to zero in the output file. Default is True.
    polydeg : int, optional
        Degree of polynomial to fit. Default is 7.
    shape_fn : str, optional
        If defined, outputs shape of central slice's central pixel to aid later flux
        calibration to the given filename. Default is None.
    plot : bool, optional
        Whether to output a diagnostic plot. Default is True.
    plot_dir : str, optional
        Directory for output of plot (if requested). Default is '.'.
    save_prefix : str, optional
        Prefix for plot (if requested). Default is 'flat_response'.
    debug : bool, optional
        Whether to report the parameters used in this function call.
        Default is False.

    Returns
    -------
    None.
    """
    if debug:
        print(arguments())

    # check if halfframe
    halfframe = is_halfframe(inimg)
    if halfframe:
        if is_taros(inimg):
            first = 1
            nslits = 12
        else:
            first = 7
            nslits = 13
    else:
        first = 1
        nslits = 25

    mid_slit_idx = min(nslits, 13 - first + 1) - 1

    # now open and operate on data
    f = pyfits.open(inimg)
    outfits = pyfits.HDUList(f)
    # ------------------------------------
    # fit a smooth polynomial to the middle slice

    midslice_data = f[mid_slit_idx].data
    if wsol_fn is not None:
        f3 = pyfits.open(wsol_fn)
        wave = f3[mid_slit_idx].data
        f3.close()
        rect_data, lam_array = transform_data(midslice_data, wave, return_lambda=True)
    else:
        rect_data = midslice_data
        lam_array = numpy.arange(len(rect_data[0, :]), dtype="d")
    # fit polynomial to median data
    curr_ff_rowwise_ave = numpy.nanmedian(rect_data, axis=0)
    # Masking negatives values before entering the log10
    curr_ff_rowwise_ave[curr_ff_rowwise_ave <= 0] = numpy.nan
    curr_y = numpy.log10(curr_ff_rowwise_ave)
    good_inds = numpy.nonzero((numpy.isfinite(curr_y)) * (curr_ff_rowwise_ave > 0.0))[0]
    # add points far away to force edge derivatives to be preserved
    next_x = lam_array[good_inds][5:-5]
    next_y = curr_y[good_inds][5:-5]
    yderiv = (next_y[1:] - next_y[:-1]) / (next_x[1:] - next_x[:-1])
    nave_lo = 50
    nave_hi = 100
    nextend_lo = 150.0
    nextend_hi = 300.0
    ydave_lo = numpy.nanmean(yderiv[:nave_lo])
    ydave_hi = numpy.nanmean(yderiv[-nave_hi:])
    dxlo = numpy.arange(1, nextend_lo + 1, dtype="d") * (next_x[1] - next_x[0])
    new_xlo = next_x[0] - dxlo
    new_ylo = next_y[0] - dxlo * ydave_lo
    dxhi = numpy.arange(1, nextend_hi + 1, dtype="d") * (next_x[-1] - next_x[-2])
    new_xhi = next_x[-1] + dxhi
    new_yhi = next_y[-1] + dxhi * ydave_hi
    fit_x = numpy.concatenate((new_xlo, next_x, new_xhi))
    fit_y = numpy.concatenate((new_ylo, next_y, new_yhi))
    smooth_poly = numpy.polyfit(fit_x, fit_y, polydeg)
    # ------------------------------------
    # divide rectified data by the smooth polynomial
    for i in range(nslits):
        curr_hdu = i + 1
        orig_data = f[curr_hdu].data
        # rectify data!
        if wsol_fn is not None:
            f3 = pyfits.open(wsol_fn)
            wave = f3[curr_hdu].data
            f3.close()
            print("Transforming data for Slitlet %d" % (first + i))
            rect_data, lam_array = transform_data(orig_data, wave, return_lambda=True)
            curr_norm_array = 10.0 ** (numpy.polyval(smooth_poly, lam_array))
            curr_norm_array_noxform = detransform_data(numpy.tile(curr_norm_array, (orig_data.shape[1], 1)), orig_data, wave)
            init_normed_data = rect_data / curr_norm_array
            normed_data = detransform_data(init_normed_data, orig_data, wave)
            if i == mid_slit_idx:
                with open(shape_fn, 'w') as of:
                    for smooth_row in range(lam_array.shape[0]):
                        of.write(f"{lam_array[smooth_row]} {curr_norm_array[smooth_row]}\n")

        else:
            lam_array = numpy.arange(len(orig_data[0, :]), dtype="d")
            curr_norm_array_noxform = 10.0 ** (numpy.polyval(smooth_poly, lam_array))
            normed_data = rect_data / curr_norm_array_noxform

        # Diagnostic plot: save some values for later
        if plot and i == mid_slit_idx:
            mid_row = (orig_data.shape[0] // 2)
            x1 = numpy.arange(orig_data.shape[1])
            y1 = orig_data[mid_row, :]
            y3 = curr_norm_array_noxform[0, :]

        outfits[curr_hdu].data = normed_data.astype("float32", casting="same_kind")
        outfits[curr_hdu].scale("float32")
        if zero_var:
            var_hdu = curr_hdu + nslits
            outfits[var_hdu].data = (0.0 * outfits[var_hdu].data).astype("float64", casting="same_kind")
            outfits[var_hdu].scale("float64")
        # need to fit this for each slitlet
    outfits[0].header.set("PYWIFES", __version__, "PyWiFeS version")
    outfits[0].header.set("PYWRESIN", "dome only", "PyWiFeS: flatfield inputs")
    outfits[0].header.set("PYWRESZV", zero_var, "PyWiFeS: 2D response zero_var")
    outfits[0].header.set("PYWRESPD", polydeg, "PyWiFeS: 2D response polydeg")
    outfits.writeto(outimg, overwrite=True)
    f.close()

    # ---------------------------------------------
    # Diagnostic plots
    if plot:
        fig = plt.figure(1, figsize=(10, 5))
        grid = gridspec.GridSpec(2, 2, height_ratios=[3, 1], width_ratios=[4, 1])

        # (1) spectral fit: plot the middle spectrum of the middle slit
        ax_left = fig.add_subplot(grid[0, 0])
        ax_left.set_title('Spectral Flatfield Correction')
        ax_left.plot(x1, y1, "C0", label='Original flat lamp spectrum')
        ax_left.plot(x1, y3, color="r", ls='dashed', label='Smooth function fit')
        ax_left.legend()
        ax_left.set_xlabel(r'X-axis pixel')
        ax_left.set_ylabel('Flux')
        ax_left.set_ylim(0.8 * numpy.nanmin(y1), 1.2 * numpy.nanmax(y1))
        ax_left.set_yscale('log')

        # (2) spectral fit residuals
        ax_bottom = fig.add_subplot(grid[1, 0])
        ax_bottom.axhline(1, ls='--', c='k')
        ax_bottom.plot(x1, y1 / y3)
        ax_bottom.set_ylabel('Ratio')
        ax_bottom.set_ylim(0.85, 1.15)

        # (3) illumination correction
        ax_right = fig.add_subplot(grid[0, 1])
        ax_right.set_title('Illumination Correction')
        ax_right.text(0.5, 0.5, 'No Twilight Flats', horizontalalignment='center')

        plt.tight_layout()
        plot_name = f"{save_prefix}.png"
        plot_path = os.path.join(plot_dir, plot_name)
        plt.savefig(plot_path, dpi=300)
        plt.close()

    return


def wifes_2dim_response(
    spec_inimg,
    spatial_inimg,
    outimg,
    wsol_fn=None,
    zero_var=True,
    plot=True,
    plot_dir='.',
    save_prefix="flat_response",
    polydeg=7,
    resp_min=1.0e-4,
    debug=False,
):
    """
    Generate response array by fitting dome and twilight flatfields with polynomials in
    log(count)-space. Operates separately on each slitlet.

    Parameters
    ----------
    spec_inimg : str
        The file path to the input domeflat image.
    spatial_inimg : str
        The file path to the input twilight flat image.
    outimg : str
        The file path to the output image.
    wsol_fn : str, optional
        The file path to the wavelength solution. Default is None.
    zero_var : bool, optional
        Whether to set the VAR extensions to zero in the output file. Default is True.
    plot : bool, optional
        Whether to output a diagnostic plot. Default is True.
    plot_dir : str, optional
        Directory for output of plot (if requested). Default is '.'.
    save_prefix : str, optional
        Prefix for plot (if requested). Default is 'flat_response'.
    polydeg : int, optional
        Degree of polynomial to fit. Default is 7.
    resp_min : float, optional
        Minimum response value in output file. Default is 1.0e-4.
    debug : bool, optional
        Whether to report the parameters used in this function call.
        Default is False.

    Returns
    -------
    None.
    """
    if debug:
        print(arguments())
    # check if halfframe
    halfframe = is_halfframe(spec_inimg)
    if halfframe:
        if is_taros(spec_inimg):
            nslits = 12
            first = 1
        else:
            nslits = 13
            first = 7
    else:
        nslits = 25
        first = 1

    mid_slit_idx = min(nslits, 13 - first + 1) - 1

    # open the two files!
    f1 = pyfits.open(spec_inimg)
    f2 = pyfits.open(spatial_inimg)
    ndy, ndx = numpy.shape(f1[1].data)
    xarr = numpy.arange(ndx)
    yarr = numpy.arange(ndy)

    try:
        bin_x, bin_y = [int(b) for b in f1[1].header["CCDSUM"].split()]
    except Exception:
        bin_x = 1
        bin_y = 1

    full_x, full_y = numpy.meshgrid(xarr, yarr)
    outfits = pyfits.HDUList(f1)
    # ------------------------------------
    # SPECTRAL FLAT
    # fit a smooth polynomial to the middle slice
    # ------------------------------------

    midslice_data = f1[mid_slit_idx].data

    if wsol_fn is not None:
        f3 = pyfits.open(wsol_fn)
        wave = f3[mid_slit_idx].data
        f3.close()

        rect_data, mid_lam_array = transform_data(
            midslice_data, wave, return_lambda=True
        )
    else:
        rect_data = midslice_data
        mid_lam_array = numpy.arange(len(rect_data[0, :]), dtype="d")

    out_y_full = numpy.meshgrid(yarr, mid_lam_array)[0]
    disp_ave = abs(numpy.nanmean(mid_lam_array[1:] - mid_lam_array[:-1]))

    # fit polynomial to median data
    curr_ff_rowwise_ave = numpy.nanmedian(rect_data, axis=0)
    # Masking negatives values before entering the log10
    curr_ff_rowwise_ave[curr_ff_rowwise_ave <= 0] = numpy.nan
    curr_y = numpy.log10(curr_ff_rowwise_ave)
    good_inds = numpy.nonzero((numpy.isfinite(curr_y)) * (curr_ff_rowwise_ave > 0.0))[0]
    # add points far away to force edge derivatives to be preserved
    if pyfits.getval(spec_inimg, "CAMERA") == "WiFeSRed":
        next_x = mid_lam_array[good_inds][500 // bin_x:-10 // bin_x]
        next_y = curr_y[good_inds][500 // bin_x:-10 // bin_x]
    else:
        next_x = mid_lam_array[good_inds][50 // bin_x:-100 // bin_x]
        next_y = curr_y[good_inds][50 // bin_x:-100 // bin_x]

    yderiv = (next_y[1:] - next_y[:-1]) / (next_x[1:] - next_x[:-1])
    nave_lo = 50 // bin_x
    nave_hi = 100 // bin_x
    nextend_lo = 150 // bin_x
    nextend_hi = 300 // bin_x
    ydave_lo = numpy.nanmean(yderiv[:nave_lo])
    ydave_hi = numpy.nanmean(yderiv[-nave_hi:])

    dxlo = numpy.arange(1, nextend_lo + 1, dtype="d") * (next_x[1] - next_x[0])
    new_xlo = next_x[0] - dxlo
    new_ylo = next_y[0] - dxlo * ydave_lo
    dxhi = numpy.arange(1, nextend_hi + 1, dtype="d") * (next_x[-1] - next_x[-2])
    new_xhi = next_x[-1] + dxhi
    new_yhi = next_y[-1] + dxhi * ydave_hi

    fit_x = numpy.concatenate((new_xlo, next_x, new_xhi))

    fit_y = numpy.concatenate((new_ylo, next_y, new_yhi))

    smooth_poly = numpy.polyfit(fit_x, fit_y, polydeg)

    # ------------------------------------
    # SPATIAL FLAT
    # get median spatial flat spectrum
    # ------------------------------------

    midslice_data = f2[mid_slit_idx].data

    if wsol_fn is not None:
        f3 = pyfits.open(wsol_fn)
        wave = f3[mid_slit_idx].data
        f3.close()
        rect_data = transform_data(midslice_data, wave)
    else:
        rect_data = midslice_data
    # fit polynomial to median data
    spatial_flat_spec = numpy.nanmedian(rect_data, axis=0)
    spec_norm = curr_ff_rowwise_ave / (
        10.0 ** (numpy.polyval(smooth_poly, mid_lam_array))
    )

    spat_interp = interp.interp1d(
        mid_lam_array, spatial_flat_spec / spec_norm, bounds_error=False, fill_value=0.0
    )

    # ------------------------------------
    # ------------------------------------
    illum = numpy.zeros([ndy, nslits], dtype="d")
    # divide rectified data by the smooth polynomial
    for i in range(nslits):
        curr_hdu = i + 1
        orig_spec_data = f1[curr_hdu].data
        orig_spat_data = f2[curr_hdu].data

        # rectify data!
        if wsol_fn is not None:
            f3 = pyfits.open(wsol_fn)
            wave = f3[curr_hdu].data
            f3.close()

            # convert the *x* pixels to lambda
            dw = numpy.abs(wave[:, 1:] - wave[:, :-1])
            full_dw = numpy.zeros(numpy.shape(wave))
            full_dw[:, 1:] = dw
            full_dw[:, 0] = dw[:, 0]
            print("Transforming data for Slitlet %d" % (i + first))

            # SPECTRAL FLAT
            rect_spec_data, lam_array = transform_data(
                orig_spec_data, wave, return_lambda=True
            )

            curr_norm_array = 10.0 ** (numpy.polyval(smooth_poly, lam_array))
            init_normed_data = rect_spec_data / curr_norm_array
            xstart = int(0.25 * len(lam_array))
            xstop = int(0.75 * len(lam_array))
            norm_region = init_normed_data[:, xstart:xstop]
            next_normed_data = (
                init_normed_data / numpy.nanmedian(norm_region, axis=1).T[:, numpy.newaxis]
            )
            next_normed_data[next_normed_data <= 0] = numpy.nan

            # Force spectral flat to 1 at wavelengths where sum of median counts < 100 (S/N ~ 10)
            try:
                nflat = float(f1[0].header["PYWFLATN"])
            except Exception:
                print("Could not retrieve number of input dome flats from header, defaulting to 1")
                nflat = 1.0
            if 'PYWICOFF' in f1[0].header:
                ic_off = float(f1[0].header['PYWICOFF'])
            else:
                ic_off = 0.0
            force_idx = numpy.nonzero(nflat * numpy.nanmedian(rect_spec_data, axis=0) < 100.0 - ic_off)
            next_normed_data[:, force_idx] = 1.

            # SPATIAL FLAT
            rect_spat_data = transform_data(orig_spat_data, wave, return_lambda=False)
            # alt_dw = lam_array[1] - lam_array[0]
            alt_flat_spec = spat_interp(lam_array)
            alt_flat_spec[alt_flat_spec <= 0] = numpy.nan
            curr_interp_spat = numpy.zeros(numpy.shape(out_y_full), dtype="d")
            alt_interp_spat = numpy.zeros(numpy.shape(rect_spat_data), dtype="d")

            for q in range(ndy):
                curr_x = wave[q, :]
                curr_y = orig_spat_data[q, :] / full_dw[q, :]
                sort_order = curr_x.argsort()
                curr_interp = interp.interp1d(
                    curr_x[sort_order],
                    curr_y[sort_order],
                    bounds_error=False,
                    fill_value=0.0,
                )
                new_data = disp_ave * curr_interp(mid_lam_array)
                curr_interp_spat[:, q] = new_data
                # norm_func = (new_data / spatial_flat_spec)
                alt_y = rect_spat_data[q, :]
                norm_func = alt_y / (alt_flat_spec * next_normed_data[q, :])
                alt_interp_spat[q, :] = norm_func
                # norm_func = (alt_y / (
                #    alt_flat_spec))
            # spat_ratio = curr_interp_spat.T / spatial_flat_spec
            spat_ratio = alt_interp_spat
            spat_ratio[numpy.nonzero(spat_ratio != spat_ratio)] = 0.0
            spat_ratio[numpy.nonzero(spat_ratio < 0.0)] = 0.0
            spat_flat = numpy.nanmedian(spat_ratio[:, xstart:xstop], axis=1)
            # transform back
            final_normed_data = next_normed_data * spat_flat.T[:, numpy.newaxis]
            normed_data = detransform_data(final_normed_data, orig_spec_data, wave)
            normed_data[numpy.nonzero(normed_data < resp_min)] = resp_min
        else:
            lam_array = numpy.arange(len(orig_spec_data[0, :]), dtype="d")
            curr_norm_array = 10.0 ** (numpy.polyval(smooth_poly, lam_array))
            normed_data = orig_spec_data / curr_norm_array
            # Force spectral flat to 1 at wavelengths where sum of median counts < 100 (S/N ~ 10)
            try:
                nflat = float(f1[0].header["PYWFLATN"])
            except Exception:
                print("Could not retrieve number of input dome flats from header, defaulting to 1")
                nflat = 1.0
            if 'PYWICOFF' in f1[0].header:
                ic_off = float(f1[0].header['PYWICOFF'])
            else:
                ic_off = 0.0
            force_idx = numpy.nonzero(nflat * numpy.nanmedian(rect_spec_data, axis=0) < 100.0 - ic_off)
            normed_data[:, force_idx] = 1.

            normed_data[numpy.nonzero(normed_data < resp_min)] = resp_min
        outfits[curr_hdu].data = normed_data.astype("float32", casting="same_kind")
        outfits[curr_hdu].scale("float32")
        illum[:, i] = numpy.nanmedian(normed_data, axis=1)
        if zero_var:
            var_hdu = curr_hdu + nslits
            outfits[var_hdu].data = (0.0 * outfits[var_hdu].data).astype("float64", casting="same_kind")
            outfits[var_hdu].scale("float64")
        # need to fit this for each slitlet

    outfits[0].header.set("PYWIFES", __version__, "PyWiFeS version")
    outfits[0].header.set("PYWRESIN", "dome+twi", "PyWiFeS: flatfield inputs")
    outfits[0].header.set("PYWRESZV", zero_var, "PyWiFeS: 2D response zero_var")
    outfits[0].header.set("PYWRESPD", polydeg, "PyWiFeS: 2D response polydeg")
    outfits[0].header.set("PYWRESMN", resp_min, "PyWiFeS: 2D response minimum response")
    outfits.writeto(outimg, overwrite=True)
    f1.close()
    f2.close()
    # ---------------------------------------------
    # diagnostic plots!
    if plot:
        # Response plots

        fig = plt.figure(figsize=(10, 5))
        grid = fig.add_gridspec(1, 3)

        # (1) spectral fit
        ax_left = fig.add_subplot(grid[0, 0:2])
        ax_left.set_title('Spectral Flatfield Correction')
        ax_left.plot(mid_lam_array, curr_ff_rowwise_ave, "C0", label='flat lamp spectrum')
        ax_left.plot(
            mid_lam_array,
            10.0 ** (numpy.polyval(smooth_poly, mid_lam_array)),
            color="r",
            ls='dashed',
            label='smooth function fit ',
        )

        ax_left.legend()
        ax_left.set_xlabel(r'Wavelength [$\AA$]')
        ax_left.set_ylabel('Flux')
        ax_left.set_yscale('log')

        # (2) illumination correction
        ax_right = fig.add_subplot(grid[0, 2])
        ax_right.set_title('Illumination Correction\n(not wire-aligned)')
        pos = ax_right.imshow(
            illum, interpolation="nearest", origin="lower",
            cmap=cm['gist_rainbow'], aspect=(0.5 * bin_y),
            extent=[first - 0.5, first + nslits - 0.5, -0.5, illum.shape[0] - 0.5]
        )
        fig.colorbar(pos, ax=ax_right, shrink=0.9)
        ax_right.set_xlabel('Slitlet')
        ax_right.set_ylabel('Detector Y')

        plt.tight_layout()
        plot_name = f"{save_prefix}.png"
        plot_path = os.path.join(plot_dir, plot_name)
        plt.savefig(plot_path, dpi=300)
        plt.close()

    # ---------------------------------------------
    return


def wifes_SG_response(
    spec_inimg,
    outimg,
    spatial_inimg=None,
    wsol_fn=None,
    zero_var=True,
    plot=True,
    plot_dir='.',
    save_prefix="flat_response",
    resp_min=1.0e-4,
    shape_fn=None,
    debug=False,
    interactive_plot=False,
):
    """
    Generate response array by fitting dome and twilight flatfields with Savitzky-Golay filter in
    log(count)-space. Operates separately on each slitlet.

    Parameters
    ----------
    spec_inimg : str
        The file path to the input domeflat image.
    spatial_inimg : str
        The file path to the input twilight flat image.
    outimg : str
        The file path to the output image.
    wsol_fn : str, optional
        The file path to the wavelength solution. Default is None.
    zero_var : bool, optional
        Whether to set the VAR extensions to zero in the output file. Default is True.
    plot : bool, optional
        Whether to output a diagnostic plot. Default is True.
    plot_dir : str, optional
        Directory for output of plot (if requested). Default is '.'.
    save_prefix : str, optional
        Prefix for plot (if requested). Default is 'flat_response'.
    resp_min : float, optional
        Minimum response value in output file. Default is 1.0e-4.
    shape_fn : str, optional
        If defined, outputs shape of central slice's central pixel to aid later flux
        calibration to the given filename. Default is None.
    debug : bool, optional
        Whether to report the parameters used in this function call. Default is False.
    interactive_plot : bool, optional
        Whether to interrupt processing to provide interactive plot to user. Default is False.

    Returns
    -------
    None.
    """
    if debug:
        print(arguments())

    # check if halfframe
    halfframe = is_halfframe(spec_inimg)
    if halfframe:
        if is_taros(spec_inimg):
            nslits = 12
            first = 1
        else:
            nslits = 13
            first = 7
    else:
        nslits = 25
        first = 1

    mid_slit_idx = min(nslits, 13 - first + 1) - 1

    # open the two files!
    f1 = pyfits.open(spec_inimg)
    if spatial_inimg is not None:
        f2 = pyfits.open(spatial_inimg)
    ndy, ndx = numpy.shape(f1[1].data)
    arm = f1[0].header["CAMERA"]
    grating = f1[0].header["GRATINGB"] if arm == "WiFeSBlue" else f1[0].header["GRATINGR"]

    try:
        bin_x, bin_y = [int(b) for b in f1[0].header["CCDSUM"].split()]
    except Exception:
        bin_x = 1
        bin_y = 1

    try:
        nflat = float(f1[0].header["PYWFLATN"])
    except Exception:
        print("Could not retrieve number of input dome flats from header, defaulting to 1")
        nflat = 1.0
    if 'PYWICOFF' in f1[0].header:
        ic_off = float(f1[0].header['PYWICOFF'])
    else:
        ic_off = 0.0

    if spatial_inimg is not None:
        try:
            ntflat = float(f2[0].header["PYWTWIN"])
        except Exception:
            print("Could not retrieve number of input twi flats from header, defaulting to 1")
            ntflat = 1.0

    outfits = pyfits.HDUList(f1)
    pixel_response = numpy.ones((nslits, ndy, ndx))

    # ------------------------------------
    illum = numpy.ones((nslits, ndy), dtype="d")
    for i in range(nslits):
        print("Computing slitlet %d" % (i + first))
        curr_hdu = i + 1
        orig_spec_data = f1[curr_hdu].data
        if spatial_inimg is not None:
            orig_spat_data = f2[curr_hdu].data

        N = 25
        window_factor = 4 if arm == "WiFeSRed" else 3
        if "7000" in grating:
            N = 60
        for row in range(orig_spec_data.shape[0]):
            this_row = orig_spec_data[row, :]
            this_x = numpy.arange(orig_spec_data.shape[1])
            this_y = numpy.log10(numpy.interp(this_x, this_x[this_row > 0],
                                              this_row[this_row > 0]))
            intermed0 = signal.savgol_filter(this_y, window_length=N, polyorder=3, mode='nearest')
            # filter out large outliers
            keep_idx = numpy.nonzero(numpy.abs(this_y - intermed0) / intermed0 < 0.2)[0]
            this_y = numpy.log10(numpy.interp(this_x, this_x[keep_idx],
                                              this_row[keep_idx]))
            intermed1 = signal.savgol_filter(this_y, window_length=N, polyorder=3, mode='nearest')
            intermed2 = signal.savgol_filter(intermed1, window_length=(window_factor * N), polyorder=3, mode='nearest')

            pixel_response[i, row, :] = this_row / numpy.power(10, intermed2)

            if row == (orig_spec_data.shape[0] // 2):
                # Interactive plot: show the middle spectrum of each slit
                if interactive_plot:
                    fig = plt.figure(figsize=(10, 6))
                    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
                    ax1 = fig.add_subplot(gs[0, 0])
                    ax1.plot(this_x, this_row, "C0", label='Original flat lamp spectrum')
                    ax1.plot(this_x, numpy.power(10, this_y), c="g", label='Data being fit')
                    ax1.plot(this_x, numpy.power(10, intermed2), color="r", ls='dashed', label='Smooth function fit')
                    ax1.legend()
                    ax1.set_ylabel('Flux')
                    ax1.set_title(f"Slitlet {i+first}")
                    ax1.set_ylim(0.8 * numpy.nanmin(this_row), 1.2 * numpy.nanmax(this_row))
                    ax1.set_yscale('log')

                    ax2 = fig.add_subplot(gs[1, 0])
                    this_pixel_response = pixel_response[i, row, :]
                    this_pixel_response[nflat * this_row < 100.] = 1.
                    ax2.axhline(1, ls='--', c='k')
                    ax2.plot(this_x, this_pixel_response)
                    ax2.set_xlabel(r'X-axis pixel')
                    ax2.set_ylabel('Ratio')
                    ax2.set_ylim(0.85, 1.15)
                    plt.show()

                # Diagnostic plot: save some values for later
                if plot and i == mid_slit_idx:
                    x1 = this_x
                    y1 = this_row
                    y2 = numpy.power(10, this_y)
                    y3 = numpy.power(10, intermed2)

        # Force pixel_response to 1 at wavelengths where sum of median counts < 100 (S/N ~ 10)
        pixel_response[i][nflat * orig_spec_data < 100. - ic_off] = 1.

        # rectify twilight data to take consistent wavelength regions
        if wsol_fn is not None:
            f3 = pyfits.open(wsol_fn)
            wave = f3[curr_hdu].data
            f3.close()

            # SPATIAL FLAT
            if spatial_inimg is not None:
                rect_spat_data, lam_array = transform_data(orig_spat_data, wave, return_lambda=True)
                # define limits in untransformed coordinates
                lam_min = numpy.amin((wave[:, 1500 // bin_x], wave[:, 2000 // bin_x]), axis=0)
                lam_max = numpy.amax((wave[:, 1500 // bin_x], wave[:, 2000 // bin_x]), axis=0)
                for row in range(rect_spat_data.shape[0]):
                    illum[i, row] = numpy.median(rect_spat_data[row, :][(lam_array >= lam_min[row]) * (lam_array <= lam_max[row])])

            # Save the smooth fit to the middle row of the middle slice of the spectral flat
            # Do this here to avoid duplicating retrieval of wavelength info
            if i == mid_slit_idx:
                smooth_wave = wave[(orig_spec_data.shape[0] // 2), :]
                y3max = numpy.nanmax(y3)
                y3norm = numpy.log10(y3 / y3max)
                y3n_interp = interp.interp1d(smooth_wave[~numpy.isnan(y3norm)],
                                             y3norm[~numpy.isnan(y3norm)],
                                             kind='linear',
                                             bounds_error=False,
                                             fill_value="extrapolate")
                y3out = numpy.power(10, y3n_interp(smooth_wave))
                with open(shape_fn, 'w') as of:
                    for smooth_pix in range(smooth_wave.shape[0]):
                        of.write(f"{smooth_wave[smooth_pix]} {y3out[smooth_pix]}\n")

        else:
            if spatial_inimg is not None:
                illum[i, :] = numpy.median(orig_spat_data[:, 1500 // bin_x:2000 // bin_x], axis=0)

    # Normalise spatial flat to a peak of 1
    if spatial_inimg is not None:
        illum /= numpy.amax(illum)
    pixel_response = numpy.clip(illum[:, :, numpy.newaxis] * pixel_response, a_min=resp_min, a_max=None)

    for i in range(nslits):
        curr_hdu = i + 1
        print(f"Writing slitlet {i + first}")
        outfits[curr_hdu].data = pixel_response[i].astype("float32", casting="same_kind")
        outfits[curr_hdu].scale("float32")
        if zero_var:
            var_hdu = curr_hdu + nslits
            outfits[var_hdu].data = (0.0 * outfits[var_hdu].data).astype("float64", casting="same_kind")
            outfits[var_hdu].scale("float64")

    outfits[0].header.set("PYWIFES", __version__, "PyWiFeS version")
    if spatial_inimg is not None:
        outfits[0].header.set("PYWRESIN", "dome+twi", "PyWiFeS: flatfield inputs")
    else:
        outfits[0].header.set("PYWRESIN", "dome", "PyWiFeS: flatfield inputs")
    if spatial_inimg is not None:
        outfits[0].header.set("PYWTWIN", ntflat, "PyWiFeS: number of twilight flats combined")
    outfits[0].header.set("PYWRESZV", zero_var, "PyWiFeS: 2D response zero_var")
    outfits[0].header.set("PYWRESW1", N, "PyWiFeS: 1st Savitzky-Golay window size")
    outfits[0].header.set("PYWRESW2", N * window_factor, "PyWiFeS: 2nd Savitzky-Golay window size")
    outfits[0].header.set("PYWRESMN", resp_min, "PyWiFeS: 2D response minimum response")
    outfits.writeto(outimg, overwrite=True)
    f1.close()
    if spatial_inimg is not None:
        f2.close()

    # ---------------------------------------------
    # Diagnostic plots
    if plot:
        fig = plt.figure(1, figsize=(10, 5))
        grid = gridspec.GridSpec(2, 2, height_ratios=[3, 1], width_ratios=[4, 1])

        # (1) spectral fit: plot the middle spectrum of the middle slit
        ax_left = fig.add_subplot(grid[0, 0])
        ax_left.set_title('Spectral Flatfield Correction')
        ax_left.plot(x1, y1, "C0", label='Original flat lamp spectrum')
        ax_left.plot(x1, y2, c="g", label='Data being fit')
        ax_left.plot(x1, y3, color="r", ls='dashed', label='Smooth function fit')
        ax_left.legend()
        ax_left.set_xlabel(r'X-axis pixel')
        ax_left.set_ylabel('Flux')
        ymax = 1.2 * numpy.nanmax(y1)
        ymin = numpy.nanmax([1, 0.8 * numpy.nanmin(y1)]) if ymax > 100 else numpy.nanmax([0.01, 0.8 * numpy.nanmin(y1)])
        ax_left.set_ylim(ymin, ymax)
        ax_left.set_yscale('log')

        # (2) spectral fit residuals
        ax_bottom = fig.add_subplot(grid[1, 0])
        ax_bottom.axhline(1, ls='--', c='k')
        ax_bottom.plot(x1, y1 / y3)
        ax_bottom.set_ylabel('Ratio')
        ax_bottom.set_ylim(0.85, 1.15)

        # (3) illumination correction
        ax_right = fig.add_subplot(grid[0, 1])
        if spatial_inimg is not None:
            ax_right.set_title('Illumination Correction\n(not wire-aligned)')
            pos = ax_right.imshow(
                illum.T, interpolation="nearest", origin="lower",
                cmap=cm['gist_rainbow'], aspect=(0.5 * bin_y),
                extent=[first - 0.5, first + nslits - 0.5, -0.5, illum.shape[1] - 0.5]
            )
            fig.colorbar(pos, ax=ax_right, shrink=0.9)
            ax_right.set_xlabel('Slitlet')
            ax_right.set_ylabel('Detector Y')
        else:
            ax_right.set_title('Illumination Correction')
            ax_right.text(0.5, 0.5, 'No Twilight Flats', horizontalalignment='center')

        plt.tight_layout()
        plot_name = f"{save_prefix}.png"
        plot_path = os.path.join(plot_dir, plot_name)
        plt.savefig(plot_path, dpi=300)
        plt.close()

    return


# ------------------------------------------------------------------------
# function to fit the wire solution!
def derive_wifes_wire_solution(
    inimg,
    out_file,
    bin_x=None,
    bin_y=None,
    fit_zones=[16, 26, 54, 70],
    flux_threshold=0.001,
    wire_polydeg=1,
    xlims="default",
    interactive_plot=False,
    plot=True,
    plot_dir='.',
    save_prefix='wire_fit_params',
    debug=False,
):
    """
    Trace the wire image to determine position of centre of each slitlet.
    
    Parameters
    ----------
    inimg : str
        Path to the input wire image.
    out_file : str
        Path to the output wire solution.
    bin_x : int, optional
        Binning factor in the x-direction. If not specified, it will be read from the header.
    bin_y : int, optional
        Binning factor in the y-direction. If not specified, it will be read from the header.
    fit_zones : list, optional
        List of per-slitlet y-axis region limits to define background against which to locate
        wire. Unbinned pixels. Default is [16, 26, 54, 70].
    flux_threshold : float, optional
        Minimum flux difference for wire relative to background. Default value assumes a
        superwire scaled by "percentileN". Default is 0.001.
    wire_polydeg : int, optional
        Degree of polynomial in x-direction to fit. Default is 1.
    xlims : str or list, optional
        Either "default" or a 2-element list defining the x-axis range for the wire fitting.
        Default is "default".
    interactive_plot : bool, optional
        Whether to interrupt processing to provide interactive plot to user. Default is False.
    plot : bool, optional
        Whether to output a diagnostic plot. Default is True.
    plot_dir : str, optional
        Directory for output of plot (if requested). Default is '.'.
    save_prefix : str, optional
        Prefix for plot (if requested). Default is 'wire_fit_params'.
    debug : bool, optional
        Whether to report the parameters used in this function call. Default is False.

    Returns
    -------
    None
    """
    if debug:
        print(arguments())
    # note: later have these parameters as user-input kwargs
    # SOME PARAMTERS FOR THE FITTING
    init_nave = 100  # number of unbinned x-pixels to average together
    bg_polydeg = 3
    xlim_defaults = {
        "U7000": [1, 2500],
        "B7000": [1, 4096],
        "R7000": [1000, 3000],
        "I7000": [1, 4096],
        "B3000": [100, 3300],
        "R3000": [500, 3700],
    }
    # define region for fitting light profile
    init_fit_pmin_1 = fit_zones[0]
    init_fit_pmax_1 = fit_zones[1]
    init_fit_pmin_2 = fit_zones[2]
    init_fit_pmax_2 = fit_zones[3]
    # ------------------------------------
    f = pyfits.open(inimg)

    # check if it is halfframe
    halfframe = is_halfframe(inimg)
    if halfframe:
        if is_taros(inimg):
            first = 1
            nslits = 12
        else:
            first = 7
            nslits = 13
    else:
        first = 1
        nslits = 25
    # figure out which channel it is
    if f[1].header["CAMERA"] == "WiFeSRed":
        grating = f[1].header["GRATINGR"]
    else:
        grating = f[1].header["GRATINGB"]
    # figure out the binning!
    try:
        default_bin_x, default_bin_y = [int(b) for b in f[1].header["CCDSUM"].split()]
    except Exception:
        default_bin_x = 1
        default_bin_y = 1
    if bin_x is None:
        bin_x = default_bin_x
    if bin_y is None:
        bin_y = default_bin_y
    # get the xmax for fitting if default selected
    if xlims == "default":
        xmin = (xlim_defaults[grating][0] - 1) // bin_x
        xmax = (xlim_defaults[grating][1] - 1) // bin_x
    else:
        xmin = xlims[0] // bin_x
        xmax = xlims[1] // bin_x
    # adjust the fit regions accordingly
    nave = init_nave // bin_x
    fit_pmin_1 = init_fit_pmin_1 // bin_y
    fit_pmax_1 = init_fit_pmax_1 // bin_y
    fit_pmin_2 = init_fit_pmin_2 // bin_y
    fit_pmax_2 = init_fit_pmax_2 // bin_y
    # ------------------------------------
    # get data size and set up output data array
    temp_data = f[1].data
    nx = temp_data.shape[1]
    ctr_results = numpy.zeros([nslits, nx], dtype="f")
    ccd_x = numpy.arange(nx, dtype="f")
    # number of groupings to do
    ng = nx // nave - 1

    nwire = f[1].header.get("PYWWIREN", default="Unknown")

    if plot:
        wparam = []
    for q in range(nslits):
        slit_ind = q + 1

        test_data = f[slit_ind].data
        nr = test_data.shape[0]
        fit_inds = numpy.nonzero(
            ((numpy.arange(nr) >= fit_pmin_1) * (numpy.arange(nr) <= fit_pmax_1))
            + ((numpy.arange(nr) >= fit_pmin_2) * (numpy.arange(nr) <= fit_pmax_2))
        )[0]
        x_full = numpy.arange(nr, dtype="d")
        x_fit = x_full[fit_inds]
        frame_fit_x = []
        frame_fit_y = []
        for i in range(ng):
            # get median profile
            yprof = numpy.nanmedian(
                test_data[
                    :,
                    max([0, nave * i - nave // 2]):min(
                        [nave * i + nave // 2, test_data.shape[1]]
                    ) + 1,
                ],
                axis=1,
            )
            # if there is no usable data, skip this group
            if numpy.nanmax(yprof) <= flux_threshold:
                continue
            # otherwise fit the region outside of ~[30:50]
            else:
                curr_fit = numpy.polyfit(
                    x_fit, numpy.log10(yprof[fit_inds]), bg_polydeg
                )
                curr_fvals = 10.0 ** (numpy.polyval(curr_fit, x_full))
                # now take residuals and calculate a simple centroid
                wire_x = numpy.arange(fit_pmin_1, fit_pmax_2)
                wire_y = (curr_fvals - yprof)[fit_pmin_1:fit_pmax_2]
                wire_ctr = single_centroid_prof_fit(wire_y, x=wire_x)
            frame_fit_x.append(float(nave * i))
            frame_fit_y.append(wire_ctr)
        fit_x_arr = numpy.array(frame_fit_x)
        fit_y_arr = numpy.array(frame_fit_y)
        good_inds = numpy.nonzero(
            (numpy.isfinite(fit_x_arr))
            * (fit_y_arr > 0)
            * (fit_x_arr >= xmin)
            * (fit_x_arr <= xmax)
        )[0]
        wire_trend = numpy.polyfit(
            fit_x_arr[good_inds], fit_y_arr[good_inds], wire_polydeg, cov=True
        )
        if plot:
            # Store slope and error
            wparam.append([wire_trend[0], numpy.sqrt(numpy.diag(wire_trend[1])[-2])])

        trend_y = numpy.polyval(wire_trend[0], ccd_x)
        # Plot wire solution
        if interactive_plot:
            plt.scatter(fit_x_arr, fit_y_arr, c='b', label='Profile median centroids')
            plt.scatter(fit_x_arr[good_inds], fit_y_arr[good_inds], c='green', label='Used in fit')
            plt.plot(ccd_x, trend_y, c='r', label="Fit")
            plt.xlabel("x pixel")
            plt.ylabel("Wire centroid y pixel")
            plt.title(f"Slitlet {slit_ind}")
            midpt = numpy.median(fit_y_arr)
            plt.ylim(midpt - 4, midpt + 4)
            plt.legend()
            plt.show()

        ctr_results[q, :] = trend_y
    f.close()
    if plot:
        for i, slice in enumerate(range(first, first + nslits)):
            if wire_polydeg == 1:
                plt.errorbar(slice, wparam[i][0][0] * 1000., yerr=(1000. * wparam[i][1]), marker='o')
            elif wire_polydeg >= 2:
                plt.scatter(slice, wparam[i][0][-3], marker='x')
                plt.errorbar(slice, wparam[i][0][-2] * 1000., yerr=(1000. * wparam[i][1]), marker='o')
                if wire_polydeg > 2:
                    plt.scatter([], [], label="Higher-order terms\nfit but not plotted")
        plt.xlabel("Slitlet")
        if wire_polydeg == 1:
            plt.ylabel("Slope (y pixels per 1000 x pixels)")
        else:
            plt.ylabel("Slope (circle), 2nd-order coeff (X)")
        plt.title(f"Wire Fit Results (order {wire_polydeg} polynomial)")
        plt.tight_layout()
        plot_name = f"{save_prefix}.png"
        plot_path = os.path.join(plot_dir, plot_name)
        plt.savefig(plot_path, dpi=300)
        plt.close()
    results = pyfits.PrimaryHDU(data=ctr_results)
    g = pyfits.HDUList([results])
    g[0].header.set("PYWIFES", __version__, "PyWiFeS version")
    g[0].header.set("PYWWIREN", nwire, "PyWiFeS: number wire images combined")
    g[0].header.set("PYWWIFZ", ", ".join(str(fz) for fz in fit_zones), "PyWiFeS: wire fit zones")
    g[0].header.set("PYWWIFT", flux_threshold, "PyWiFeS: wire flux_threshold")
    g[0].header.set("PYWWIWPD", wire_polydeg, "PyWiFeS: wire_polydeg")
    g.writeto(out_file, overwrite=True)
    return


def _scale_grid_data(
    points, values, xi, method="linear", fill_value=0, scale_factor=1.0
):
    return (
        numpy.abs(scale_factor)
        * interp.griddata(
            points, values, xi, method=method, fill_value=fill_value
        ).T
    )


# -------------------------------------------------------------
# DATA CUBE!!!
def generate_wifes_cube(
    inimg,
    outimg,
    wire_fn,
    wsol_fn,
    wmin_set=None,
    wmax_set=None,
    dw_set=None,
    wavelength_ref="AIR",
    wave_native=False,
    bin_x=None,
    bin_y=None,
    ny_orig=76,
    offset_orig=2,
    verbose=True,
    adr=False,
    subsample=1,
    multithread=False,
    max_processes=-1,
    print_progress=False,
    debug=False,
):
    """
    Prepare data cube for assembly by unifying wavelength solution and slitlet centering
    (and optionally resampling flux to correct atmospheric differential refraction). The
    output cube remains as a multi-extension FITS file with one extension per slitlet.

    Parameters
    ----------
    inimg : str
        Path to the input image.
    outimg : str
        Path to the output image.
    wire_fn : str
        Path to the wire solution.
    wsol_fn : str
        Path to the wavelength solution.
    wmin_set : float, optional
        Minimum wavelength of output cube. Reverts to minimum in data if wmin_set is smaller or None. Default is None.
    wmax_set : float, optional
        Maximum wavelength of output cube. Reverts to maximum in data if wmax_set is larger or None. Default is None.
    dw_set : float, optional
        Wavelength step in output cube. Reverts to mean of per-slitlet mean wavelength steps if dw_set is None. Default is None.
    wavelength_ref : str, optional
        Wavelength reference for output cube. Options are "AIR", "VACUUM". Default is "AIR".
    wave_native : bool, optional
        Override wmin_set, wmax_set, dw_set and use native dispersion of central slitlet.
        N.B.: Prevents clean splicing of blue/red cubes and spectra. Default is False.
    bin_x : int, optional
        Binning factor in the x-direction. If not specified, it will be read from the header.
    bin_y : int, optional
        Binning factor in the y-direction. If not specified, it will be read from the header.
    ny_orig : int, optional
        Number of unbinned y-axis pixels in each slitlet. Default is 76.
    offset_orig : int, optional
        Number of (unbinned) y-axis pixels that the wire is offset from the field centre.
        Default is 2.
    verbose : bool, optional
        Whether to report additional diagnostic info. Default is True.
    adr : bool, optional
        Apply atmospheric differential refraction correction. Default is False.
    subsample : int, optional
        Divide each spatial dimension into "subsample" components. Minimises effects of
        intergerisation of pixel shifts, but increases processing time by subsample^2. Default is 1.
    multithread : bool, optional
        Run step using "multiprocessing" module. Default is False.
    max_processes : int, optional
        Number of simultaneous processes allowed. Non-positive values default to os.cpu_count().
        Default is -1.
    print_progress : bool, optional
        Write interpolation progress to screen/log. Default is False.
    debug : bool, optional
        Whether to report the parameters used in this function call. Default is False.

    Returns
    -------
    None
    """
    if debug:
        print(arguments())
    # ---------------------------
    # check if halfframe
    halfframe = is_halfframe(inimg)
    if halfframe:
        if is_taros(inimg):
            nslits = 12
            central_slit = 12
        else:
            nslits = 13
            central_slit = 6
    else:
        nslits = 25
        central_slit = 12

    # setup base x/y array
    f3 = pyfits.open(inimg)
    ndy_orig, ndx_orig = numpy.shape(f3[1].data)

    if subsample < 1 or subsample > 10:
        print(f"generate_wifes_cube: subsample must be between 1 and 10. Received {subsample}, setting to 1.")
        subsample = 1

    ndx = ndx_orig
    ndy = int(numpy.ceil(ndy_orig * subsample))
    xarr = numpy.arange(ndx)
    yarr = numpy.arange(ndy)
    full_y = numpy.meshgrid(xarr, yarr)[1]
    obs_hdr = f3[0].header

    # figure out the binning!
    try:
        default_bin_x, default_bin_y = [int(b) for b in f3[1].header["CCDSUM"].split()]
    except Exception:
        default_bin_x = 1
        default_bin_y = 1
    if bin_x is None:
        bin_x = default_bin_x
    if bin_y is None:
        bin_y = default_bin_y
    # get the min/max wavelengths across all slits, and average dispersion
    frame_wmin = 0.0
    frame_wmax = 20000.0
    frame_wdisps = []

    f4 = pyfits.open(wsol_fn)
    convert_wave = False
    if wavelength_ref.upper() == "VACUUM":
        wavelength_ref = "VACUUM"  # Ensure capitalised
        convert_wave = True
    else:
        wavelength_ref = "AIR"
    kwwavemodel = f4[0].header.get("PYWWAVEM", default="Unknown")
    kwwaverms = f4[0].header.get("PYWWRMSE", default="Unknown")
    kwwavenum = f4[0].header.get("PYWARCN", default="Unknown")

    central_wave = None
    for i in range(nslits):
        # Wavelenghts
        wave = f4[i + 1].data
        if convert_wave:
            # Remove the vacuum-to-air conversion applied by NIST to arcline
            # wavelengths. From Peck & Reeder (1972).
            n = 1.0 + 1E-8 * (
                8060.51 + 2480990.0 / (132.274 - numpy.float_power(wave / 1E4, -2))
                + 17455.7 / (39.32957 - numpy.float_power(wave / 1E4, -2))
            )
            wave = wave * n
        curr_wmin = numpy.nanmax(numpy.nanmin(wave, axis=1))
        curr_wmax = numpy.nanmin(numpy.nanmax(wave, axis=1))
        curr_wdisp = numpy.abs(numpy.nanmean(wave[:, 1:] - wave[:, :-1]))
        if curr_wmin > frame_wmin:
            frame_wmin = curr_wmin
        if curr_wmax < frame_wmax:
            frame_wmax = curr_wmax
        frame_wdisps.append(curr_wdisp)
        if wave_native and i == central_slit:
            central_wave = wave[wave.shape[0] // 2]
    f4.close()

    if dw_set is not None:
        disp_ave = dw_set
    else:
        disp_ave = numpy.nanmean(frame_wdisps)
    # shift lowest pixel so interpolation doesn't fail
    frame_wmin += numpy.abs(disp_ave)
    # finally check against the user input value
    if wmin_set is not None:
        final_frame_wmin = max(wmin_set, frame_wmin)
    else:
        final_frame_wmin = frame_wmin
    if wmax_set is not None:
        final_frame_wmax = min(wmax_set, frame_wmax)
    else:
        final_frame_wmax = frame_wmax

    # Output wavelength arrangement of datacube
    if wave_native and central_wave is not None:
        out_lambda = central_wave
        # Reset values that go into headers, even if non-linear
        final_frame_wmin = numpy.amin(out_lambda)
        final_frame_wmax = numpy.amax(out_lambda)
        mididx = out_lambda.shape[0] // 2
        disp_ave = out_lambda[mididx] - out_lambda[mididx - 1]
    else:
        out_lambda = numpy.arange(final_frame_wmin, final_frame_wmax, disp_ave)

    if verbose:
        print(
            " Data spectral resolution (min/max):",
            numpy.round(numpy.nanmin(frame_wdisps), 4),
            numpy.round(numpy.nanmax(frame_wdisps), 4),
        )
        print(" Cube spectral resolution : ", disp_ave)

    # load in spatial solutions
    try:
        f5 = pyfits.open(wire_fn)
        wire_trans = f5[0].data
        if subsample > 1:
            wire_trans = subsample * ndimage.zoom(wire_trans, zoom=[subsample, 1], order=0, mode='nearest', grid_mode=True)
        kwwiredeg = f5[0].header.get("PYWWIWPD", default="Unknown")
        kwwirenum = f5[0].header.get("PYWWIREN", default="Unknown")
        f5.close()
    except Exception:
        wire_trans = numpy.zeros([ndy, ndx], dtype="d") + numpy.nanmax(yarr) / 2
        kwwiredeg = "N/A"
        kwwirenum = "N/A"
    wire_offset = float(offset_orig) / float(bin_y) * subsample

    # set up output data
    ny = int(numpy.ceil(ny_orig / bin_y * subsample))
    nx = int(numpy.ceil(nslits * subsample))
    nlam = len(out_lambda)

    # for each slitlet...
    init_out_y = numpy.arange(ny, dtype="d")
    out_y = init_out_y - numpy.nanmedian(init_out_y)
    out_y_full, out_lambda_full = numpy.meshgrid(out_y, out_lambda)
    # ---------------------------
    # Prepare ADR corrections ...
    if adr:
        # observatory stuff
        if "LAT-OBS" in obs_hdr:
            lat = obs_hdr["LAT-OBS"]  # degrees
        else:
            lat = -31.27336  # Standard value for 2.3m
        # alt = obs_hdr["ALT-OBS"]  # meters
        if set(["DEC", "HA", "HAEND", "ZD", "ZDEND", "TELPAN"]).issubset(obs_hdr):
            dec = dec_dms2dd(obs_hdr["DEC"])
            # want to calculate average HA...
            ha_start = obs_hdr["HA"]
            ha_end = obs_hdr["HAEND"]
            ha = 0.5 * (ha_degrees(ha_start) + ha_degrees(ha_end))
            # and average ZD...
            zd_start = obs_hdr["ZD"]
            zd_end = obs_hdr["ZDEND"]
            zd = numpy.radians(0.5 * (zd_start + zd_end))
            secz = 1.0 / numpy.cos(zd)
            # tanz = numpy.tan(zd)
            # telescope PA!
            telpa = numpy.radians(obs_hdr["TELPAN"])

            # Assumes 12 C air temperature, 665 mmHg (887 hPa) air pressure,
            # and 6.0 mmHg water vapour, the approximate median clear-night
            # values for SSO from 10 years of SkyMapper data (2014-2024).
            # Water vapour pressure derived from air temperature and relative
            # humidity, following Stone 1996 (PASP, 108, 1051), equations 18-21.
            sso_temp = 12.0
            sso_pres = 665.0
            sso_wvp = 6.0
        else:
            # TCS info missing or incomplete, no ADR correction possible
            adr = False

    # ---------------------------
    # Create a temporary storage array for first iteration
    flux_data_cube_tmp = numpy.zeros([nx, ny, nlam])
    var_data_cube_tmp = numpy.ones([nx, ny, nlam])
    dq_data_cube_tmp = numpy.ones([nx, ny, nlam])

    if multithread:
        tasks = []

    # First interpolation : Wavelength + y (=wire & ADR)
    if verbose:
        print(" -> Step 1: interpolating along lambda and y (2D interp.)\r")
    if print_progress:
        sys.stdout.write("\r 0%")
        sys.stdout.flush()

    f4 = pyfits.open(wsol_fn)
    for i in range(nslits):
        curr_hdu = i + 1
        wave = f4[curr_hdu].data
        if convert_wave:
            # Remove the vacuum-to-air conversion applied by NIST to arcline
            # wavelengths. From Peck & Reeder (1972).
            n = 1.0 + 1E-8 * (
                8060.51 + 2480990.0 / (132.274 - numpy.float_power(wave / 1E4, -2))
                + 17455.7 / (39.32957 - numpy.float_power(wave / 1E4, -2))
            )
            wave = wave * n

        curr_flux = f3[curr_hdu].data
        curr_var = f3[curr_hdu + nslits].data
        curr_dq = f3[curr_hdu + 2 * nslits].data

        if subsample > 1:
            wave = ndimage.zoom(wave, zoom=[subsample, 1], order=0, mode='nearest', grid_mode=True)
            curr_flux = ndimage.zoom(curr_flux, zoom=[subsample, 1], order=0, mode='nearest', grid_mode=True)
            curr_var = ndimage.zoom(curr_var, zoom=[subsample, 1], order=0, mode='nearest', grid_mode=True)
            curr_dq = ndimage.zoom(curr_dq, zoom=[subsample, 1], order=0, mode='nearest', grid_mode=True)

        # convert the *x* pixels to lambda
        dw = numpy.abs(wave[:, 1:] - wave[:, :-1])
        full_dw = numpy.zeros(wave.shape)
        full_dw[:, 1:] = dw
        full_dw[:, 0] = dw[:, 0]
        for ii in range(int(numpy.ceil(i * subsample)), int(numpy.ceil((i + 1) * subsample))):
            # and *y* to real y
            curr_wire = wire_trans[ii, :]
            all_ypos = full_y - curr_wire - wire_offset

            # from the y-lambda-flux data, interpolate the flux
            # for the desired output y-lambda grid
            wave_flat = wave.flatten()
            all_ypos_flat = all_ypos.flatten()
            curr_flux_flat = (curr_flux / full_dw).flatten()
            curr_var_flat = (curr_var / full_dw**2).flatten()
            curr_dq_flat = curr_dq.flatten()
            # Calculate the ADR corrections (this is slow)
            if adr:
                adr_corr = adr_x_y(
                    wave_flat, secz=secz, objha=ha, objdec=dec, tellat=lat, teltemp=sso_temp, telpres=sso_pres, telwvp=sso_wvp, telpa=telpa, ref_wl=5600.
                )
                adr_y = adr_corr[1] * 0.5 * float(bin_y)
                all_ypos_flat -= adr_y

            if multithread:
                # Create tasks to calculate flux, var, and dq.
                flux_task = get_task(
                    _scale_grid_data,
                    (wave_flat, all_ypos_flat),
                    curr_flux_flat,
                    (out_lambda_full, out_y_full),
                    method="linear",
                    scale_factor=disp_ave,
                )

                var_task = get_task(
                    _scale_grid_data,
                    (wave_flat, all_ypos_flat),
                    curr_var_flat,
                    (out_lambda_full, out_y_full),
                    method="linear",
                    scale_factor=disp_ave**2,
                )

                dq_task = get_task(
                    _scale_grid_data,
                    (wave_flat, all_ypos_flat),
                    curr_dq_flat,
                    (out_lambda_full, out_y_full),
                    method="nearest",
                    fill_value=1.0,
                    scale_factor=1.0,
                )

                tasks.append(flux_task)
                tasks.append(var_task)
                tasks.append(dq_task)

            else:  # not multithread
                # Does the interpolation (this is equally slow ...)
                flux_data_cube_tmp[ii, :, :] = _scale_grid_data(
                    (wave_flat, all_ypos_flat),
                    curr_flux_flat,
                    (out_lambda_full, out_y_full),
                    method="linear",
                    scale_factor=disp_ave,
                )

                var_data_cube_tmp[ii, :, :] = _scale_grid_data(
                    (wave_flat, all_ypos_flat),
                    curr_var_flat,
                    (out_lambda_full, out_y_full),
                    method="linear",
                    scale_factor=disp_ave**2,
                )

                dq_data_cube_tmp[ii, :, :] = _scale_grid_data(
                    (wave_flat, all_ypos_flat),
                    curr_dq_flat,
                    (out_lambda_full, out_y_full),
                    method="nearest",
                    fill_value=1.0,
                    scale_factor=1.0,
                )

                if print_progress:
                    sys.stdout.flush()
                    sys.stdout.write("\r\r %d" % (ii / (numpy.ceil(nslits * subsample)) * 100.0) + "%")
                    sys.stdout.flush()
                    if ii == int(numpy.ceil(nslits * subsample)) - 1:
                        sys.stdout.write("\n")
    if multithread:
        results = map_tasks(tasks, max_processes=max_processes)
        for i in range(int(numpy.ceil(nslits * subsample))):
            flux_data_cube_tmp[i, :, :] = results[3 * i]
            var_data_cube_tmp[i, :, :] = results[3 * i + 1]
            dq_data_cube_tmp[i, :, :] = results[3 * i + 2]

    f4.close()

    # Second interpolation : x (=ADR)
    if adr:
        # To avoid interpolation issues at the edges,
        # add two extra values on either side (repeating each of the endpoints).
        in_x = numpy.arange(-1, int(numpy.ceil(nslits * subsample)) + 1, 1, dtype="d")
        out_x = numpy.arange(int(numpy.ceil(nslits * subsample)), dtype="d")
        if verbose:
            print(" -> Step 2: interpolating along x (1D interp.)")
        for i in range(0, nlam):
            # Adopting same default parameters as above
            adr_corr = adr_x_y(
                numpy.array([out_lambda[i]]),
                secz=secz,
                objha=ha,
                objdec=dec,
                tellat=lat,
                teltemp=sso_temp,
                telpres=sso_pres,
                telwvp=sso_wvp,
                telpa=telpa,
                ref_wl=5600.
            )
            adr_x = adr_corr[0]
            for j in range(0, ny):
                # here is the actual appending of endpoint data
                this_flux = numpy.append(
                    numpy.append(
                        flux_data_cube_tmp[0, j, i], flux_data_cube_tmp[:nx, j, i]
                    ),
                    flux_data_cube_tmp[nx - 1, j, i],
                )
                this_var = numpy.append(
                    numpy.append(
                        var_data_cube_tmp[0, j, i], var_data_cube_tmp[:nx, j, i]
                    ),
                    var_data_cube_tmp[nx - 1, j, i],
                )
                this_dq = numpy.append(
                    numpy.append(
                        dq_data_cube_tmp[0, j, i], dq_data_cube_tmp[:nx, j, i]
                    ),
                    dq_data_cube_tmp[nx - 1, j, i],
                )

                # do interpolation
                # print("Interp shapes: ", in_x.shape, adr_x.shape, this_flux.shape, out_x.shape, flux_data_cube_tmp.shape)
                # Interp shapes for subsample=3:  (41,) (1,) (41,) (39,) (39, 114, 3506)
                f = interp.interp1d(
                    in_x - adr_x,
                    this_flux,
                    kind="linear",
                    fill_value=0.0,
                    bounds_error=False,
                )
                g = interp.interp1d(
                    in_x - adr_x,
                    this_var,
                    kind="linear",
                    fill_value=0.0,
                    bounds_error=False,
                )
                h = interp.interp1d(
                    in_x - adr_x,
                    this_dq,
                    kind="nearest",
                    fill_value=3,
                    bounds_error=False,
                )

                flux_data_cube_tmp[:, j, i] = f(out_x)
                var_data_cube_tmp[:, j, i] = g(out_x)
                dq_data_cube_tmp[:, j, i] = h(out_x)

            if print_progress:
                if i > 0:
                    sys.stdout.flush()
                sys.stdout.write("\r\r %d" % (i / (nlam - 1.0) * 100.0) + "%")
                if i == nlam - 1:
                    sys.stdout.write("\n")

    # All done, at last ! Now, let's save it all ...
    if subsample > 1:
        flux_data_cube_tmp = blockwise_mean_3D(flux_data_cube_tmp, [subsample, subsample, 1])
        var_data_cube_tmp = blockwise_mean_3D(var_data_cube_tmp, [subsample, subsample, 1])
        dq_data_cube_tmp = blockwise_mean_3D(dq_data_cube_tmp, [subsample, subsample, 1])
    outfits = pyfits.HDUList(f3)
    flux_data_cube_tmp = flux_data_cube_tmp.astype("float32", casting="same_kind")
    var_data_cube_tmp = var_data_cube_tmp.astype("float64", casting="same_kind")
    # trim data beyond range
    dq_data_cube_tmp[dq_data_cube_tmp > 32767] = 32767
    dq_data_cube_tmp[dq_data_cube_tmp < -32768] = -32768
    dq_data_cube_tmp = dq_data_cube_tmp.astype("int16", casting="unsafe")
    for i in range(nslits):
        # save to data cube
        outfits[i + 1].data = flux_data_cube_tmp[i, :, :]
        outfits[i + 1].scale("float32")
        if not wave_native:
            outfits[i + 1].header.set("CRVAL1", final_frame_wmin)
            outfits[i + 1].header.set("CRPIX1", 1)
            outfits[i + 1].header.set("CDELT1", disp_ave)
        outfits[i + 1].header.set("NAXIS1", len(out_lambda))
        outfits[i + 1 + nslits].data = var_data_cube_tmp[i, :, :]
        outfits[i + 1 + nslits].scale("float32")
        if not wave_native:
            outfits[i + 1 + nslits].header.set("CRVAL1", final_frame_wmin)
            outfits[i + 1 + nslits].header.set("CRPIX1", 1)
            outfits[i + 1 + nslits].header.set("CDELT1", disp_ave)
        outfits[i + 1 + nslits].header.set("NAXIS1", len(out_lambda))
        outfits[i + 1 + 2 * nslits].data = dq_data_cube_tmp[i, :, :]
        outfits[i + 1 + 2 * nslits].scale("int16")
        if not wave_native:
            outfits[i + 1 + 2 * nslits].header.set("CRVAL1", final_frame_wmin)
            outfits[i + 1 + 2 * nslits].header.set("CRPIX1", 1)
            outfits[i + 1 + 2 * nslits].header.set("CDELT1", disp_ave)
        outfits[i + 1 + 2 * nslits].header.set("NAXIS1", len(out_lambda))
    outfits[0].header.set("PYWIFES", __version__, "PyWiFeS version")
    outfits[0].header.set("PYWWAVEM", kwwavemodel, "PyWiFeS: method for wavelength solution")
    outfits[0].header.set("PYWWRMSE", kwwaverms, "PyWiFeS: Final RMSE of wavelength solution")
    outfits[0].header.set("PYWARCN", kwwavenum, "PyWiFeS: number of arc exposures combined")
    outfits[0].header.set("PYWWIWPD", kwwiredeg, "PyWiFeS: wire_polydeg")
    outfits[0].header.set("PYWWIREN", kwwirenum, "PyWiFeS: number of wire exposures combined")
    outfits[0].header.set("PYWYORIG", ny_orig, "PyWiFeS: ny_orig")
    outfits[0].header.set("PYWOORIG", offset_orig, "PyWiFeS: offset_orig")
    outfits[0].header.set("PYWADR", adr, "PyWiFeS: ADR correction applied")
    outfits[0].header.set("PYWWVREF", wavelength_ref, "PyWiFeS: wavelength reference (air or vacuum)")
    if subsample > 1:
        outfits[0].header.set("PYWSSAMP", subsample, "PyWiFeS: wire/ADR spatial subsampling factor")
    if wave_native:
        outfits[0].header.set("PYWWNONL", True, "PyWiFeS: non-linear wavelengths axis")
        outfits.append(pyfits.ImageHDU(data=out_lambda, header=outfits[1].header, name="WAVELENGTH"))
    outfits.writeto(outimg, overwrite=True)
    f3.close()
    return


# ------------------------------------------------------------------------
def generate_wifes_3dcube(inimg, outimg, halfframe=False, taros=False, nan_bad_pixels=False, debug=False):
    """
    Convert multi-extension FITS datacube into a single three-dimensional FITS image.

    Parameters
    ----------
    inimg : str
        Path to the input image.
    outimg : str
        Path to the output image.
    halfframe : bool, optional
        Whether the input image uses stellar (half-frame) readout mode. Default is False.
    taros : bool, optional
        Whether the input image was acquired with Taros (prior to the Automated 2.3m operations).
        Default is False.
    nan_bad_pixels : bool, optional
        Set FLUX extension to NaN where DQ extension > 0. Default is False.
    debug : bool, optional
        Whether to report the parameters used in this function call. Default is False.

    Returns
    -------
    None
    """
    # load in data
    if debug:
        print(arguments())
    f = pyfits.open(inimg)
    # full frame or half
    if len(f) >= 76 or (
        halfframe and (
            (taros and len(f) >= 37)
            or (not taros and len(f) >= 40)
        )
    ):
        ny, nlam = numpy.shape(f[1].data)
        if halfframe:
            if taros:
                nx = 12
            else:
                nx = 13
        else:
            nx = 25

        if "WAVELENGTH" in f:
            lam0 = f["WAVELENGTH"].data[0]
            wave_ext = True
        else:
            lam0 = f[1].header["CRVAL1"]
            pix0 = f[1].header["CRPIX1"]
            dlam = f[1].header["CDELT1"]
            wave_ext = False

        # get data, variance and data quality
        obj_cube_data = numpy.zeros([nlam, ny, nx], dtype="d")
        obj_cube_var = numpy.zeros([nlam, ny, nx], dtype="d")
        obj_cube_dq = numpy.zeros([nlam, ny, nx], dtype="d")

        for i in range(nx):
            curr_hdu = i + 1
            curr_data = f[curr_hdu].data
            curr_var = f[curr_hdu + nx].data
            curr_dq = f[curr_hdu + 2 * nx].data
            obj_cube_data[:, :, nx - i - 1] = curr_data.T
            obj_cube_var[:, :, nx - i - 1] = curr_var.T
            obj_cube_dq[:, :, nx - i - 1] = curr_dq.T

    else:
        nlam, ny, nx = numpy.shape(f[1].data)
        # get wavelength array
        if "WAVELENGTH" in f:
            lam0 = f["WAVELENGTH"].data[0]
            wave_ext = True
        else:
            lam0 = f[1].header["CRVAL1"]
            pix0 = f[1].header["CRPIX1"]
            dlam = f[1].header["CDELT1"]
            wave_ext = False
        # get data and variance
        obj_cube_data = numpy.zeros([nlam, ny, nx], dtype="d")
        obj_cube_var = numpy.zeros([nlam, ny, nx], dtype="d")
        obj_cube_dq = numpy.zeros([nlam, ny, nx], dtype="d")
        for i in range(nx):
            curr_data = f[1].data[:, :, i]
            curr_var = f[2].data[:, :, i]
            curr_dq = f[3].data[:, :, i]
            if nan_bad_pixels:
                curr_data[curr_dq > 0] = numpy.nan
            obj_cube_data[:, :, nx - i - 1] = curr_data
            obj_cube_var[:, :, nx - i - 1] = curr_var
            obj_cube_dq[:, :, nx - i - 1] = curr_dq

    # ASTROMETRY: obtained from pointing
    try:
        # # Celestial coordinates
        ra_str = f[1].header["RA"]
        dec_str = f[1].header["DEC"]
        # Convert celestial coordinates to degrees
        coord = SkyCoord(ra=ra_str, dec=dec_str, unit=(u.hourangle, u.deg))
        crval1 = coord.ra.deg
        crval2 = coord.dec.deg
    except KeyError:
        crval1 = 0.
        crval2 = 0.
    crval3 = lam0

    # Set up a tangential projection
    ctype1 = "RA---TAN"
    ctype2 = "DEC--TAN"
    if wave_ext:
        ctype3 = "PIXEL"
    else:
        ctype3 = "WAVE"

    # Coordinate transformations
    # Telescope angle
    try:
        telpa = f[1].header["TELPAN"]  # in deg
        telpa_radians = numpy.radians(telpa)  # in rad
        #  Calculate matrix elements for 3D rotation (around z-axis)
        cos_telpa = numpy.cos(telpa_radians)
        sin_telpa = numpy.sin(telpa_radians)

        pc1_1 = cos_telpa
        pc1_2 = -sin_telpa

        pc2_1 = sin_telpa
        pc2_2 = cos_telpa
    except KeyError:
        pc1_1 = 1
        pc1_2 = 0
        pc2_1 = 0
        pc2_2 = 1

    # Equiv. to 1 pixel width in each axis' units
    try:
        binning_2 = int(f[0].header["CCDSUM"].split()[1])
    except KeyError:
        # Default to common 1x2 binning assumption
        binning_2 = 2
    arsec_deg = 0.00027777777777778  # 1 arcsecond in degrees
    cdelt1 = -arsec_deg
    cdelt2 = arsec_deg / 2 * binning_2
    if wave_ext:
        cdelt3 = 1
    else:
        cdelt3 = dlam

    # Central pixel: defined in the centre of the central pixel
    crpix1 = nx / 2 + 0.5  # Central pixel
    crpix2 = ny / 2 + 0.5  # Central pixel
    if wave_ext:
        crpix3 = 1
    else:
        crpix3 = pix0

    # Axis units
    cunit1 = "deg"
    cunit2 = "deg"
    if wave_ext:
        cunit3 = "pixel"
    else:
        cunit3 = "Angstrom"

    # save to data cube
    # DATA
    cube_hdu = pyfits.PrimaryHDU(obj_cube_data.astype("float32", casting="same_kind"), header=f[0].header)
    cube_hdu.scale("float32")
    # Add header info
    cube_hdu.name = 'SCI'
    cube_hdu.header.set("CTYPE1", ctype1, "Type of co-ordinate on axis 1")
    cube_hdu.header.set("CTYPE2", ctype2, "Type of co-ordinate on axis 2")
    cube_hdu.header.set("CUNIT1", cunit1, "Units for axis 1")
    cube_hdu.header.set("CUNIT2", cunit2, "Units for axis 2")
    cube_hdu.header.set("CRVAL1", crval1, "Value at ref. pixel on axis 1")
    cube_hdu.header.set("CRVAL2", crval2, "Value at ref. pixel on axis 2")
    cube_hdu.header.set("CDELT1", cdelt1, "RA axis pixel width")
    cube_hdu.header.set("CDELT2", cdelt2, "Dec axis pixel width")
    cube_hdu.header.set("CRPIX1", crpix1, "Reference pixel on axis 1")
    cube_hdu.header.set("CRPIX2", crpix2, "Reference pixel on axis 2")
    cube_hdu.header.set("PC1_1", pc1_1, "Transformation matrix element")
    cube_hdu.header.set("PC1_2", pc1_2, "Transformation matrix element")
    cube_hdu.header.set("PC2_1", pc2_1, "Transformation matrix element")
    cube_hdu.header.set("PC2_2", pc2_2, "Transformation matrix element")
    cube_hdu.header.set("CTYPE3", ctype3, "Type of co-ordinate on axis 3")
    cube_hdu.header.set("CUNIT3", cunit3, "Units for axis 3")
    cube_hdu.header.set("CRVAL3", crval3, "Reference wavelength")
    cube_hdu.header.set("CDELT3", cdelt3, "Wavelength step")
    cube_hdu.header.set("CRPIX3", crpix3, "Reference pixel on wavelength (axis 3)")
    outfits = pyfits.HDUList([cube_hdu])

    # VARIANCE
    var_hdu = pyfits.PrimaryHDU(obj_cube_var.astype("float64", casting="same_kind"), header=f[0].header)
    var_hdu.scale("float64")
    # Add header info
    var_hdu.name = 'VAR'
    var_hdu.header.set("CTYPE1", ctype1, "Type of co-ordinate on axis 1")
    var_hdu.header.set("CTYPE2", ctype2, "Type of co-ordinate on axis 2")
    var_hdu.header.set("CUNIT1", cunit1, "Units for axis 1")
    var_hdu.header.set("CUNIT2", cunit2, "Units for axis 2")
    var_hdu.header.set("CRVAL1", crval1, "Value at ref. pixel on axis 1")
    var_hdu.header.set("CRVAL2", crval2, "Value at ref. pixel on axis 2")
    var_hdu.header.set("CDELT1", cdelt1, "RA axis pixel width")
    var_hdu.header.set("CDELT2", cdelt2, "Dec axis pixel width")
    var_hdu.header.set("CRPIX1", crpix1, "Reference pixel on axis 1")
    var_hdu.header.set("CRPIX2", crpix2, "Reference pixel on axis 2")
    var_hdu.header.set("PC1_1", pc1_1, "Transformation matrix element")
    var_hdu.header.set("PC1_2", pc1_2, "Transformation matrix element")
    var_hdu.header.set("PC2_1", pc2_1, "Transformation matrix element")
    var_hdu.header.set("PC2_2", pc2_2, "Transformation matrix element")
    var_hdu.header.set("CTYPE3", ctype3, "Type of co-ordinate on axis 3")
    var_hdu.header.set("CUNIT3", cunit3, "Units for axis 3")
    var_hdu.header.set("CRVAL3", crval3, "Reference wavelength")
    var_hdu.header.set("CDELT3", cdelt3, "Wavelength step")
    var_hdu.header.set("CRPIX3", crpix3, "Reference pixel on wavelength (axis 3)")
    outfits.append(var_hdu)

    # DQ
    # trim data beyond range
    obj_cube_dq[obj_cube_dq > 32767] = 32767
    obj_cube_dq[obj_cube_dq < -32768] = -32768
    dq_hdu = pyfits.PrimaryHDU(obj_cube_dq.astype("int16", casting="unsafe"), header=f[0].header)
    dq_hdu.scale("int16")
    # Add header info
    dq_hdu.name = 'DQ'
    dq_hdu.header.set("CTYPE1", ctype1, "Type of co-ordinate on axis 1")
    dq_hdu.header.set("CTYPE2", ctype2, "Type of co-ordinate on axis 2")
    dq_hdu.header.set("CUNIT1", cunit1, "Units for axis 1")
    dq_hdu.header.set("CUNIT2", cunit2, "Units for axis 2")
    dq_hdu.header.set("CRVAL1", crval1, "Value at ref. pixel on axis 1")
    dq_hdu.header.set("CRVAL2", crval2, "Value at ref. pixel on axis 2")
    dq_hdu.header.set("CDELT1", cdelt1, "RA axis pixel width")
    dq_hdu.header.set("CDELT2", cdelt2, "Dec axis pixel width")
    dq_hdu.header.set("CRPIX1", crpix1, "Reference pixel on axis 1")
    dq_hdu.header.set("CRPIX2", crpix2, "Reference pixel on axis 2")
    dq_hdu.header.set("PC1_1", pc1_1, "Transformation matrix element")
    dq_hdu.header.set("PC1_2", pc1_2, "Transformation matrix element")
    dq_hdu.header.set("PC2_1", pc2_1, "Transformation matrix element")
    dq_hdu.header.set("PC2_2", pc2_2, "Transformation matrix element")
    dq_hdu.header.set("CTYPE3", ctype3, "Type of co-ordinate on axis 3")
    dq_hdu.header.set("CUNIT3", cunit3, "Units for axis 3")
    dq_hdu.header.set("CRVAL3", crval3, "Reference wavelength")
    dq_hdu.header.set("CDELT3", cdelt3, "Wavelength step")
    dq_hdu.header.set("CRPIX3", crpix3, "Reference pixel on wavelength (axis 3)")
    outfits.append(dq_hdu)

    # Pass along wavelength array, if present
    try:
        wdata = f['WAVELENGTH'].data
        whead = f['WAVELENGTH'].header
        wext = pyfits.ImageHDU(data=wdata, header=whead, name="WAVELENGTH")
        outfits.append(wext)
    except KeyError:
        pass
    except Exception as e:
        print(f"Error attaching wavelength array: {e}")

    # Pass along telluric spectrum, if present
    try:
        tdata = f['TelluricModel'].data
        thead = f['TelluricModel'].header
        tellext = pyfits.ImageHDU(data=tdata, header=thead, name="TelluricModel")
        outfits.append(tellext)
    except KeyError:
        pass
    except Exception as e:
        print(f"Error attaching telluric model: {e}")

    # Pass along corrected extinction spectrum, if present
    try:
        edata = f['Extinction'].data
        ehead = f['Extinction'].header
        ext_ext = pyfits.ImageHDU(data=edata, header=ehead, name="Extinction")
        outfits.append(ext_ext)
    except KeyError:
        pass
    except Exception as e:
        print(f"Error attaching extinction spectrum: {e}")

    # SAVE IT
    outfits[0].header.set("PYWIFES", __version__, "PyWiFeS version")
    if "EQUINOX" in outfits[0].header and isinstance(outfits[0].header["EQUINOX"], str) \
            and "J" in outfits[0].header["EQUINOX"]:
        outfits[0].header["EQUINOX"] = (float(outfits[0].header["EQUINOX"].replace("J", "")),
                                        "Equinox of coordinates")
    outfits.writeto(outimg, overwrite=True)
    f.close()
    return
