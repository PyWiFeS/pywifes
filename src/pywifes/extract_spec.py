import astropy.io.fits as fits
from astropy.stats import sigma_clipped_stats
from astropy.table import Column, vstack
from astropy.wcs import WCS
import matplotlib.colors as mcolors
from matplotlib.patheffects import withStroke
import matplotlib.pyplot as plt
import numpy
import os
from photutils.aperture import EllipticalAperture, EllipticalAnnulus
from photutils.detection import DAOStarFinder
import re

from pywifes.wifes_utils import arguments, is_halfframe

# Suppress the NoDetectionsWarning as we have set a warning for no detection
import warnings
from photutils.utils.exceptions import NoDetectionsWarning
warnings.filterwarnings("ignore", category=NoDetectionsWarning)


def extract_detection_name(spec_name):
    match = re.search(r"det(\d)", spec_name)
    if match:
        detection_number = match.group(1)
        return f"Detection {detection_number}"
    return None


def extract_object_name(hdr):
    if "OBJECT" in hdr and hdr["OBJECT"] != "":
        return hdr["OBJECT"] + " - "
    elif "OBJNAME" in hdr and hdr["OBJNAME"] != "":
        return hdr["OBJNAME"] + " - "
    return ""


def extract_and_save(
    cube_path, sci, var, source_region, det_index, sky_region, output_dir, sci_hdr,
    var_hdr, extraction_method='aperture', sky_method='annulus', sky_stat='wmean',
    dq_data=None, dq_hdr=None, wave_data=None, tell_data=None, ext=None, ext_hdr=None,
):
    if cube_path is not None:
        kwlist = [
            ["PYWXMETH", extraction_method,
             "PyWiFeS: extraction method"],
            ["PYWXAPX", numpy.round(source_region.positions[0] + 1.0, decimals=2),
             "PyWiFeS: extraction x centroid"],
            ["PYWXAPY", numpy.round(source_region.positions[1] + 1.0, decimals=2),
             "PyWiFeS: extraction y centroid"]
        ]
        if sky_region is None:
            bkg_inner = None
            bkg_outer = None
            kwlist.extend([["PYWXSKYS", False,
                            "PyWiFeS: extraction was sky-subtracted"]])
        else:
            if sky_method == "annulus":
                bkg_inner = sky_region.a_in
                bkg_outer = sky_region.a_out
                kwlist.extend([
                    ["PYWXSKYS", True,
                     "PyWiFeS: extraction was sky-subtracted"],
                    ["PYWXSKYM", "annulus",
                     "PyWiFeS: extraction sky method"],
                    ["PYWXSKYI", numpy.round(bkg_inner, decimals=2),
                     "PyWiFeS: extraction sky annulus r_in (pixels)"],
                    ["PYWXSKYO", numpy.round(bkg_outer, decimals=2),
                     "PyWiFeS: extraction sky annulus r_out (pixels)"],
                ])
            elif sky_method == "same_slice":
                kwlist.extend([
                    ["PYWXSKYS", True,
                     "PyWiFeS: extraction was sky-subtracted"],
                    ["PYWXSKYM", "same_slice",
                     "PyWiFeS: extraction sky method"],
                    ["PYWXSKYA", numpy.round(sky_region.a, decimals=2),
                     "PyWiFeS: extraction sky semi-major axis (pix)"],
                    ["PYWXSKYB", numpy.round(sky_region.b, decimals=2),
                     "PyWiFeS: extraction sky semi-minor axis (pix)"],
                    ["PYWXSKYX", numpy.round(sky_region.positions[0] + 1.0, decimals=2),
                     "PyWiFeS: extraction sky X position"],
                    ["PYWXSKYY", numpy.round(sky_region.positions[1] + 1.0, decimals=2),
                     "PyWiFeS: extraction sky Y position"],
                ])
            if extraction_method == 'aperture':
                kwlist.extend([
                    ["PYWXSKY", "science aperture total",
                     "PyWiFeS: extraction SKY extension type"]])

        # Extraction
        if extraction_method == 'aperture':
            flux, fvar, dq, sky, tell = aperture_extract(sci, var, source_region,
                                                         sky_ap=sky_region,
                                                         dq_cube=dq_data,
                                                         tell_data=tell_data,
                                                         sky_stat=sky_stat)

            kwlist.extend([
                ["PYWXAMAJ", source_region.a,
                 "PyWiFeS: extraction semi-major axis (pixels)"],
                ["PYWXAMIN", source_region.b,
                 "PyWiFeS: extraction semi-minor axis (pixels)"]])

        else:
            print(f"Unknown extraction method <{extraction_method}>. Not performed.")
            return

        # Write out the results
        base = os.path.basename(cube_path)
        base_output = base.replace("cube.fits", f"spec.det{det_index}.fits")
        print(f"Saving extracted spectra for detection {det_index} in {base}")
        output_path = os.path.join(output_dir, base_output)
        write_1D_spec(flux, fvar, sci_hdr, var_hdr, output_path, dq_data=dq,
                      dq_cube_header=dq_hdr, sky_data=sky, wave_data=wave_data,
                      tell_data=tell, kwlist=kwlist, ext_data=ext, ext_hdr=ext_hdr)


def plot_arm(ax, cube_path, sci, title, source_regions, sky_regions, border_width,
             bin_y):
    ax.set_title(title)
    if cube_path is not None:
        plot_regions(
            sci, source_regions, sky_regions=sky_regions, border_width=border_width,
            bin_y=bin_y
        )
        ax.set_xlabel("Right Ascension")
        ax.set_ylabel("Declination")
    else:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        text = f"No {title.lower()} arm data"
        ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=12, color="black")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")


def collapse_cube(*data_cubes):
    """
    Collapse one or more data cubes by computing the median along the wavelength
    dimension.

    Parameters
    ----------
    *data_cubes : array_like
        Variable-length argument list of data cubes to be collapsed.
        Each data cube should have the same spatial shape.

    Returns
    -------
    median_image : numpy.ndarray
        2D array representing the median-collapsed image along the wavelength dimension.

    Notes
    -----
    This function takes one or more data cubes and collapses them by computing the
    median along the wavelength dimension. The resulting 2D array represents the
    median image.
    """
    # Filter out None elements from the data_cubes
    valid_data_cubes = [cube for cube in data_cubes if cube is not None]

    if len(valid_data_cubes) == 1:
        joined_cubes = valid_data_cubes[0]
    else:
        joined_cubes = numpy.concatenate(valid_data_cubes, axis=0)

    median = numpy.nanmedian(joined_cubes, axis=0)

    return median


def sec_image(ap, image):
    """
    Extract pixel values from a section within the aperture.

    Parameters
    ----------
    aperture : `photutils.aperture` object
        Aperture object representing the region of interest.

    image : array_like
        2D or 3D array representing the image data.

    Returns
    -------
    sec_data : numpy.ndarray
        1D or 2D array containing the weighted pixel values from the section within the
        aperture.

    Notes
    -----
    This function extracts pixel values from a section within the specified aperture
    from the input image. The extraction is performed using a aperture mask,
    where pixel values outside the aperture are zero. The resulting
    1D or 2D array contains the pixel values from the section within the aperture.
    """

    mask = ap.to_mask(method="center")
    if image.ndim == 2:
        sec_data = mask.get_values(image)
    elif image.ndim == 3:
        sec_data = image[:, mask.to_image(image.shape[1:3]) > 0]
    return sec_data


def aperture_extract(sci_cube, var_cube, source_ap, sky_ap=None, dq_cube=None,
                     tell_data=None, sky_stat="wmean"):
    """
    Extract and calculate spectral flux and variance within the given aperture.

    Parameters
    ----------
    sci_cube : array_like
        3D array representing the scientific data cube.

    var_cube : array_like
        3D array representing the variance data cube.

    source_ap : `photutils.aperture` object
        Aperture object representing the source region to be extracted.

    sky_ap : `photutils.aperture` object, optional
        Aperture object representing the sky region to be extracted.
        Default is None.

    dq_cube : array_like, optional
        3D array representing the data quality cube.
        Default: None.

    tell_data : array_like, optional
        1D or 2D array with the applied telluric model.
        Default: None.

    sky_stat : str, optional
        Statistical method for calculating sky spectrum. Options: "wmean", "median".
        Default: "wmean".

    Returns
    -------
    flux : numpy.ndarray
        1D array containing the spectral flux values.

    variance : numpy.ndarray
        1D array containing the spectral variance values.

    dq : numpy.ndarray
        1D array containing the data quality values.

    sky : numpy.ndarray
        1D array containing the sky spectrum that was subtracted, or None if no sky
        extracted (e.g., Nod & Shuffle).

    tell : numpy.ndarray
        1D array containing the median telluric model applied to the spectrum, or
        None if not available in the datacube.

    Notes
    -----
    This function extracts spectral flux and variance values within the specified
    source aperture from the input data cubes. If a sky aperture is provided,
    it calculates and subtracts the corresponding sky values. If a data quality
    cube is provided, it also extracts the corresponding spectrum. If a telluric
    model is provided, it also extracts the median telluric spectrum applicable
    to the extraction region.
    """

    fscale = numpy.abs(numpy.nanmax(sci_cube))
    if fscale == 0.0 or numpy.isnan(fscale):
        fscale = 1.0
    else:
        print(f"Temporary scaling by {fscale}")
    sci_cube = sci_cube.copy() / fscale
    var_cube = var_cube.copy() / (fscale * fscale)

    fl = numpy.nansum(sec_image(source_ap, sci_cube), axis=1)
    var = numpy.nansum(sec_image(source_ap, var_cube), axis=1)
    sky = None

    if sky_ap is not None:
        area = source_ap.area_overlap(sci_cube[sci_cube.shape[1] // 2, :, :],
                                      method="center")
        sky_var_sec = sec_image(sky_ap, var_cube)
        if sky_stat == "wmean":
            if dq_cube is not None:
                sky_dq_sec = sec_image(sky_ap, dq_cube)
                sky_var_sec[sky_dq_sec > 0] = 0
            num_zero = numpy.count_nonzero(sky_var_sec == 0)
            if num_zero:
                med_var = numpy.nanmedian(var_cube)
                print(f"Replacing {num_zero} 0-variance sky VAR pixels with 10^8 x VAR median: {1.0E8 * med_var:.3g}")
                sky_var_sec[sky_var_sec == 0.0] = 1.0E8 * med_var
            sky_average = numpy.average(
                sec_image(sky_ap, sci_cube),
                weights=numpy.reciprocal(sky_var_sec),
                axis=1
            )
            sky_var = numpy.reciprocal(numpy.nansum(numpy.reciprocal(sky_var_sec)))
        elif sky_stat == "median":
            sky_sec = sec_image(sky_ap, sci_cube)
            if dq_cube is not None:
                sky_dq_sec = sec_image(sky_ap, dq_cube)
                sky_sec[sky_dq_sec > 0] = numpy.nan
                sky_var_sec[sky_dq_sec > 0] = numpy.nan
            sky_average = numpy.nanmedian(sky_sec, axis=1)
            # Approximate the variance of the median with Woodruff (1952) estimator
            lo_pcent = numpy.clip(0.5 * (area - numpy.sqrt(area)) / area * 100.0, a_min=1.0, a_max=99.0)
            hi_pcent = numpy.clip(0.5 * (area + numpy.sqrt(area)) / area * 100.0, a_min=1.0, a_max=99.0)
            sky_lo = numpy.nanpercentile(sky_sec, q=lo_pcent, axis=1)
            sky_hi = numpy.nanpercentile(sky_sec, q=hi_pcent, axis=1)
            sky_var = numpy.power((sky_hi - sky_lo) / 2.0, 2)

        sky = sky_average * area

        fl -= sky
        var += sky_var * area

        sky *= fscale
    fl *= fscale
    var *= (fscale * fscale)

    if dq_cube is not None:
        dq = numpy.nansum(sec_image(source_ap, dq_cube), axis=1)
    else:
        dq = None

    if tell_data is not None:
        if tell_data.ndim > 1:
            # Rearrange the axes and extract the median telluric correction applicable
            # to the aperture
            tell = numpy.nanmedian(
                sec_image(source_ap,
                          numpy.transpose(numpy.tile(tell_data,
                                                     (sci_cube.shape[1], 1, 1)),
                                          [2, 0, 1])),
                axis=1)
        else:
            tell = tell_data
    else:
        tell = None

    return fl, var, dq, sky, tell


def write_1D_spec(sci_data, var_data, sci_cube_header, var_cube_header, output,
                  dq_data=None, dq_cube_header=None, sky_data=None, wave_data=None,
                  tell_data=None, kwlist=None, ext_data=None, ext_hdr=None):
    """
    This function writes the 1D spectrum data and its variance to a FITS file.
    It updates the WCS headers to match the 1D spectrum header.

    Parameters
    ----------
    sci_data : numpy.ndarray
        1D array containing the science data (flux) of the spectrum.
    var_data : numpy.ndarray
        1D array containing the variance data of the spectrum.
    sci_cube_header : astropy.io.fits.Header
        Header from the science cube.
    var_cube_header : astropy.io.fits.Header
        Header object from the variance data.
    output : str
        Output file path for the FITS file containing the 1D spectrum.
    dq_data : numpy.ndarray, optional
        1D array containing the data quality data of the spectrum. Optional.
    dq_cube_header : astropy.io.fits.Header, optional
        Header object from the data quality extension.
    sky_data : numpy.ndarray, optional
        1D array containing the subtracted sky spectrum, or None.
    wave_data : numpy.ndarray, optional
        1D array containing wavelength value per pixel, or None.
    tell_data : numpy.ndarray, optional
        1D or 2D array containing the applied telluric model, or None.
    kwlist : list of lists, optional
        List of header [keyword, list, commment] entries to write to output file.
    ext_data : numpy.ndarray, optional
        1D array containing the corrected extinction that was applied, or None.
    ext_hdr : astropy.io.fits.Header, optional
        Header object from corrected extinction extension, or None.

    Returns
    -------
    None
    """

    # Create a blank HDUList for the 1D spectrum
    hdulist = fits.HDUList(fits.PrimaryHDU())

    # Update WCS headers to match the 1D spectrum header
    sci_cube_header_copy = sci_cube_header.copy()
    var_cube_header_copy = var_cube_header.copy()
    headers = [sci_cube_header_copy, var_cube_header_copy]
    if dq_data is not None and dq_cube_header is not None:
        dq_cube_header_copy = dq_cube_header.copy()
        headers.append(dq_cube_header_copy)

    for header in headers:
        if wave_data is None:
            # Update axis 1 WCS information to match wavelength solution (axis 3)
            header["CDELT1"] = header["CDELT3"]
            header["CRPIX1"] = header["CRPIX3"]
            header["CRVAL1"] = header["CRVAL3"]
            header["CUNIT1"] = "Angstrom"
            header["CTYPE1"] = header["CTYPE3"]
            header["NAXIS1"] = header["NAXIS3"]
        else:
            # Wavelengths provided in separate extension
            header["CDELT1"] = 1
            header["CRPIX1"] = 1
            header["CRVAL1"] = 1
            header["CUNIT1"] = "pixel"
            header["CTYPE1"] = "Pixel"
            header["NAXIS1"] = wave_data.shape[0]
        header["NAXIS"] = 1

        # Remove spatial WCS keys
        keys_to_remove = [
            "NAXIS2",
            "NAXIS3",
            "CRVAL2",
            "CRVAL3",
            "CTYPE2",
            "CTYPE3",
            "CDELT2",
            "CDELT3",
            "CRPIX2",
            "CRPIX3",
            "CUNIT2",
            "CUNIT3",
            "PC1_1",
            "PC1_2",
            "PC2_1",
            "PC2_2",
        ]
        for key in keys_to_remove:
            if key in header:
                header.remove(key)

    # Set science data and headers in the blank HDUList
    hdulist[0].data = sci_data.astype("float32", casting="same_kind")
    hdulist[0].scale("float32")
    hdulist[0].header = sci_cube_header_copy
    if kwlist is not None:
        for kw, val, desc in kwlist:
            hdulist[0].header.set(kw, val, desc)

    # Add variance data as an additional HDU
    hdu_fluxvar = fits.ImageHDU(data=var_data.astype("float64", casting="same_kind"),
                                header=var_cube_header_copy)
    hdu_fluxvar.scale("float64")
    hdulist.append(hdu_fluxvar)

    # Add data quality extention if provided
    if dq_data is not None and dq_cube_header is not None:
        hdu_dq = fits.ImageHDU(data=dq_data.astype("int16", casting="unsafe"),
                               header=dq_cube_header_copy)
        hdu_dq.scale("int16")
        hdulist.append(hdu_dq)

    # Add sky extention if provided
    if sky_data is not None:
        hdu_sky = fits.ImageHDU(data=sky_data.astype("float32", casting="same_kind"),
                                header=sci_cube_header_copy)
        hdu_sky.scale("float32")
        hdu_sky.header['EXTNAME'] = 'SKY'
        hdulist.append(hdu_sky)

    # Add wavelength extension if provided
    if wave_data is not None:
        hdu_wave = fits.ImageHDU(data=wave_data.astype("float32", casting="same_kind"),
                                 header=sci_cube_header_copy)
        hdu_wave.scale("float32")
        hdu_wave.header['EXTNAME'] = 'WAVELENGTH'
        hdulist.append(hdu_wave)

    # Add telluric extension, if provided
    if tell_data is not None:
        hdu_tell = fits.ImageHDU(data=tell_data.astype("float32", casting="same_kind"),
                                 header=sci_cube_header_copy)
        hdu_tell.scale("float32")
        hdu_tell.header['EXTNAME'] = 'TELLURICMODEL'
        hdulist.append(hdu_tell)

    # Add corrected extinction extension, if provided
    if ext_data is not None:
        hdu_ext = fits.ImageHDU(data=ext_data.astype("float32", casting="same_kind"))
        if ext_hdr is None:
            for kw in ['CTYPE1', 'CUNIT1', 'CRVAL1', 'CDELT1', 'CRPIX1']:
                if kw in sci_cube_header_copy:
                    hdu_ext.header[kw] = sci_cube_header_copy[kw]
            hdu_ext.header['COMMENT'] = 'Flux extinction correction applied to reach airmass zero'
            hdu_ext.header['COMMENT'] = 'NB: the correction _in magnitudes_ scales linearly with airmass'
            hdu_ext.header['AIRMASS'] = hdulist[0].header['AIRMASS']
            hdu_ext.header['BUNIT'] = 'Flux fraction'
        else:
            hdu_ext.header = ext_hdr
        hdu_ext.scale("float32")
        hdu_ext.header['EXTNAME'] = 'EXTINCTION'
        hdulist.append(hdu_ext)

    # Write the FITS file containing the 1D spectrum
    hdulist.writeto(output, overwrite=True)
    hdulist.close()


def plot_regions(data_cube, source_regions, sky_regions=None, border_width=0, bin_y=1):
    """
    Plot regions on a collapsed data cube image.

    Parameters
    ----------
    data_cube : numpy.ndarray
        The 3D data cube containing the image data.

    source_regions : list of photutils.aperture objects
        List of source regions to plot on the image.

    sky_regions : list of photutils.aperture objects, optional
        List of sky regions to plot on the image.
        Default is None.

    border_width : int, optional
        Width of the border to exclude from the image when calculating the image
        contrast.
        Default is 0.

    bin_y : int, optional
        Binning of y-axis, to preserve real on-sky shape.
        Default is 1.

    Returns
    -------
    None
        This function does not return any value. It plots the regions on the image.

    """

    # Collapse cube in the waevelength dimesion for obtaing a median image
    collapsed_cube = collapse_cube(data_cube)
    cc_shape = collapsed_cube.shape

    # Improve contrast on the image - useful to see faint sources
    vmin, vmax = numpy.nanpercentile(
        collapsed_cube[border_width:cc_shape[0] - border_width,
                       border_width:cc_shape[1] - border_width], (2, 95)
    )
    # Show collapsed image
    plt.imshow(collapsed_cube, vmin=vmin, vmax=vmax)

    # Color of the overlay area
    cmap = mcolors.ListedColormap(["white"])
    alpha = 0.5

    for index, source_reg in enumerate(source_regions):
        det_index = index + 1
        # Plot an overlapped transparent area
        mask = source_reg.to_mask(method="center").to_image(cc_shape)
        mask[mask == 0] = numpy.nan
        plt.imshow(mask, alpha=alpha, cmap=cmap)
        # Plot theoretical region outline
        source_reg.plot(color="white", lw=0.8, ls="--")
        # Plot the region
        plt.text(
            source_reg.positions[0],
            source_reg.positions[1],
            det_index,
            color="white",
            ha="center",
            va="center",
            fontsize=12,
            path_effects=[withStroke(linewidth=2, foreground="black")],
        )

    if sky_regions is not None:
        for sky_reg in sky_regions:
            # Plot an overlapped transparent area
            mask = sky_reg.to_mask(method="center").to_image(cc_shape)
            mask[mask == 0] = numpy.nan
            plt.imshow(mask, alpha=alpha, cmap=cmap)
            # Plot theoretical region outline
            sky_reg.plot(color="white", lw=0.8, ls="--")

    # Set axis proportions to right values
    plt.gca().set_aspect(0.5 * bin_y)


def read_cube_data(cube_path, get_dq=False, get_tell=False, get_extinct=False):
    cube_data = {
        "sci": None,
        "sci_hdr": None,
        "var": None,
        "var_hdr": None,
        "wcs": None,
        "dq": None,
        "dq_hdr": None,
        "binning_x": None,
        "binning_y": None,
        "wave": None,
        "tell": None,
        "ext": None,
        "ext_hdr": None,
    }

    if cube_path is not None:
        cube = fits.open(cube_path, mode='readonly')
        sci = cube[0].data
        sci_hdr = cube[0].header
        var = cube["VAR"].data
        var_hdr = cube["VAR"].header
        if get_dq:
            if "DQ" in cube:
                dq = cube["DQ"].data
                dq_hdr = cube["DQ"].header
            else:
                print("WARNING: No DQ extension in input cube.")
                dq = None
                dq_hdr = None
        else:
            dq = None
            dq_hdr = None

        if "WAVELENGTH" in cube:
            wave = cube["WAVELENGTH"].data
        else:
            wave = None

        if get_tell:
            if "TELLURICMODEL" in cube:
                tell = cube["TELLURICMODEL"].data
            else:
                tell = None
        else:
            tell = None

        if get_extinct:
            if "EXTINCTION" in cube:
                ext = cube["EXTINCTION"].data
                ext_hdr = cube["EXTINCTION"].header
            else:
                ext = None
                ext_hdr = None
        else:
            ext = None
            ext_hdr = None

        cube.close()

        wcs = WCS(sci_hdr).celestial
        binning_x, binning_y = [int(x) for x in sci_hdr["CCDSUM"].split()]
        cube_data["sci"] = sci
        cube_data["sci_hdr"] = sci_hdr
        cube_data["var"] = var
        cube_data["var_hdr"] = var_hdr
        cube_data["dq"] = dq
        cube_data["dq_hdr"] = dq_hdr
        cube_data["wcs"] = wcs
        cube_data["binning_x"] = binning_x
        cube_data["binning_y"] = binning_y
        cube_data["wave"] = wave
        cube_data["tell"] = tell
        cube_data["ext"] = ext
        cube_data["ext_hdr"] = ext_hdr

    return cube_data


def detect_extract_and_save(
    blue_cube_path=None,
    red_cube_path=None,
    output_dir=None,
    ns=False,
    subns=False,
    plot_path="detected_sources_plot.png",
    nsources=3,
    sigma_threshold=3,
    det_fwhm=2.0,
    extraction_method="aperture",
    r_arcsec=2,
    sky_method="annulus",
    sky_stat="wmean",
    bkg_in_factor=3,
    bkg_out_factor=4,
    border_width=2,
    ns_skysub=True,
    plot=True,
    get_dq=True,
    get_tell=True,
    get_extinct=True,
    debug=False,
):
    """
    Detects sources in the input cubes, extracts and saves their spectra. Optionally,
    it can plot the extracted sources and the sky regions around them.

    Parameters:
    -----------
    blue_cube_path : str, optional
        Path to the blue cube file.
    red_cube_path : str, optional
        Path to the red cube file.
    output_dir : str, optional
        Directory where the extracted spectra will be saved.
    ns : bool, optional
        Flag indicating whether this is a nod & shuffle exposure.
        Default: False.
    subns : bool, optional
        Flag indicating whether this is a sub-aperture nod & shuffle exposure.
        Default: False.
    plot_path : str, optional
        Path to save the plot.
        Default: "detected_sources_plot.png".

    Detection Parameters:
    ---------------------
    nsources : int, optional
        Number of sources above the threshold that will be found and extracted (in
        descending brightness).
        Default: 3.
    sigma_threshold : float, optional
        Number of sigma-clipped standard deviations above the median of the collapsed
        cube to use for source-finding.
        Default: 3.
    border_width : int, optional
        Width of the border to be excluded from the statistics and source-finding.
        Default: 2.
    det_fwhm : float, optional
        Full-width at half maximum (in arcsec) of the Gaussian convolution kernal used
        for source-finding.
        Default: 2.0.

    Extraction Parameters:
    ----------------------
    extraction_method : str, optional
        Method to extract source spectra from datacubes. Options: "aperture".
        Default: "aperture".
    r_arcsec : float, optional
        Radius of the circular extraction aperture in arcseconds.
        Default: 2.
    sky_method : str, optional
        Method to determine sky background. Options: "annulus", "same_slice".
        Default: "annulus".
    sky_stat : str, optional
        Statistical method for calculating sky spectrum. Options: "wmean", "median".
        Default: "wmean".
    bkg_in_factor : float, optional
        Inner radius of background annulus: bkg_in_factor * r_arcsec.
        Default: 3.
    bkg_out_factor : float, optional
        Outer radius of background annulus: bkg_out_factor * r_arcsec.
        Default: 4.
    ns_skysub : bool, optional
        Flag indicating whether to force sky subtraction even if a nod & shuffle of sub-frame nod & shuffle exposure.
        Default: True.
    get_dq : bool, optional
        Whether to include the DQ extension in the output.
        Default: True.
    get_tell : bool, optional
        Whether to include the telluric model extension in the output.
        Default: True.
    get_extinct : bool, optional
        Whether to include the corrected extinction extension in the output.
        Default: True.


    General Parameters:
    -------------------
    plot : bool, optional
        Flag indicating whether to generate a plot of the detected sources.
        Default: True.
    debug : bool, optional
        Whether to report the parameters used in this function call.
        Default: False.

    Returns:
    --------
    None
    """
    if debug:
        print(arguments())

    # reading the data from cubes
    # Blue arm
    blue_cube_data = read_cube_data(blue_cube_path, get_dq=get_dq, get_tell=get_tell,
                                    get_extinct=get_extinct)
    blue_sci = blue_cube_data["sci"]
    blue_sci_hdr = blue_cube_data["sci_hdr"]
    blue_var = blue_cube_data["var"]
    blue_var_hdr = blue_cube_data["var_hdr"]
    blue_dq = blue_cube_data["dq"]
    blue_dq_hdr = blue_cube_data["dq_hdr"]
    blue_wcs = blue_cube_data["wcs"]
    blue_ext = blue_cube_data["ext"]
    blue_ext_hdr = blue_cube_data["ext_hdr"]

    if blue_cube_data["sci"] is not None:
        binning_x = blue_cube_data["binning_x"]
        binning_y = blue_cube_data["binning_y"]
        objname = extract_object_name(blue_sci_hdr)

    # Red arm
    red_cube_data = read_cube_data(red_cube_path, get_dq=get_dq, get_tell=get_tell,
                                   get_extinct=get_extinct)
    red_sci = red_cube_data["sci"]
    red_sci_hdr = red_cube_data["sci_hdr"]
    red_var = red_cube_data["var"]
    red_var_hdr = red_cube_data["var_hdr"]
    red_dq = red_cube_data["dq"]
    red_dq_hdr = red_cube_data["dq_hdr"]
    red_wcs = red_cube_data["wcs"]
    red_tell = red_cube_data["tell"]
    red_ext = red_cube_data["ext"]
    red_ext_hdr = red_cube_data["ext_hdr"]

    if red_cube_data["sci"] is not None:
        binning_x = red_cube_data["binning_x"]
        binning_y = red_cube_data["binning_y"]
        objname = extract_object_name(red_sci_hdr)

    # Calculate pixel scale from binning
    pixel_scale_x = binning_x  # arcsec/pix
    pixel_scale_y = binning_y / 2.0  # arcsec/pix
    fine_axis = pixel_scale_x if pixel_scale_x <= pixel_scale_y else pixel_scale_y  # arcsec/pix
    coarse_axis = pixel_scale_y if pixel_scale_x <= pixel_scale_y else pixel_scale_x  # arcsec/pix

    det_fwhm_pix = det_fwhm / fine_axis

    # Sky treatment
    if ns or subns:
        sky_sub = True if ns_skysub else False
    else:
        sky_sub = True

    # Average all the data for detecting the source
    collapsed_cube = collapse_cube(blue_sci, red_sci)

    # Automatic source detection in the collapsed cubes (red + blue)

    # Avoid the edges of size border_width on the statistics
    this_cube_path = red_cube_path if red_cube_path is not None else blue_cube_path
    xmin = 0 if is_halfframe(this_cube_path) else border_width
    xmax = collapsed_cube.shape[1] if is_halfframe(this_cube_path) else -border_width
    collapsed_cube_no_edge = collapsed_cube[
        border_width:-border_width, xmin:xmax
    ]
    _, median, std = sigma_clipped_stats(collapsed_cube_no_edge, sigma=3.0)
    threshold = median + (sigma_threshold * std)

    # Automatic source detection in the collapsed cubes (red + blue)
    finder = DAOStarFinder(threshold,
                           fwhm=det_fwhm_pix,
                           brightest=nsources,
                           exclude_border=False,
                           ratio=numpy.clip(fine_axis / float(coarse_axis),
                                            a_min=0.05, a_max=1.0),
                           theta=(90.0 if pixel_scale_y == fine_axis else 0.0),
                           sharplo=0.3,
                           sharphi=1.0,
                           roundlo=-2.0 / fine_axis,
                           roundhi=2.0 / fine_axis,
                           min_separation=3)

    try:
        detections = finder(collapsed_cube_no_edge)
    except ValueError:
        detections = None

    if detections is None:
        print(f"No positive sources found in {blue_cube_path} / {red_cube_path}")
    else:
        fcol = Column(name='flux_sense', data=numpy.ones(len(detections),
                                                         dtype='float32'))
        detections.add_column(fcol)

    if subns:
        try:
            det2 = finder(-1.0 * collapsed_cube_no_edge)
        except ValueError:
            det2 = None

        if det2 is None:
            print(f"No negative sources found in {blue_cube_path} / {red_cube_path}")
        else:
            fcol = Column(name='flux_sense', data=-1. * numpy.ones(len(det2),
                                                                   dtype='float32'))
            det2.add_column(fcol)
            if detections:
                detections = vstack([detections, det2])
            else:
                detections = det2

    if detections is not None:
        # Sort detections as brighter peaks first
        detections.sort("flux", reverse=True)

        # Detected sources positions
        positions = numpy.transpose((detections["xcentroid"], detections["ycentroid"]))

        # Account for any trimmed border
        positions = positions + numpy.array([xmin, border_width])

        # Set the apertures
        a = r_arcsec / fine_axis
        b = r_arcsec / coarse_axis

        # Creates source regions in the detected positions
        source_regions = EllipticalAperture(positions, a=a, b=b, theta=numpy.radians((90.0 if pixel_scale_y == fine_axis else 0.0)))

        if sky_sub:
            if sky_method == "same_slice":
                source_mask = source_regions.to_mask(method='center')
                total_mask = numpy.zeros_like(collapsed_cube)
                for sm in range(len(source_mask)):
                    total_mask += source_mask[sm].to_image(collapsed_cube.shape)

                sky_positions = positions.copy()
                this_a = a  # fine axis
                this_b = b  # coarse axis
                orig_y = a if pixel_scale_y == fine_axis else b
                for sp, op in zip(sky_positions, positions):
                    stop_shifting = False
                    # Reduce radial size
                    for rscale in [1., 0.75, 0.5]:
                        this_a = a * rscale  # fine axis
                        this_b = b * rscale  # coarse axis
                        this_y = this_a if pixel_scale_y == fine_axis else this_b
                        # Shift one slice either direction
                        for xoffset in [0, -1, 1]:
                            # Shift beyond science aperture extent, even if away from centre
                            for yoffset in [1.1, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, -1.1, -1.25, -1.5, 2.5]:
                                sp[0] = op[0] + xoffset
                                if op[1] + 1 > collapsed_cube.shape[0] / 2.0:
                                    sp[1] = op[1] - orig_y - this_y * yoffset
                                else:
                                    sp[1] = op[1] + orig_y + this_y * yoffset
                                if ((sp[0] < 1 or sp[0] > collapsed_cube.shape[1] - 2)
                                        or (sp[1] < this_y or sp[1] > collapsed_cube.shape[0] - 1 - this_y)):
                                    continue
                                test_shape = EllipticalAperture(sp, a=this_a, b=this_b, theta=numpy.radians((90.0 if pixel_scale_y == fine_axis else 0.0))).to_mask(method='center').to_image(collapsed_cube.shape)
                                if numpy.sum(test_shape * total_mask) == 0:
                                    stop_shifting = True
                                    break
                            if stop_shifting:
                                break
                        if stop_shifting:
                            break
                    if not stop_shifting:
                        print("WARNING: Failed to find safe sky region for all science apertures "
                              "with method <same_slice>, falling back to sky_method <annulus>.")
                        sky_method = "annulus"
                        break
                if stop_shifting:
                    sky_regions = EllipticalAperture(sky_positions, a=this_a, b=this_b, theta=numpy.radians((90.0 if pixel_scale_y == fine_axis else 0.0)))

            if sky_method == "annulus":
                a_in = a * bkg_in_factor
                a_out = a * bkg_out_factor

                b_in = b * bkg_in_factor
                b_out = b * bkg_out_factor

                sky_regions = EllipticalAnnulus(
                    positions,
                    a_in=a_in,
                    a_out=a_out,
                    b_in=b_in,
                    b_out=b_out,
                    theta=numpy.radians((90.0 if pixel_scale_y == fine_axis else 0.0))
                )
        else:
            sky_regions = None

        # Flux extraction at all the wavelengths
        for index, (detection, source_reg) in enumerate(zip(detections,
                                                            source_regions)):
            if sky_regions is not None:
                sky_reg = sky_regions[index]
            else:
                sky_reg = None
            det_index = index + 1

            # Extracting blue cube
            if blue_cube_path is not None:
                extract_and_save(
                    blue_cube_path,
                    blue_sci * detection['flux_sense'],
                    blue_var,
                    source_reg,
                    det_index,
                    sky_reg,
                    output_dir,
                    blue_sci_hdr,
                    blue_var_hdr,
                    extraction_method=extraction_method,
                    sky_method=sky_method,
                    sky_stat=sky_stat,
                    dq_data=blue_dq,
                    dq_hdr=blue_dq_hdr,
                    wave_data=blue_cube_data["wave"],
                    ext=blue_ext,
                    ext_hdr=blue_ext_hdr,
                )

            # Extracting red cube
            if red_cube_path is not None:
                extract_and_save(
                    red_cube_path,
                    red_sci * detection['flux_sense'],
                    red_var,
                    source_reg,
                    det_index,
                    sky_reg,
                    output_dir,
                    red_sci_hdr,
                    red_var_hdr,
                    extraction_method=extraction_method,
                    sky_method=sky_method,
                    sky_stat=sky_stat,
                    dq_data=red_dq,
                    dq_hdr=red_dq_hdr,
                    wave_data=red_cube_data["wave"],
                    tell_data=red_tell,
                    ext=red_ext,
                    ext_hdr=red_ext_hdr,
                )

        if plot:
            plt.suptitle(objname)
            # Plot Red
            ax1 = plt.subplot(1, 2, 2, projection=red_wcs)
            plot_arm(
                ax=ax1,
                cube_path=red_cube_path,
                sci=red_sci,
                title="Red arm",
                source_regions=source_regions,
                sky_regions=sky_regions,
                border_width=border_width,
                bin_y=binning_y,
            )
            # Catch rotations near 90deg to adapt axis labels and ticks
            if red_sci_hdr is not None and "TELPAN" in red_sci_hdr \
                    and numpy.abs(numpy.mod(red_sci_hdr["TELPAN"], 180) - 90.) < 5.0:
                ax1.coords[0].set_ticklabel_position('l')
                ax1.set_ylabel('Right Ascension')
                ax1.coords[1].set_ticklabel_position('b')
                ax1.set_xlabel('Declination')

            # Plot Blue
            ax0 = plt.subplot(1, 2, 1, projection=blue_wcs)
            plot_arm(
                ax=ax0,
                cube_path=blue_cube_path,
                sci=blue_sci,
                title="Blue arm",
                source_regions=source_regions,
                sky_regions=sky_regions,
                border_width=border_width,
                bin_y=binning_y,
            )
            # Catch rotations near 90deg to adapt axis labels and ticks
            if blue_sci_hdr is not None and "TELPAN" in blue_sci_hdr \
                    and numpy.abs(numpy.mod(blue_sci_hdr["TELPAN"], 180) - 90.) < 5.0:
                ax0.coords[0].set_ticklabel_position('l')
                ax0.set_ylabel('Right Ascension')
                ax0.coords[1].set_ticklabel_position('b')
                ax0.set_xlabel('Declination')

            # Add PA to plot
            if blue_sci_hdr is not None and "ROTREF" in blue_sci_hdr\
                    and blue_sci_hdr["ROTREF"] == "POSITION_ANGLE" \
                    and "TELPAN" in blue_sci_hdr:
                plt.gcf().text(0.05, 0.05, r"PA = {:.1f}$^\circ$".format(blue_sci_hdr["TELPAN"]))
            elif red_sci_hdr is not None and "ROTREF" in red_sci_hdr\
                    and red_sci_hdr["ROTREF"] == "POSITION_ANGLE" \
                    and "TELPAN" in red_sci_hdr:
                plt.gcf().text(0.05, 0.05, r"PA = {:.1f}$^\circ$".format(red_sci_hdr["TELPAN"]))

            plt.tight_layout()
            fig_output = os.path.join(output_dir, plot_path)
            plt.savefig(fig_output, bbox_inches="tight", dpi=300)
            plt.close('all')


# TODO Replace with a proper SpecUtils loader


class SingleSpec(object):
    """
    Class representing a single spectrum for analysis
    """

    def __init__(self, fitsFILE):
        f = fits.open(fitsFILE)
        self.flux = f[0].data
        self.header = f[0].header
        if "WAVELENGTH" in f:
            self.wl = f["WAVELENGTH"].data
        else:
            self.wl = (
                (
                    numpy.arange(self.header["NAXIS1"], dtype="d")
                ) - self.header["CRPIX1"] + 1
            ) * self.header["CDELT1"] + self.header["CRVAL1"]
        self.min_wl = min(self.wl)
        self.max_wl = max(self.wl)
        # Temporary
        self.fluxvar = f["VAR"].data
        f.close()
        return


def plot_1D_spectrum(spec_path, plot_dir):
    """
    Plot a 1D spectrum with error bars and save the plot as a PNG file in the provided
    directory.

    Parameters
    ----------
    spec_path : str
        The path to the spectrum file.
    plot_dir : str
        The directory where the plot will be saved.

    Returns
    -------
    None
        This function does not return anything.

    """

    spec = SingleSpec(spec_path)
    flux = spec.flux
    wl = spec.wl
    fluxvar = spec.fluxvar
    # Calculate the error as the square root of the flux variance
    flux_error = numpy.sqrt(fluxvar)

    # Plot the spectrum with error bars
    spec_name = os.path.basename(spec_path)
    plot_name = spec_name.replace(".fits", ".png")
    plot_path = os.path.join(plot_dir, plot_name)
    detection_name = extract_detection_name(spec_name)
    objname = extract_object_name(spec.header)

    # Plot the error bars
    plt.errorbar(
        wl,
        flux,
        yerr=flux_error,
        fmt="none",
        ecolor="grey",
        elinewidth=0.5,
        capsize=1.5,
    )

    # Plot the step plot for the spectrum values
    plt.step(wl, flux, where="mid", color="b")

    # Customize the plot
    good = numpy.abs(flux) > 1E-37
    ylims = [numpy.nanmin(flux[good]) - numpy.nanmedian(flux_error[good]),
             numpy.nanmax(flux[good]) + numpy.nanmedian(flux_error[good])]
    # Catch bad values
    if numpy.any(numpy.isnan(ylims)) or numpy.any(numpy.isinf(ylims)):
        ylims = [0, 1]
    plt.ylim(ylims)
    plt.title(f"{spec_name} \n{objname}{detection_name}")
    plt.xlabel("Wavelength (Å)")
    plt.ylabel("Flux")
    plt.grid(True)

    plt.savefig(plot_path, dpi=300)
    plt.close('all')
