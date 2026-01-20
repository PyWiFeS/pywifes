import astropy.io.fits as fits
import numpy
import scipy.sparse as sp

from pywifes.wifes_utils import arguments


def calculate_wavelength_array(
    blue_CRVAL, blue_CDELT, blue_CRPIX, red_CRVAL, red_CDELT, red_CRPIX, red_NAXIS,
    wstep
):
    """
    Calculates the wavelength array for the merged blue and red spectra, using the
    wavelength step of the blue spectrum.

    Parameters
    ----------
    blue_CRVAL : float
        Wavelength of the reference pixel for the blue spectrum.
    blue_CDELT : float
        Wavelength increment per pixel for the blue spectrum.
    blue_CRPIX : float
        Reference pixel for the blue spectrum.
    red_CRVAL : float
        Wavelength of the reference pixel for the red spectrum.
    red_CDELT : float
        Wavelength increment per pixel for the red spectrum.
    red_CRPIX : float
        Reference pixel for the red spectrum.
    red_NAXIS : int
        Number of pixels in the red spectrum.
    wstep : float
        Wavelength increment per pixel for the output spectrum.

    Returns
    -------
    Wavelength array calculated from the given parameters.
    """
    # Calculate the number of wavelength points needed
    nwl = int(numpy.ceil((red_CRVAL + red_CDELT * (red_NAXIS - red_CRPIX)
                          - (blue_CRVAL - (blue_CRPIX - 1) * blue_CDELT)) / wstep))
    # Generate the wavelength array
    wl = (numpy.arange(nwl) - blue_CRPIX + 1) * wstep + blue_CRVAL
    return wl


def is_evenly_sampled(x_original):
    dx = x_original[1:] - x_original[:-1]
    return abs((dx.max() - dx.min()) / numpy.median(dx)) < 1e-3


def a_lanczos(x_original, x, a=2, missing=numpy.nan):
    """
    Returns transformation matrix for Lanczos (sinc) interpolation

    Inputs
        x_original : array
            x-values at which the original function is sampled
        x : array
            x-values to which we want to interpolate
        a : int, optional
            integer order of the kernel. Default is 2.
        missing : float/NaN
            value to use for missing data (default:  numpy.nan)
    Outputs
        A : array
            sparse representation of the transformation matrix
    """
    if not is_evenly_sampled(x_original):
        print("a_lanczos:  warning -- x_original not evenly sampled!")
        print("Lanczos interpolation may not give good results.")
    # Figure out which x values are in range
    x = numpy.atleast_1d(x)
    ok = numpy.all([x >= x_original[0], x < x_original[-1]], axis=0)
    # Find bins and fractional positions within bins, and interpolate
    new_fractional_pos = numpy.array(
        (len(x_original) - 1) * (x - x_original[0]) / (x_original[-1] - x_original[0])
    )
    new_pos = numpy.array([int(f) for f in new_fractional_pos])
    # Write down the matrix for Lanczos interpolation
    epsilon = numpy.finfo(numpy.double).eps

    # The following line of code results in a deprecation waring
    # A sparce matrix in link list format
    A = sp.lil_matrix((len(x), len(x_original)))
    for i in range(len(x)):
        # If x[i] is in interpolation range...
        if ok[i]:
            # Generate a range of new_pos within the original signal
            ik = range(
                max(0, new_pos[i] - a + 1), min(len(x_original) - 1, new_pos[i] + a + 1)
            )
            # Find L(x-i) for each of these
            xk = numpy.pi * (new_fractional_pos[i] - numpy.array(ik) + epsilon)
            w = a * numpy.sin(xk) * numpy.sin(xk / a) / xk**2
            A[i, ik] = w / w.sum()
        elif x[i] == x_original[-1]:
            A[i, -1] = 1
        else:
            A[i, 0] = missing
    return A


class SingleSpec(object):
    """
    Class representing a single spectrum for analysis
    """

    def __init__(self, fits_path):
        f = fits.open(fits_path)
        self.flux = f[0].data
        self.header = f[0].header
        if "WAVELENGTH" in f:
            self.wl = f["WAVELENGTH"].data
            self.wave_ext = True
        else:
            self.wl = (
                numpy.arange(self.header["NAXIS1"], dtype="d")
                - self.header["CRPIX1"] + 1
            ) * self.header["CDELT1"] + self.header["CRVAL1"]
            self.wave_ext = False
        self.min_wl = min(self.wl)
        self.max_wl = max(self.wl)
        self.fluxvar = f["VAR"].data
        if "DQ" in f:
            self.dq = f["DQ"].data
        else:
            self.dq = None
        if "SKY" in f:
            self.sky = f["SKY"].data
        else:
            self.sky = None
        if "TELLURICMODEL" in f:
            self.tell = f["TELLURICMODEL"].data
        else:
            self.tell = None
        if "EXTINCTION" in f:
            self.ext = f["EXTINCTION"].data
        else:
            self.ext = None
        f.close()


def join_spectra(blueSpec, redSpec, get_dq=False, wstep=None):
    # Generate the resampling matrices
    if blueSpec.wave_ext:
        if wstep is None:
            wstep = min(numpy.abs(blueSpec.wl - numpy.roll(blueSpec.wl, 1)))
        wl = numpy.arange(min(blueSpec.wl), max(redSpec.wl) + wstep, wstep)
        blue_NAXIS1 = blueSpec.wl.size
        red_NAXIS1 = redSpec.wl.size
    else:
        blue_CRVAL1 = blueSpec.header["CRVAL1"]
        blue_CDELT1 = blueSpec.header["CDELT1"]
        blue_CRPIX1 = blueSpec.header["CRPIX1"]
        blue_NAXIS1 = blueSpec.header["NAXIS1"]

        red_CRVAL1 = redSpec.header["CRVAL1"]
        red_CDELT1 = redSpec.header["CDELT1"]
        red_CRPIX1 = redSpec.header["CRPIX1"]
        red_NAXIS1 = redSpec.header["NAXIS1"]

        if wstep is None:
            wstep = blue_CDELT1
        wl = calculate_wavelength_array(
            blue_CRVAL1, blue_CDELT1, blue_CRPIX1, red_CRVAL1, red_CDELT1,
            red_CRPIX1, red_NAXIS1, wstep
        )

    # One does not need to interplolate the blue spectra if the wavelength is
    # set to the blue spectrum. However the code is robust to this.

    AB = a_lanczos(blueSpec.wl, wl, 3).tocsr()
    AR = a_lanczos(redSpec.wl, wl, 3).tocsr()

    # Blue
    flux_B = numpy.array(AB * blueSpec.flux).ravel()
    diag_B = sp.dia_matrix(
        ([blueSpec.fluxvar], [0]), shape=[blue_NAXIS1, blue_NAXIS1]
    ).astype(numpy.float64)
    fluxvar_B = numpy.array((AB * diag_B * AB.T).sum(axis=1)).ravel()

    # Red
    flux_R = numpy.array(AR * redSpec.flux).ravel()
    diag_R = sp.dia_matrix(
        ([redSpec.fluxvar], [0]), shape=[red_NAXIS1, red_NAXIS1]
    ).astype(numpy.float64)
    fluxvar_R = numpy.array((AR * diag_R * AR.T).sum(axis=1)).ravel()

    BUFFER = 10.0

    # Check for wavelngth overlap
    blue_only = numpy.where(wl <= redSpec.min_wl + BUFFER)
    overlap = numpy.where(
        (wl < blueSpec.max_wl - BUFFER) & (wl > redSpec.min_wl + BUFFER)
    )
    red_only = numpy.where(wl >= blueSpec.max_wl - BUFFER)

    # Average the two taking into account the buffer region and weighting

    flux = numpy.zeros(len(flux_B), dtype=numpy.float32)
    fluxVar = numpy.zeros(len(flux_B), dtype=numpy.float32)

    flux[blue_only] = flux_B[blue_only]
    fluxVar[blue_only] = fluxvar_B[blue_only]

    fluxVar[overlap] = 1.0 / (1.0 / fluxvar_B[overlap] + 1.0 / fluxvar_R[overlap])
    flux[overlap] = (
        flux_B[overlap] / fluxvar_B[overlap] + flux_R[overlap] / fluxvar_R[overlap]
    ) * fluxVar[overlap]

    flux[red_only] = flux_R[red_only]
    fluxVar[red_only] = fluxvar_R[red_only]

    # Ensure no bad effects from Lanczos interpolation.
    fluxVar = numpy.abs(fluxVar)

    if get_dq:
        dq = numpy.zeros(len(flux_B), dtype=numpy.int16)
        dq_B = numpy.array(AB * blueSpec.dq).ravel()
        dq_R = numpy.array(AR * redSpec.dq).ravel()
        dq_B[numpy.isnan(dq_B)] = 1
        dq_R[numpy.isnan(dq_R)] = 1
        dq[blue_only] = dq_B[blue_only]
        dq[overlap] = numpy.amin((dq_B[overlap], dq_R[overlap]), axis=0)
        dq[red_only] = dq_R[red_only]

        dq[dq > 0.1] = 1
        dq[dq < 0] = 0
    else:
        dq = None

    if blueSpec.sky is not None and redSpec.sky is not None:
        sky = numpy.zeros(len(flux_B), dtype=numpy.float32)
        sky_B = numpy.array(AB * blueSpec.sky).ravel()
        sky_R = numpy.array(AR * redSpec.sky).ravel()
        sky[blue_only] = sky_B[blue_only]
        sky[overlap] = numpy.nanmean((sky_B[overlap], sky_R[overlap]), axis=0)
        sky[red_only] = sky_R[red_only]
    else:
        sky = None

    if redSpec.tell is not None:
        tell = numpy.ones(len(flux_R), dtype=numpy.float32)
        tell_R = numpy.array(AR * redSpec.tell).ravel()
        tell[red_only] = tell_R[red_only]
        tell[tell > 1] = 1
        tell[tell < 0] = 0
    else:
        tell = None

    if blueSpec.ext is not None and redSpec.ext is not None:
        ext = numpy.ones(len(flux_B), dtype=numpy.float32)
        ext_B = numpy.array(AB * blueSpec.ext).ravel()
        ext_R = numpy.array(AR * redSpec.ext).ravel()
        ext[blue_only] = ext_B[blue_only]
        ext[overlap] = numpy.nanmean((ext_B[overlap], ext_R[overlap]), axis=0)
        ext[red_only] = ext_R[red_only]
    else:
        ext = None

    return wstep, min(wl), flux, fluxVar, dq, sky, tell, ext


def splice_spectra(blue_spec_path, red_spec_path, output_path, get_dq=False,
                   wstep=None):
    """
    Main routine to splice two spectra together.

    Parameters
    ----------
    blue_spec_path : str
        Path to the blue spectrum FITS file.
    red_spec_path : str
        Path to the red spectrum FITS file.
    output_path : str
        Path to save the spliced spectrum FITS file.
    get_dq : bool, optional
        Whether to include the DQ extension in the output.
        Default: False.
    wstep : float, optional
        User-defined wavelength step of output spectra.
        Default: None (minimum blue wavelength step).

    Returns
    -------
    None

    Notes
    -----
    This function reads two spectra from FITS files, joins them together, and saves the
    spliced spectrum to a new FITS file.

    The splicing process involves combining the flux data from the blue and red spectra,
    and updating the header information of the red spectrum to match the blue spectrum.

    The spliced spectrum is saved as a FITS file with the provided output path.

    Examples
    --------
    splice_spectra('blue_spectrum.fits', 'red_spectrum.fits', 'spliced_spectrum.fits')
    """
    # Create to instances of SingleSpec
    blueSpec = SingleSpec(blue_spec_path)
    redSpec = SingleSpec(red_spec_path)

    # Join the spectra
    print("Splicing spectra.")
    wstep_out, wave_min, flux, fluxVar, dq, sky, tell, ext = join_spectra(
        blueSpec, redSpec, get_dq=get_dq, wstep=wstep
    )

    # Write out the results
    # Use the header in red arm to start with
    # Add additional blue CCD keywords as required
    hdulist = fits.HDUList(fits.PrimaryHDU())
    hdulist[0].data = flux.astype("float32", casting="same_kind")
    hdulist[0].scale("float32")

    hdulist[0].header = redSpec.header.copy()
    if "EXPTIME" in blueSpec.header and \
            blueSpec.header["EXPTIME"] != hdulist[0].header["EXPTIME"]:
        hdulist[0].header["EXPTRED"] = (hdulist[0].header["EXPTIME"],
                                        "Exposure time in red arm")
        hdulist[0].header["EXPTBLUE"] = (blueSpec.header["EXPTIME"],
                                         "Exposure time in blue arm")
        hdulist[0].header.remove("EXPTIME")
    wave_ref = "WAVE" if "PYWWVREF" in hdulist[0].header and hdulist[0].header["PYWWVREF"] == "VACUUM" else "AWAV"
    hdulist[0].header["CRPIX1"] = 1
    hdulist[0].header["CRVAL1"] = wave_min
    hdulist[0].header["CDELT1"] = wstep_out
    hdulist[0].header["CTYPE1"] = wave_ref
    hdulist[0].header["CUNIT1"] = "Angstrom"

    hdr_fluxvar = fits.Header()
    hdr_fluxvar["EXTNAME"] = "VAR"
    hdr_fluxvar["CRPIX1"] = 1
    hdr_fluxvar["CRVAL1"] = wave_min
    hdr_fluxvar["CDELT1"] = wstep_out
    hdr_fluxvar["CTYPE1"] = wave_ref
    hdr_fluxvar["CUNIT1"] = "Angstrom"
    hdr_fluxvar["BUNIT"] = "(Flux)^2"

    hdu_fluxvar = fits.ImageHDU(data=fluxVar.astype("float64", casting="same_kind"),
                                header=hdr_fluxvar)
    hdu_fluxvar.scale("float64")
    hdulist.append(hdu_fluxvar)

    if get_dq:
        hdr_dq = fits.Header()
        hdr_dq["EXTNAME"] = "DQ"
        hdr_dq["CRPIX1"] = 1
        hdr_dq["CRVAL1"] = wave_min
        hdr_dq["CDELT1"] = wstep_out
        hdr_dq["CTYPE1"] = wave_ref
        hdr_dq["CUNIT1"] = "Angstrom"

        hdu_dq = fits.ImageHDU(data=dq.astype("int16", casting="unsafe"),
                               header=hdr_dq)
        hdu_dq.scale('int16')
        hdulist.append(hdu_dq)

    if sky is not None:
        hdr_sky = fits.Header()
        hdr_sky["EXTNAME"] = "SKY"
        hdr_sky["CRPIX1"] = 1
        hdr_sky["CRVAL1"] = wave_min
        hdr_sky["CDELT1"] = wstep_out
        hdr_sky["CTYPE1"] = wave_ref
        hdr_sky["CUNIT1"] = "Angstrom"

        hdu_sky = fits.ImageHDU(data=sky.astype("float32", casting="same_kind"),
                                header=hdr_sky)
        hdu_sky.scale("float32")
        hdulist.append(hdu_sky)

    if tell is not None:
        hdr_tell = fits.Header()
        hdr_tell["EXTNAME"] = "TELLURICMODEL"
        hdr_tell["CRPIX1"] = 1
        hdr_tell["CRVAL1"] = wave_min
        hdr_tell["CDELT1"] = wstep_out
        hdr_tell["CTYPE1"] = wave_ref
        hdr_tell["CUNIT1"] = "Angstrom"

        tell_data = tell.astype("float32", casting="same_kind")
        tell_data[tell_data > 1] = 1
        tell_data[tell_data < 0] = 0
        hdu_tell = fits.ImageHDU(data=tell_data, header=hdr_tell)
        hdu_tell.scale("float32")
        hdulist.append(hdu_tell)

    if ext is not None:
        hdr_ext = fits.Header()
        hdr_ext["EXTNAME"] = "EXTINCTION"
        hdr_ext["CRPIX1"] = 1
        hdr_ext["CRVAL1"] = wave_min
        hdr_ext["CDELT1"] = wstep_out
        hdr_ext["CTYPE1"] = wave_ref
        hdr_ext["CUNIT1"] = "Angstrom"
        hdr_ext['COMMENT'] = 'Flux extinction correction applied to reach airmass zero'
        hdr_ext['COMMENT'] = 'NB: the correction _in magnitudes_ scales linearly with airmass'
        if 'AIRMASS' in hdulist[0].header:
            hdr_ext['AIRMASS'] = hdulist[0].header['AIRMASS']
        else:
            hdr_ext['AIRMASS'] = (1.0, 'Airmass assumed equal to 1')
        hdr_ext['BUNIT'] = 'Flux fraction'

        ext_data = ext.astype("float32", casting="same_kind")
        hdu_ext = fits.ImageHDU(data=ext_data, header=hdr_ext)
        hdu_ext.scale("float32")
        hdulist.append(hdu_ext)

    print("Saving spliced spectra.")
    hdulist.writeto(output_path, overwrite=True)
    hdulist.close()


def join_cubes(blue_path, red_path, get_dq=False, wstep=None):

    # Read red data and metadata
    red_hdu = fits.open(red_path)
    red_flux_cube = red_hdu["SCI"].data
    red_header = red_hdu["SCI"].header
    red_fluxvar_cube = red_hdu["VAR"].data
    if get_dq:
        if "DQ" in red_hdu:
            red_dq_cube = red_hdu["DQ"].data
        else:
            red_dq_cube = None
    wave_kw = True
    if "WAVELENGTH" in red_hdu:
        red_wavelength = red_hdu["WAVELENGTH"].data
        red_NAXIS1 = red_wavelength.size
        wave_kw = False
    else:
        red_CRVAL1 = red_header["CRVAL3"]
        red_CDELT1 = red_header["CDELT3"]
        red_CRPIX1 = red_header["CRPIX3"]
        red_NAXIS1 = red_header["NAXIS3"]
        red_wavelength = (
            (numpy.arange(red_NAXIS1) - red_CRPIX1 + 1) * red_CDELT1 + red_CRVAL1
        )

    if "TELLURICMODEL" in red_hdu:
        red_tell_array = red_hdu["TELLURICMODEL"].data
        if red_tell_array.ndim == 1:
            red_tell_array = red_tell_array[numpy.newaxis, :]
        red_tell_array = numpy.transpose(red_tell_array)
    else:
        red_tell_array = None

    if "EXTINCTION" in red_hdu:
        red_ext_array = red_hdu["EXTINCTION"].data
    else:
        red_ext_array = None

    red_hdu.close()
    red_min_wavelength = min(red_wavelength)

    # Read blue data and metadata
    blue_hdu = fits.open(blue_path)
    blue_flux_cube = blue_hdu["SCI"].data
    blue_header = blue_hdu["SCI"].header
    blue_fluxvar_cube = blue_hdu["VAR"].data
    if get_dq:
        if "DQ" in blue_hdu:
            blue_dq_cube = blue_hdu["DQ"].data
        else:
            blue_dq_cube = None
    if "WAVELENGTH" in blue_hdu:
        blue_wavelength = blue_hdu["WAVELENGTH"].data
        blue_NAXIS1 = blue_wavelength.size
        wave_kw = False
    else:
        blue_CRVAL1 = blue_header["CRVAL3"]
        blue_CDELT1 = blue_header["CDELT3"]
        blue_CRPIX1 = blue_header["CRPIX3"]
        blue_NAXIS1 = blue_header["NAXIS3"]
        blue_wavelength = (
            (numpy.arange(blue_NAXIS1) - blue_CRPIX1 + 1) * blue_CDELT1 + blue_CRVAL1
        )

    if "TELLURICMODEL" in blue_hdu:
        blue_tell_array = blue_hdu["TELLURICMODEL"].data
        if blue_tell_array.ndim == 1:
            blue_tell_array = blue_tell_array[numpy.newaxis, :]
        blue_tell_array = numpy.transpose(blue_tell_array)
        if red_tell_array is None:
            red_tell_array = numpy.ones((len(red_wavelength), blue_tell_array.shape[1]), dtype=float)
    else:
        if red_tell_array is None:
            blue_tell_array = None
        else:
            blue_tell_array = numpy.ones((len(blue_wavelength), red_tell_array.shape[1]), dtype=float)

    if "EXTINCTION" in blue_hdu:
        blue_ext_array = blue_hdu["EXTINCTION"].data
    else:
        blue_ext_array = None

    blue_hdu.close()
    blue_max_wavelength = max(blue_wavelength)

    # Define final wavelength grid
    if wave_kw:
        if wstep is None:
            wstep = blue_CDELT1
        wl = calculate_wavelength_array(
            blue_CRVAL1, blue_CDELT1, blue_CRPIX1, red_CRVAL1, red_CDELT1, red_CRPIX1,
            red_NAXIS1, wstep
        )
    else:
        if wstep is None:
            wstep = min(numpy.abs(blue_wavelength - numpy.roll(blue_wavelength, 1)))
        wl = numpy.arange(min(blue_wavelength), max(red_wavelength) + wstep, wstep)

    # One does not need to interplolate the blue spectra if the wavelength is
    # set to the blue spectrum. However the code is robust to this.

    AB = a_lanczos(blue_wavelength, wl, 3).tocsr()
    AR = a_lanczos(red_wavelength, wl, 3).tocsr()

    wave_dim, y_dim, x_dim = numpy.shape(red_flux_cube)
    wave_dim = len(wl)

    flux_cube = numpy.zeros((wave_dim, y_dim, x_dim))
    fluxvar_cube = numpy.zeros((wave_dim, y_dim, x_dim))
    if get_dq:
        dq_cube = numpy.zeros((wave_dim, y_dim, x_dim))
    else:
        dq_cube = None

    # Check for wavelength overlap
    BUFFER = 10.0
    blue_only = numpy.where(wl <= red_min_wavelength + BUFFER)
    overlap = numpy.where(
        (wl < blue_max_wavelength - BUFFER) & (wl > red_min_wavelength + BUFFER)
    )
    red_only = numpy.where(wl >= blue_max_wavelength - BUFFER)

    # Run over every point (i,j) in the spatial plane
    for i in range(y_dim):
        for j in range(x_dim):
            # Red
            red_flux = red_flux_cube[:, i, j]
            flux_R = numpy.array(AR * red_flux).ravel()

            red_fluxvar = red_fluxvar_cube[:, i, j]
            diag_R = sp.dia_matrix(
                ([red_fluxvar], [0]), shape=[red_NAXIS1, red_NAXIS1]
            ).astype(numpy.float64)
            fluxvar_R = numpy.array((AR * diag_R * AR.T).sum(axis=1)).ravel()

            # Blue
            blue_flux = blue_flux_cube[:, i, j]
            flux_B = numpy.array(AB * blue_flux).ravel()

            blue_fluxvar = blue_fluxvar_cube[:, i, j]
            diag_B = sp.dia_matrix(
                ([blue_fluxvar], [0]), shape=[blue_NAXIS1, blue_NAXIS1]
            ).astype(numpy.float64)
            fluxvar_B = numpy.array((AB * diag_B * AB.T).sum(axis=1)).ravel()

            # Average the two taking into account the buffer region and weighting
            flux = numpy.zeros(len(flux_B), float)
            fluxVar = numpy.zeros(len(flux_B), float)

            flux[blue_only] = flux_B[blue_only]
            fluxVar[blue_only] = fluxvar_B[blue_only]

            fluxVar[overlap] = 1.0 / (
                1.0 / fluxvar_B[overlap] + 1.0 / fluxvar_R[overlap]
            )
            flux[overlap] = (
                flux_B[overlap] / fluxvar_B[overlap]
                + flux_R[overlap] / fluxvar_R[overlap]
            ) * fluxVar[overlap]

            flux[red_only] = flux_R[red_only]
            fluxVar[red_only] = fluxvar_R[red_only]
            flux_cube[:, i, j] = flux
            fluxvar_cube[:, i, j] = fluxVar

    # Reshape the cube to the corresponding shape
    wave_dim = len(flux)
    flux_cube = flux_cube.reshape(wave_dim, y_dim, x_dim)
    fluxvar_cube = fluxvar_cube.reshape(wave_dim, y_dim, x_dim)
    if get_dq:
        for i in range(y_dim):
            for j in range(x_dim):
                dq_B = numpy.array(AB * blue_dq_cube[:, i, j]).ravel()
                dq_R = numpy.array(AR * red_dq_cube[:, i, j]).ravel()
                dq_cube[blue_only, i, j] = dq_B[blue_only]
                dq_cube[overlap, i, j] = numpy.amax((dq_B[overlap],
                                                    dq_R[overlap]), axis=0)
                dq_cube[red_only, i, j] = dq_R[red_only]
        dq_cube[numpy.isnan(dq_cube)] = 1
        dq_cube[dq_cube > 0.1] = 1
        dq_cube[dq_cube < 0] = 0

    if red_tell_array is not None:
        tell = numpy.ones((len(flux), red_tell_array.shape[1]), dtype=float)
        for slit in range(tell.shape[1]):
            tell_B = numpy.array(AB * blue_tell_array[:, slit]).ravel()
            tell_R = numpy.array(AR * red_tell_array[:, slit]).ravel()
            tell[blue_only, slit] = tell_B[blue_only]
            tell[overlap, slit] = numpy.mean((tell_B[overlap], tell_R[overlap]), axis=0)
            tell[red_only, slit] = tell_R[red_only]
        tell[tell > 1] = 1
        tell[tell < 0] = 0
    else:
        tell = None

    if blue_ext_array is not None and red_ext_array is not None:
        ext = numpy.ones(len(flux), dtype=float)
        ext_B = numpy.array(AB * blue_ext_array).ravel()
        ext_R = numpy.array(AR * red_ext_array).ravel()
        ext[blue_only] = ext_B[blue_only]
        ext[overlap] = numpy.mean((ext_B[overlap], ext_R[overlap]), axis=0)
        ext[red_only] = ext_R[red_only]
        ext[ext > 1] = 1
        ext[ext < 0] = 0
    else:
        ext = None

    return wstep, min(wl), flux_cube, fluxvar_cube, dq_cube, tell, ext


def splice_cubes(blue_path, red_path, output_path, get_dq=False, wstep=None,
                 debug=False):
    """
    Main function to splice two previously generated cubes from both WiFeS arms
    together.

    Parameters
    ----------
    blue_path : str
        Path to the blue cube FITS file.
    red_path : str
        Path to the red cube FITS file.
    output_path : str
        Path to save the spliced cube FITS file.
    get_dq : bool, optional
        Whether to include the DQ extension in the output.
        Default: False.
    wstep : float, optional
        User-defined wavelength step of output cubes.
        Default: None (minimum blue wavelength step).
    debug : bool, optional
        Whether to report the parameters used in this function call.
        Default: False.

    Returns
    -------
    None

    Notes
    -----
    This function joins the blue and red cubes together, and saves the spliced cube
    to the specified output path. It uses the header information from the red cube
    as a starting point and adds additional blue CCD keywords as required.

    The spliced cube is saved in FITS format with the following extensions:
    - Primary HDU: Contains the spliced flux data.
    - Extension HDU: Contains the variance data.
    - Optional Extension HDU: Contains the data quality data.

    Examples
    --------
    >>> splice_cubes('blue_cube.fits', 'red_cube.fits', 'spliced_cube.fits')
    """
    if debug:
        print(arguments())

    blue_header = fits.getheader(blue_path, 0)
    red_header = fits.getheader(red_path, 0)

    # Join the cubes
    print("Splicing cubes.")
    wstep_out, wave_min, flux, fluxVar, dq, tell, ext = join_cubes(
        blue_path, red_path, get_dq=get_dq, wstep=wstep
    )

    # Write out the results. Use the header in red arm to start with then add
    # additional blue CCD keywords as required
    hdulist = fits.HDUList(fits.PrimaryHDU())
    hdulist[0].data = flux.astype("float32", casting="same_kind")
    hdulist[0].scale("float32")

    hdulist[0].header = red_header
    if "EXPTIME" in blue_header and \
            blue_header["EXPTIME"] != hdulist[0].header["EXPTIME"]:
        hdulist[0].header["EXPTRED"] = (hdulist[0].header["EXPTIME"],
                                        "Exposure time in red arm")
        hdulist[0].header["EXPTBLUE"] = (blue_header["EXPTIME"],
                                         "Exposure time in blue arm")
        hdulist[0].header.remove("EXPTIME")
    wave_ref = "WAVE" if "PYWWVREF" in hdulist[0].header and hdulist[0].header["PYWWVREF"] == "VACUUM" else "AWAV"
    hdulist[0].header["CRPIX3"] = 1
    hdulist[0].header["CRVAL3"] = wave_min
    hdulist[0].header["CDELT3"] = wstep_out
    hdulist[0].header["CTYPE3"] = wave_ref
    hdulist[0].header["CUNIT3"] = "Angstrom"

    hdr_fluxvar = fits.Header()
    hdr_fluxvar["EXTNAME"] = "VAR"
    hdr_fluxvar["CRPIX3"] = 1
    hdr_fluxvar["CRVAL3"] = wave_min
    hdr_fluxvar["CDELT3"] = wstep_out
    hdr_fluxvar["CTYPE3"] = wave_ref
    hdr_fluxvar["CUNIT3"] = "Angstrom"
    hdr_fluxvar["BUNIT"] = "(Flux)^2"

    hdu_fluxvar = fits.ImageHDU(data=fluxVar.astype("float64", casting="same_kind"),
                                header=hdr_fluxvar)
    hdu_fluxvar.scale("float64")
    hdulist.append(hdu_fluxvar)

    if get_dq:
        hdr_dq = fits.Header()
        hdr_dq["EXTNAME"] = "DQ"
        hdr_dq["CRPIX3"] = 1
        hdr_dq["CRVAL3"] = wave_min
        hdr_dq["CDELT3"] = wstep_out
        hdr_dq["CTYPE3"] = wave_ref
        hdr_dq["CUNIT3"] = "Angstrom"

        hdu_dq = fits.ImageHDU(data=dq.astype("int16", casting="unsafe"), header=hdr_dq)
        hdu_dq.scale("int16")
        hdulist.append(hdu_dq)

    if tell is not None:
        hdr_tell = fits.Header()
        hdr_tell["EXTNAME"] = "TELLURICMODEL"
        hdr_tell["CRPIX1"] = 1
        hdr_tell["CRVAL1"] = wave_min
        hdr_tell["CDELT1"] = wstep_out
        hdr_tell["CTYPE1"] = wave_ref
        hdr_tell["CUNIT1"] = "Angstrom"

        tell_data = tell.astype("float32", casting="same_kind")
        tell_data[tell_data > 1] = 1
        tell_data[tell_data < 0] = 0
        hdu_tell = fits.ImageHDU(data=tell_data, header=hdr_tell)
        hdu_tell.scale("float32")
        hdulist.append(hdu_tell)

    if ext is not None:
        hdr_ext = fits.Header()
        hdr_ext["EXTNAME"] = "EXTINCTION"
        hdr_ext["CRPIX1"] = 1
        hdr_ext["CRVAL1"] = wave_min
        hdr_ext["CDELT1"] = wstep_out
        hdr_ext["CTYPE1"] = wave_ref
        hdr_ext["CUNIT1"] = "Angstrom"
        hdr_ext['COMMENT'] = 'Flux extinction correction applied to reach airmass zero'
        hdr_ext['COMMENT'] = 'NB: the correction _in magnitudes_ scales linearly with airmass'
        if 'AIRMASS' in hdulist[0].header:
            hdr_ext['AIRMASS'] = hdulist[0].header['AIRMASS']
        else:
            hdr_ext['AIRMASS'] = (1.0, 'Airmass assumed equal to 1')
        hdr_ext['BUNIT'] = 'Flux fraction'

        ext_data = ext.astype("float32", casting="same_kind")
        hdu_ext = fits.ImageHDU(data=ext_data, header=hdr_ext)
        hdu_ext.scale("float32")
        hdulist.append(hdu_ext)

    print("Saving spliced cube.")
    hdulist.writeto(output_path, overwrite=True)
    hdulist.close()
