from astropy.io import fits as pyfits
from inspect import getargvalues, stack
import numpy
import re
import glob
import os
import shutil
import pyjson5
import datetime


def arguments():
    """
    Report current functions source file and as-called arguments.

    Adapted from http://kbyanc.blogspot.com/2007/07/python-aggregating-function-arguments.html
    """
    this_stack = stack()
    posname, kwname, args = getargvalues(this_stack[1][0])[-3:]
    posargs = args.pop(posname, [])
    args.update(dict(args.pop(kwname, [])))
    args_string = ", ".join(
        [
            str(a).strip("'")
            + (f"='{args[a]}'" if isinstance(args[a], str) else f"={args[a]}")
            for a in args
        ]
    ).rstrip(", ")
    if posargs:
        posargs_string = ", " + ", ".join(
            [
                str(a).strip("'")
                + (f"='{args[a]}'" if isinstance(args[a], str) else f"={args[a]}")
                for a in posargs
            ]
        ).rstrip(", ")
    else:
        posargs_string = ""
    return f"In file {this_stack[1][1]}, calling function {this_stack[1][3]}({args_string}{posargs_string})"


def fits_scale_from_bitpix(bitpix):
    """
    Map common BITPIX header value to astropy.io.fits 'scale' argument.
    """
    if bitpix == -64:
        return "float64"
    elif bitpix == -32:
        return "float32"
    elif bitpix == 32:
        return "int32"
    elif bitpix == 16:
        return "int16"
    elif bitpix == 8:
        return "uint8"
    else:
        print(f"BITPIX value {bitpix} has no mapping, using current data type")
        return None


# ------------------------------------------------------------------------
# high-level functions to check if an observation is half-frame or N+S
def is_halfframe(inimg, data_hdu=0):
    """
    Report whether this exposure (filename or HDUList) is a half-frame (a.k.a. Stellar mode) image.
    """
    if isinstance(inimg, str):
        extnum = data_hdu + 1 if re.search(".fz", inimg) else data_hdu
        header = pyfits.getheader(inimg, ext=extnum)
        detsec = header["DETSEC"]
    elif isinstance(inimg, pyfits.hdu.hdulist.HDUList):
        f = inimg
        detsec = f[data_hdu].header["DETSEC"]
    else:
        raise ValueError(
            f"is_halfframe takes filepath or HDUList as inputs, not type {type(inimg)}"
        )
    ystart, ystop = [int(pix) for pix in detsec.split(",")[1].rstrip("]").split(":")]
    return ystop - ystart + 1 == 2056


def is_nodshuffle(inimg):
    """
    Report whether this exposure is a Nod & Shuffle frame.
    """
    if isinstance(inimg, pyfits.header.Header):
        header = inimg
    elif isinstance(inimg, str):
        extnum = 1 if re.search(".fz", inimg) else 0
        header = pyfits.getheader(inimg, ext=extnum)
    elif isinstance(inimg, pyfits.hdu.hdulist.HDUList):
        extnum = 1 if re.search(".fz", inimg.filename()) else 0
        header = inimg[extnum].header
    else:
        raise TypeError(
            f"The is_nodshuffle function takes headers, filenames, or HDULists; given {type(inimg)}"
        )
    ns = header["WIFESOBS"]
    return ns == "NodAndShuffle"


def is_standard(inimg, data_hdu=0):
    """
    Report whether this exposure is a Nod & Shuffle frame.
    """
    f = pyfits.open(inimg)
    imtype = f[data_hdu].header["IMAGETYP"]
    f.close()
    return imtype.upper() == "STANDARD"


def is_subnodshuffle(inimg, data_hdu=0):
    """
    Report whether this exposure is a SubNodAndShuffle frame.
    """
    f = pyfits.open(inimg)
    sns = f[data_hdu].header["WIFESOBS"]
    f.close()
    return sns == "SubNodAndShuffle"


def is_taros(inimg):
    """
    Report whether this exposure was taken with TAROS.
    Useful for determining whether last slit of half-frame readouts is truncated.
    """
    if isinstance(inimg, pyfits.header.Header):
        header = inimg
    elif isinstance(inimg, str):
        extnum = 1 if re.search(".fz", inimg) else 0
        header = pyfits.getheader(inimg, ext=extnum)
    elif isinstance(inimg, pyfits.hdu.hdulist.HDUList):
        extnum = 1 if re.search(".fz", inimg.filename()) else 0
        header = inimg[extnum].header
    else:
        raise TypeError(
            f"The is_taros function takes headers, filenames, or HDULists; given {type(inimg)}"
        )
    return "OBSEQID" in header


def hl_envelopes_idx(s, dmin=1, dmax=1, split=False, as_bool=False):
    """
    Inputs:

    s : 1d-array
        data signal from which to extract high and low envelopes

    dmin, dmax : int, optional
        size of chunks, use this if the size of the input signal is too big

    split : bool, optional
        if True, split the signal in half along its mean, might help to generate the
        envelope in some cases

    as_bool : bool, optional
        whether to return boolean arrays indicating if each element is a min or max value

    Output:
    lmin, lmax : low/high envelope idx of input signal s, or if as_bool=True, then boolean arrays where min/max

    Adapted from https://stackoverflow.com/questions/34235530/how-to-get-high-and-low-envelope-of-a-signal/60402647#60402647
    """
    # locals min
    lmin = (numpy.diff(numpy.sign(numpy.diff(s))) > 0).nonzero()[0] + 1
    # locals max
    lmax = (numpy.diff(numpy.sign(numpy.diff(s))) < 0).nonzero()[0] + 1

    if split:
        # s_mid is zero if s centered around x-axis or more generally mean of signal
        s_mid = numpy.mean(s)
        # pre-sorting of locals min based on relative position with respect to s_mid
        lmin = lmin[s[lmin] < s_mid]
        # pre-sorting of local max based on relative position with respect to s_mid
        lmax = lmax[s[lmax] > s_mid]

    # global min of dmin-chunks of locals min
    lmin = lmin[
        [i + numpy.argmin(s[lmin[i : i + dmin]]) for i in range(0, len(lmin), dmin)]
    ]
    # global max of dmax-chunks of locals max
    lmax = lmax[
        [i + numpy.argmax(s[lmax[i : i + dmax]]) for i in range(0, len(lmax), dmax)]
    ]

    if as_bool:
        lmin = numpy.isin(numpy.arange(s.shape[0]), lmin)
        lmax = numpy.isin(numpy.arange(s.shape[0]), lmax)

    return lmin, lmax


def nan_helper(y):
    """
    Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs

    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices

    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= numpy.interp(x(nans), x(~nans), y[~nans])

    https://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array
    """

    return numpy.isnan(y), lambda z: z.nonzero()[0]


# -----------------------------------------------------------------------
# Decorator to print name and execution time of each recipe step
# -----------------------------------------------------------------------
def wifes_recipe(func):
    def wrapper(*args, **kwargs):
        start_time = datetime.datetime.now()
        print("Start of WiFeS Recipe {}.".format(func.__name__))
        result = func(*args, **kwargs)
        duration = datetime.datetime.now() - start_time
        print(
            "End of WiFeS Recipe {}: took {} seconds.".format(
                func.__name__, duration.total_seconds()
            )
        )
        return result

    return wrapper


# ------------------------------------------------------------------------
# Function definition
# ------------------------------------------------------------------------


def move_files(src_dir_path, destination_dir_path, filenames):
    """
    Move files from the source directory to the destination directory.

    Parameters
    ----------
    src_dir_path : str
        The path of the source directory.
    destination_dir_path : str
        The path of the destination directory.
    filenames : list of str
        The list of filenames to be moved.

    Raises
    ------
    Exception
        If an error occurs while moving the files.
    """
    try:
        for file in filenames:
            src_file = os.path.join(src_dir_path, file)
            dest_file = os.path.join(destination_dir_path, file)
            # print(f"Moving file {src_file} to {dest_file}")
            shutil.move(src_file, dest_file)
    except Exception as e:
        print(f"Error moving files: {e}")


def copy_files(src_dir_path, destination_dir_path, filenames):
    """
    Copy files from the source directory to the destination directory.

    Parameters
    ----------
    src_dir_path : str
        The path to the source directory.
    destination_dir_path : str
        The path to the destination directory.
    filenames : list
        A list of filenames to be copied.

    Raises
    ------
    Exception
        If there is an error while copying the files.
    """
    try:
        for file in filenames:
            src_file = os.path.join(src_dir_path, file)
            dest_file = os.path.join(destination_dir_path, file)
            # Handle common file compression for raw data
            if re.search("\\.fz$", src_file):
                dest_file = dest_file.rstrip(".fz")
                if os.path.isfile(dest_file) and os.path.getmtime(
                    src_file
                ) < os.path.getmtime(dest_file):
                    continue
                temph = pyfits.open(src_file)
                pyfits.writeto(
                    dest_file,
                    data=temph[1].data,
                    header=temph[1].header,
                    output_verify="fix",
                    overwrite=True,
                )
                temph.close()
            elif re.search("\\.gz$", src_file):
                dest_file = dest_file.rstrip(".gz")
                if os.path.isfile(dest_file) and os.path.getmtime(
                    src_file
                ) < os.path.getmtime(dest_file):
                    continue
                temph = pyfits.open(src_file)
                pyfits.writeto(
                    dest_file,
                    data=temph[0].data,
                    header=temph[0].header,
                    output_verify="fix",
                    overwrite=True,
                )
                temph.close()
            else:
                if os.path.isfile(dest_file) and os.path.getmtime(
                    src_file
                ) < os.path.getmtime(dest_file):
                    continue
                shutil.copy(src_file, dest_file)
    except Exception as e:
        print(f"Error copying files: {e}")


def get_file_names(src_dir_path, glob_pattern):
    """
    Get the names of files in a directory that match a given search query in a glob
    pattern. Does not search in subfolders.

    Parameters
    ----------
    src_dir_path : str
        The path to the source directory.

    glob_pattern : str
        The glob pattern (search query) to match the filenames.

    Returns
    -------
    list
        A list of filenames that match the glob pattern.
    """
    filepaths = glob.glob(os.path.join(src_dir_path, glob_pattern), recursive=False)
    names = []
    for filepath in filepaths:
        filename = os.path.basename(filepath)
        names.append(filename)
    return names


def load_config_file(filename):
    """
    Load a configuration file in JSON5 format.

    Parameters
    ----------
    filename : str
        The name of the configuration file.

    Returns
    -------
    dict
        A dictionary containing the loaded configuration data.
    """
    reduction_scripts_dir = os.path.dirname(__file__)
    file_path = os.path.join(reduction_scripts_dir, filename)
    print(f"Loading configuration file: {file_path}")
    with open(file_path, "r") as f:
        return pyjson5.load(f)


# Update header
def set_header(filename, kw, kw_value, data_hdu=0):
    fh = pyfits.open(filename, mode="update")
    fh[data_hdu].header[kw] = kw_value
    fh.close()


# ------------------------------------------------------------------------
# METADATA WRANGLING FUNCTIONS
# ------------------------------------------------------------------------
def get_full_obs_list(metadata, exclude=None):
    """
    Get the full observation list from the given metadata.

    Parameters
    ----------
        metadata : dict
            A dictionary containing metadata information.

        exclude : list
            A list of imagetypes to exclude from the returned observation list.
            Default: None.

    Returns
    -------
        list
            The full observation list.
    """
    keep_type = ["bias", "arc", "wire", "dark", "domeflat", "twiflat"]
    # Allow types to be excluded
    if exclude is not None:
        keep_type = [t for t in keep_type if t not in exclude]
    full_obs_list = []
    base_fn_list = [fn for t in keep_type for fn in metadata[t]]
    for fn in base_fn_list:
        if fn not in full_obs_list:
            full_obs_list.append(fn)

    keep_type = ["sci", "std"]
    # Allow types to be excluded
    if exclude is not None:
        keep_type = [t for t in keep_type if t not in exclude]
    for obs in [fdict for t in keep_type for fdict in metadata[t]]:
        for key in obs.keys():
            if key != "stdtype" and key != "name":
                for fn in obs[key]:
                    if fn not in full_obs_list:
                        full_obs_list.append(fn)
    print(f"Full observation list: {full_obs_list}")
    return full_obs_list


def get_sci_obs_list(metadata):
    """
    Get a list of science observations from the metadata.

    Parameters
    ----------
    metadata : dict
        The metadata containing information about the observations.

    Returns
    -------
    list
        A list of science observation filenames.
    """
    sci_obs_list = []
    for obs in metadata["sci"]:
        for fn in obs["sci"]:
            if fn not in sci_obs_list:
                sci_obs_list.append(fn)
    print(f"Science observation list: {sci_obs_list}")
    return sci_obs_list


def get_std_obs_list(metadata, type="all"):
    """
    Get a list of standard observations.

    Parameters
    ----------
    - metadata : dict
        The metadata containing information about the observations.
    - type : str, optional
        The type of observations to include in the list.
        Options: 'all', 'flux', 'telluric'.
        Default: 'all'.

    Returns
    -------
    - std_obs_list : list
        A list of standard observation filenames.
    """
    std_obs_list = []
    for obs in metadata["std"]:
        for fn in obs["sci"]:
            if fn not in std_obs_list and type == "all":
                std_obs_list.append(fn)
            if fn not in std_obs_list and (type in obs["stdtype"]):
                std_obs_list.append(fn)
    print(f"Standard observation list ({type}): {std_obs_list}")
    return std_obs_list


def get_sky_obs_list(metadata):
    """
    Get a list of sky observations from the metadata.

    Parameters
    ----------
        metadata : dict
            The metadata containing information about the observations.

    Returns
    -------
        list
            A list of sky observation filenames.
    """
    sky_obs_list = []
    for obs in metadata["sci"] + metadata["std"]:
        if "sky" not in obs.keys():
            continue
        for fn in obs["sky"]:
            if fn not in sky_obs_list:
                sky_obs_list.append(fn)
    print(f"Sky observation list: {sky_obs_list}")
    return sky_obs_list


def get_associated_calib(metadata, this_fn, type):
    """
    Get the associated calibration file for a given data file.

    Parameters
    ----------
    metadata : dict
        The metadata dictionary containing information about the observations.
    this_fn : str
        The filename of the data file for which to find the associated calibration file.
    type : str
        The type of calibration file to search for.

    Returns
    -------
    str or bool
        The filename of the associated calibration file if found, or False if not found.
    """
    for obs in metadata["sci"] + metadata["std"]:
        if "sky" in obs.keys():
            sky = obs["sky"]
        else:
            sky = []
        for fn in obs["sci"] + sky:
            if fn == this_fn:
                if type in obs.keys():
                    if obs[type] != "":
                        return obs[type]
    return False


def get_primary_sci_obs_list(metadata):
    """
    Get the list of primary science observations from the metadata.

    Parameters
    ----------
    metadata : dict
        The metadata containing information about the observations.

    Returns
    -------
    list
        The list of primary science observations.
    """
    sci_obs_list = [obs["sci"][0] for obs in metadata["sci"]]
    print(f"Primary science observation list: {sci_obs_list}")
    return sci_obs_list


def get_primary_std_obs_list(metadata, stdtype="all"):
    """
    Get the list of primary standard observations based on the given metadata and type.

    Parameters
    ----------
        metadata : dict
            The metadata containing information about the observations.
        stdtype : str, optional
            The type of standard star observations to include in the list.
            Options: 'all', 'telluric', 'flux'.
            Default: 'all'.

    Returns
    -------
        list
            The list of primary standard observations.

    Raises
    ------
        ValueError
            If the standard star type is not understood.
    """
    print(f"In get_primary_std_obs_list wanting type {stdtype} from {metadata['std']}")
    if stdtype == "all":
        std_obs_list = [obs["sci"][0] for obs in metadata["std"]]
    elif stdtype == "telluric" or stdtype == "flux":
        std_obs_list = []
        for obs in metadata["std"]:
            if obs["sci"][0] not in std_obs_list and (stdtype in obs["stdtype"]):
                std_obs_list.append(obs["sci"][0])
    else:
        print("Standard star type not understood!")
        print("PyWiFeS Data Reduction pipeline will crash now ...")
        raise ValueError("Standard star type not understood")
    print(f"Primary standard observation list ({stdtype}): {std_obs_list}")
    return std_obs_list


# ------------------------------------------------------------------------
