import os
from pywifes import wifes_calib
from pywifes.wifes_utils import get_primary_std_obs_list, wifes_recipe


# ------------------------------------------------------
# Standard star extraction
# ------------------------------------------------------
@wifes_recipe
def _run_extract_stars(metadata, gargs, prev_suffix, curr_suffix, stdtype="all", **args):
    """
    Extract standard stars spectrum.

    Parameters
    ----------
    metadata : dict
        The metadata containing information about the FITS files of the
        observations.
    gargs : dict
        A dictionary containing global arguments used by the processing steps.
    prev_suffix : str
        The suffix of the previous data files.
    curr_suffix : str
        The suffix of the current data files.
    stdtype : str, optional
        The type of standard stars to extract.
        Options: 'flux', 'telluric', 'all'.
        Default: 'all'.

    Optional Function Arguments
    ---------------------------
    x_ctr : float, optional
        The x-coordinate of the standard star centroid. If not provided, the
        centroid will be determined automatically.
        Default: None.
    y_ctr : float, optional
        The y-coordinate of the standard star centroid. If not provided, the
        centroid will be determined automatically.
        Default: None.
    extract_radius : float, optional
        The radius (in arcseconds) within which to extract the standard star
        spectrum.
        Default: 5.0.
    sky_radius : float, optional
        The radius (in arcseconds) outside which to estimate the sky background.
        Default: 8.0.
    xtrim : int, optional
        The number of pixels to trim from the left and right of the data cube in
        the x spatial direction.
        Default: 2.
    ytrim : int, optional
        The number of (unbinned) pixels to trim from the top and bottom of the data
        cube in the y spatial direction.
        Default: 4.
    wmask: int, optional
        The number of (unbinned) wavelength pixels to mask from each end when
        peak-finding.
        Default: 500.
    debug : bool, optional
        Whether to report the parameters used in this function call.
        Default: False.
    interactive_plot : bool, optional
        Whether to interrupt processing to provide interactive plot to user.
        Default: False.

    Returns
    -------
    None
    """
    # For each std, extract spectrum as desired
    std_obs_list = get_primary_std_obs_list(metadata, stdtype=stdtype)
    for fn in std_obs_list:
        in_fn = os.path.join(gargs['out_dir_arm'], f"{fn}.p{prev_suffix}.fits")
        out_fn = os.path.join(gargs['out_dir_arm'], f"{fn}.x{prev_suffix}.dat")
        if gargs['skip_done'] and os.path.isfile(out_fn) \
                and os.path.getmtime(in_fn) < os.path.getmtime(out_fn):
            continue
        print(f"Extract {stdtype} standard star from {os.path.basename(in_fn)}")
        wifes_calib.extract_wifes_stdstar(
            in_fn, save_fn=out_fn, save_mode="ascii", **args
        )
    return
