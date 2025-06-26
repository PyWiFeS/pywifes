import os
from pywifes import wifes_calib
from pywifes.wifes_utils import get_primary_std_obs_list, move_files, wifes_recipe


# ------------------------------------------------------
# Telluric - derive
# ------------------------------------------------------
@wifes_recipe
def _run_derive_telluric(metadata, gargs, prev_suffix, curr_suffix, **args):
    """
    Derive the telluric correction from the standard star.

    Parameters
    ----------
    metadata : dict
        Metadata containing information about the observations.
    gargs : dict
        A dictionary containing global arguments used by the processing steps.
    prev_suffix : str
        Previous suffix used in the file names.
    curr_suffix : str
        Current suffix to be used in the file names.

    Optional Function Arguments
    ---------------------------
    plot : bool, optional
        Flag indicating whether to generate and save a plot of the telluric
        correction function.
        Default: True.
    plot_stars : bool, optional
        Flag indicating whether to plot the individual O2 and H2O corrections for
        each star.
        Default: False.
    save_prefix : str, optional
        Prefix for the saved plot file.
        Default: 'telluric'.
    airmass_list : list, optional
        List of airmass values for each cube file. If not provided, the airmass
        values will be extracted from the cube headers.
        Default: None.
    telluric_threshold : float, optional
        Threshold value for the telluric correction. Regions with a correction
        ratio above this threshold will be set to 1.0.
        Default: 0.97.
    fit_wmin : float, optional
        Minimum wavelength for fitting the smooth polynomial to non-telluric
        regions.
        Default: 5400.0.
    fit_wmax : float, optional
        Maximum wavelength for fitting the smooth polynomial to non-telluric
        regions.
        Default: 10000.0.
    H2O_power : float, optional
        Power value for the H2O correction.
        Default: 0.72.
    O2_power : float, optional
        Power value for the O2 correction.
        Default: 0.40.
    polydeg : int, optional
        Degree of the polynomial used for fitting the smooth continuum.
        Default: 4.
    ytrim : int, optional
        Number of (unbinned) pixels to trim from the top and bottom of the data
        cube if standard star spectrum has not been extracted previously.
        Default: 4.
    debug : bool, optional
        Whether to report the parameters used in this function call.
        Default: False.

    Returns
    -------
    None
    """
    std_obs_list = get_primary_std_obs_list(metadata, stdtype="telluric")
    if len(std_obs_list) == 0:
        print("No telluric standard stars found. Skipping.")
        return

    std_cube_list = [
        os.path.join(gargs['out_dir_arm'], f"{fn}.p{prev_suffix}.fits")
        for fn in std_obs_list
    ]
    extract_list = [
        os.path.join(gargs['out_dir_arm'], f"{fn}.x{prev_suffix}.dat")
        for fn in std_obs_list
    ]
    if gargs['skip_done'] and os.path.isfile(gargs['tellcorr_fn']) \
        and (os.path.getmtime(std_cube_list[0]) < os.path.getmtime(gargs['tellcorr_fn'])
             or gargs['from_master']):
        print("Skipping derivation of telluric correction due to existing file.")
        return
    print("Deriving telluric correction")
    wifes_calib.derive_wifes_telluric(
        std_cube_list, gargs['tellcorr_fn'], extract_in_list=extract_list,
        plot_dir=gargs['plot_dir_arm'], **args
    )
    if gargs['from_master']:
        move_files(gargs['master_dir'], gargs['output_master_dir'], [os.path.basename(gargs['tellcorr_fn'])])
    return
