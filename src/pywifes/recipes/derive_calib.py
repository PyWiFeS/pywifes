import os
from pywifes import wifes_calib
from pywifes.wifes_utils import get_primary_std_obs_list, move_files, wifes_recipe


# Sensitivity Function fit
@wifes_recipe
def _run_derive_calib(metadata, gargs, prev_suffix, curr_suffix, method="poly", prefactor=False, **args):
    """
    Derive the sensitivity function from the extracted standard stars.

    Parameters
    ----------
    metadata : dict
        Metadata containing information about the standard stars.
    gargs : dict
        A dictionary containing global arguments used by the processing steps.
    prev_suffix : str
        Previous suffix of the file names.
    curr_suffix : str
        Current suffix of the file names.
    method : str, optional
        Method for deriving the sensitivity function.
        Options: 'smooth_SG', 'poly.
        Default: 'poly'.
    prefactor : bool, optional
        Whether to remove a shape (e.g., from the flat lamp) before fitting the
        shape of the standard star (and reapply to final sensitivity).
        Default: False.

    Optional Function Arguments
    ---------------------------
    polydeg : int, optional
        Degree of the polynomial for fitting the calibration solution.
        Default: 30.
    wave_min : float, optional
        Minimum wavelength to compute flux ratio. If None, defaults to minimum
        observed wavelength.
        Default: None.
    wave_max : float, optional
        Maximum wavelength to compute flux ratio. If None, defaults to maximum
        observed wavelength.
        Default: None.
    norm_stars : bool, optional
        Whether to normalize the correction factor amongst multiple standard stars
        at the wavelength midpoint.
        Default: False.
    excise_cut : float, optional
        Fractional threshold for outliers (applied to magnitude units). Overridden
        with 0.003 if 'method'='smooth_SG'.
        Default: 0.5.
    ytrim : int, optional
        If standard star spectrum not previously extracted, mask this number of
        (unbinned) y-axis pixels when peak-finding.
        Default: 4.
    boxcar : int, optional
        Boxcar smoothing length for'method'='smooth_SG'.
        Default: 11.
    stdstar_name_list : list, optional
        List of standard star names corresponding to each cube.
        Default: None.
    airmass_list : list, optional
        List of airmass values corresponding to each cube. If None, extracts
        airmass from header.
        Default: None.
    ref_dir : str, optional
        Directory path to the reference data.
        Default: Pipeline's 'reference_data' directory.
    ref_fname_list : list, optional
        List of file names of the reference data corresponding to each cube. If
        None, looks up name in reference list based on image headers.
        Default: None.
    extinction_fn : str, optional
        File path to the extinction curve data. If None, defaults to standard SSO
        extinction curve.
        Default: None.
    plot_stars : bool, optional
        Whether to plot the stars during the calibration process.
        Default: False.
    plot_sensf : bool, optional
        Whether to plot the sensitivity function.
        Default: False.
    save_prefix : str, optional
        Prefix for the saved calibration files.
        Default: 'calib_'.
    debug : bool, optional
        Whether to report the parameters used in this function call.
        Default: False.

    Returns
    -------
    None
    """
    std_obs_list = get_primary_std_obs_list(metadata, stdtype="flux")

    if len(std_obs_list) == 0:
        print("No flux standard stars to derive calibration. Skipping.")
        return

    std_cube_list = [
        os.path.join(gargs['out_dir_arm'], f"{fn}.p{prev_suffix}.fits")
        for fn in std_obs_list
    ]
    extract_list = [
        os.path.join(gargs['out_dir_arm'], f"{fn}.x{prev_suffix}.dat")
        for fn in std_obs_list
    ]
    if prefactor:
        if os.path.isfile(gargs['smooth_shape_fn']):
            print(f"Using prefactor file {gargs['smooth_shape_fn']}")
            pf_fn = gargs['smooth_shape_fn']
        else:
            print(f"Could not find prefactor file {gargs['smooth_shape_fn']}")
            prefactor = False
            pf_fn = None
    else:
        pf_fn = None
    if gargs['skip_done'] and os.path.isfile(gargs['calib_fn']) and \
        (os.path.getmtime(std_cube_list[0]) < os.path.getmtime(gargs['calib_fn'])
            or gargs['from_master']):
        print("Skipping derivation of sensitivity function due to existing file.")
        return
    print("Deriving sensitivity function")
    wifes_calib.derive_wifes_calibration(
        std_cube_list, gargs['calib_fn'], extract_in_list=extract_list, method=method,
        plot_dir=gargs['plot_dir_arm'], prefactor=prefactor, prefactor_fn=pf_fn, **args
    )
    if gargs['from_master']:
        move_files(gargs['master_dir'], gargs['output_master_dir'], [os.path.basename(gargs['calib_fn'])])
    return
