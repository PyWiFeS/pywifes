import os
from pywifes import wifes_calib
from pywifes.wifes_utils import (
    get_primary_sci_obs_list, get_primary_std_obs_list, wifes_recipe
)


# ------------------------------------------------------
# Flux Calibration
# ------------------------------------------------------
@wifes_recipe
def _run_flux_calib(metadata, gargs, prev_suffix, curr_suffix, mode="pywifes", **args):
    """
    Flux calibrate all science and standard observations.

    Parameters
    ----------
    metadata : dict
        Metadata containing information about the observations.
    gargs : dict
        A dictionary containing global arguments used by the processing steps.
    prev_suffix : str
        Previous suffix of the file name (input).
    curr_suffix : str
        Current suffix of the file name (output).
    mode : str, optional
        Calibration mode.
        Options: 'iraf', 'pywifes'.
        Default is "pywifes".

    Optional Function Arguments
    ---------------------------
    extinction_fn : str, optional
        Extinction file path containing the extinction curve information. If None,
        defaults to standard SSO extinction curve.
        Default: None.
    save_extinction : bool, optional
        Whether to save the extinction model that was applied to translate the flux
        calibration to airmass zero.

    Returns
    -------
    None
    """
    sci_obs_list = get_primary_sci_obs_list(metadata)
    std_obs_list = get_primary_std_obs_list(metadata)
    if gargs['from_master'] and \
        os.path.isfile(os.path.join(gargs['output_master_dir'],
                                    os.path.basename(gargs['calib_fn']))):
        this_calib_fn = os.path.join(gargs['output_master_dir'],
                                     os.path.basename(gargs['calib_fn']))
    else:
        this_calib_fn = gargs['calib_fn']
    for fn in sci_obs_list + std_obs_list:
        in_fn = os.path.join(gargs['out_dir_arm'], f"{fn}.p{prev_suffix}.fits")
        out_fn = os.path.join(gargs['out_dir_arm'], f"{fn}.p{curr_suffix}.fits")
        if gargs['skip_done'] and os.path.isfile(out_fn) \
                and os.path.getmtime(in_fn) < os.path.getmtime(out_fn):
            continue
        print(f"Flux-calibrating cube {os.path.basename(in_fn)}")
        wifes_calib.calibrate_wifes_cube(in_fn, out_fn, this_calib_fn, mode, **args)
    return
