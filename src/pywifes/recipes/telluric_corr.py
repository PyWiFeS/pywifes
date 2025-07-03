import os

from pywifes.wifes_utils import get_primary_sci_obs_list, get_primary_std_obs_list, wifes_recipe
from pywifes import wifes_calib


# ------------------------------------------------------
# Apply telluric correction
# ------------------------------------------------------
@wifes_recipe
def _run_telluric_corr(metadata, gargs, prev_suffix, curr_suffix, **args):
    """
    Apply telluric correction for all science and standard observations.

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

    Optional Function Arguments
    ---------------------------
    shift_sky : bool, optional
        Whether to shift the telluric to better align the sky lines between telluric and object.
        Default: True.
    sky_wmin : float, optional
        Minimum wavelength to fit if shifting based on sky lines.
        Default: 7200.0.
    sky_wmax : float, optional
        Maximum wavelength to fit if shifting based on sky lines.
        Default: 8100.0.
    save_telluric : bool, optional
        Whether to save the applied telluric model in the datacube.
        Default: False.
    interactive_plot : bool, optional
        Whether to interrupt processing to provide interactive plot to user.
        Default: False.

    Returns
    -------
    None
    """
    sci_obs_list = get_primary_sci_obs_list(metadata)
    std_obs_list = get_primary_std_obs_list(metadata)
    if gargs['from_master'] and os.path.isfile(os.path.join(gargs['output_master_dir'], os.path.basename(gargs['tellcorr_fn']))):
        this_tellcorr_fn = os.path.join(gargs['output_master_dir'], os.path.basename(gargs['tellcorr_fn']))
    else:
        this_tellcorr_fn = gargs['tellcorr_fn']
    for fn in sci_obs_list + std_obs_list:
        in_fn = os.path.join(gargs['out_dir_arm'], f"{fn}.p{prev_suffix}.fits")
        out_fn = os.path.join(gargs['out_dir_arm'], f"{fn}.p{curr_suffix}.fits")
        if gargs['skip_done'] and os.path.isfile(out_fn) \
                and os.path.getmtime(in_fn) < os.path.getmtime(out_fn):
            continue
        print(f"Correcting telluric in {os.path.basename(in_fn)}")
        wifes_calib.apply_wifes_telluric(in_fn, out_fn, this_tellcorr_fn, **args)
    return
