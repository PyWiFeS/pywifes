import os
from pywifes import pywifes
from pywifes.wifes_utils import get_full_obs_list, wifes_recipe


# ------------------------------------------------------
# repair bad pixels!
# ------------------------------------------------------
@wifes_recipe
def _run_bpm_repair(metadata, gargs, prev_suffix, curr_suffix, **args):
    """
    Repairs bad pixels in the input FITS files and saves the repaired files.

    Parameters
    ----------
    metadata : dict
        A dictionary containing metadata information of the FITS files.
    gargs : dict
        A dictionary containing global arguments used by the processing steps.
    prev_suffix : str
        The suffix of the previous version of the FITS files.
    curr_suffix : str
        The suffix of the current version of the FITS files.

    Optional Function Arguments
    ---------------------------
    flat_littrow : bool
        Whether to interpolate over the Littrow ghosts in dome and twilight flats.
        Default: False.
    interp_buffer : int
        When linearly interpolating over bad pixels, how many additional pixels to
        include when median-combining from one edge to the other.
        Default: 3.
    interactive_plot : bool
        Whether to interrupt processing to provide interactive plot to user.
        Default: False.
    verbose : bool
        Whether to output extra messages.
        Default: False.
    debug : bool
        Whether to report the parameters used in this function call.
        Default: False.

    Returns
    -------
    None
    """
    full_obs_list = get_full_obs_list(metadata)
    for basename in full_obs_list:
        input_filename = f"{basename}.p{prev_suffix}.fits"
        output_filename = f"{basename}.p{curr_suffix}.fits"
        input_filepath = os.path.join(gargs['out_dir_arm'], input_filename)
        output_filepath = os.path.join(gargs['out_dir_arm'], output_filename)
        if gargs['skip_done'] and os.path.isfile(output_filepath) \
                and os.path.getmtime(input_filepath) < os.path.getmtime(output_filepath):
            continue
        print(f"Repairing {gargs['arm']} bad pixels for {input_filename}")
        pywifes.repair_bad_pix(
            input_filepath, output_filepath, gargs['arm'], data_hdu=gargs['my_data_hdu'], **args
        )
