import os
from pywifes import pywifes
from pywifes.wifes_utils import (
    get_primary_sci_obs_list, get_primary_std_obs_list, is_halfframe, is_taros,
    wifes_recipe
)


@wifes_recipe
def _run_save_3dcube(metadata, gargs, prev_suffix, curr_suffix, **args):
    '''
    Save 3D Data Cube for all science and standard observations.
    Final data product.

    Parameters
    ----------
    metadata : dict
        Metadata containing information about the observations.
    gargs : dict
        A dictionary containing global arguments used by the processing steps.
    prev_suffix : str
        Previous suffix of the file name (input).
    curr_suffix : str
        Current suffix of the file name.
        Saves final "*.cube.fits" file (output).

    Optional Function Arguments
    ---------------------------
    nan_bad_pixels : bool
        Whether to set masked pixels (DQ > 0) to NaN.
        Default: False.
    debug : bool
        Whether to report the parameters used in this function call.
        Default: False.

    Returns
    -------
    None
    '''
    sci_obs_list = get_primary_sci_obs_list(metadata)
    std_obs_list = get_primary_std_obs_list(metadata)

    # Check if is half-frame from the first sci image
    if sci_obs_list:
        sci_filename = gargs['data_dir'] + sci_obs_list[0] + ".fits"
    elif std_obs_list:
        sci_filename = gargs['data_dir'] + std_obs_list[0] + ".fits"
    else:
        # no files to process
        return

    halfframe = is_halfframe(sci_filename)
    taros = is_taros(sci_filename)

    # now generate cubes
    for fn in sci_obs_list + std_obs_list:
        in_fn = os.path.join(gargs['out_dir_arm'], f"{fn}.p{prev_suffix}.fits")
        out_fn = os.path.join(gargs['out_dir_arm'], f"{fn}.{curr_suffix}.fits")
        if gargs['skip_done'] and (
            (
                os.path.isfile(out_fn)
                and os.path.getmtime(in_fn) < os.path.getmtime(out_fn)
            ) or (
                os.path.isfile(os.path.join(gargs['output_dir'], f"{fn}.{curr_suffix}.fits"))
                and os.path.getmtime(in_fn) < os.path.getmtime(os.path.join(gargs['output_dir'], f"{fn}.{curr_suffix}.fits"))
            )
        ):
            continue
        print(f"Saving 3D Data Cube for {os.path.basename(in_fn)}")
        pywifes.generate_wifes_3dcube(in_fn, out_fn, halfframe=halfframe,
                                      taros=taros, **args)
    return
