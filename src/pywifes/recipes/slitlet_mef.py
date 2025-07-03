import os
import gc
import multiprocessing
from pywifes import pywifes
from pywifes.multiprocessing_utils import _get_num_processes as get_num_proc
from pywifes.wifes_utils import (
    get_full_obs_list, is_nodshuffle, is_subnodshuffle, wifes_recipe
)


# ------------------------------------------------------
# MEF file creation
# ------------------------------------------------------
def _run_slitlet_mef_indiv(fn, gargs, prev_suffix, curr_suffix, slitlet_fn):
    in_fn = os.path.join(gargs['out_dir_arm'], '%s.p%s.fits' % (fn, prev_suffix))
    out_fn = os.path.join(gargs['out_dir_arm'], '%s.p%s.fits' % (fn, curr_suffix))
    if gargs['skip_done'] and os.path.isfile(out_fn) \
            and os.path.getmtime(in_fn) < os.path.getmtime(out_fn):
        return

    if is_nodshuffle(in_fn) or is_subnodshuffle(in_fn):
        sky_fn = os.path.join(gargs['out_dir_arm'], '%s.s%s.fits' % (fn, curr_suffix))
        pywifes.wifes_slitlet_mef_ns(in_fn, out_fn, sky_fn,
                                     data_hdu=gargs['my_data_hdu'],
                                     slitlet_def_file=slitlet_fn)
    else:
        pywifes.wifes_slitlet_mef(in_fn, out_fn, data_hdu=gargs['my_data_hdu'],
                                  slitlet_def_file=slitlet_fn)
    gc.collect()


@wifes_recipe
def _run_slitlet_mef(metadata, gargs, prev_suffix, curr_suffix, **args):
    """
    Create a Multi-Extension Fits (MEF) file for each observation in the metadata.

    Parameters
    ----------
    metadata : dict
        Metadata containing information about the observations.
    gargs : dict
        A dictionary containing global arguments used by the processing steps.
    prev_suffix : str
        Previous suffix of the fits file.
    curr_suffix : str
        Current suffix of the fits file.

    Optional Function Arguments
    ---------------------------
    nod_dy : int
        For Nod & Shuffle images only: offset in (unbinned) pixels between the
        object and sky slitlets in the y-direction.
        Default: 80.
    nan_method : str
        Method for treating NaN pixels: linear interpolation in the x-direction,
        or replace with a constant value.
        Options: 'interp', 'replace'.
        Default: 'interp'.
    repl_val : float
        If 'nan_method'='replace', replace NaN pixels with the value of 'repl_val'.
        Default: 0.
    bin_x : int
        If specified, override the x-axis binning defined in the header.
        Default: None.
    bin_y : int
        If specified, override the y-axis binning defined in the header.
        Default: None.
    debug : bool
        Whether to report the parameters used in this function call.
        Default: False.


    Returns
    -------
    None
    """
    # Do not need MEF versions of individual frames for those with supercals (and with no "locals")
    excl_list = ["bias", "dark", "domeflat", "twiflat"]
    full_obs_list = get_full_obs_list(metadata, exclude=excl_list)
    # check the slitlet definition file
    if os.path.isfile(gargs['slitlet_def_fn']):
        slitlet_fn = gargs['slitlet_def_fn']
    else:
        slitlet_fn = None

    nworkers = get_num_proc()
    nobs = len(full_obs_list)

    for worker in range(0, nobs, nworkers):
        nmax = worker + nworkers
        if nmax > nobs:
            nmax = nobs
        jobs = []
        for fidx in range(worker, nmax):
            fn = full_obs_list[fidx]
            p = multiprocessing.Process(target=_run_slitlet_mef_indiv, args=(
                fn, gargs, prev_suffix, curr_suffix, slitlet_fn))
            jobs.append(p)
            p.start()

        for proc in jobs:
            proc.join()
    return
