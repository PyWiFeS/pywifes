import os
import multiprocessing
from pywifes import pywifes
from pywifes.multiprocessing_utils import _get_num_processes as get_num_proc
from pywifes.wifes_utils import wifes_recipe


# ------------------------------------------------------
# Image coaddition for science and standards
# ------------------------------------------------------
def _run_obs_coadd_indiv(obs, gargs, prev_suffix, curr_suffix, method, scale, **args):
    # if just one, then copy it
    if len(obs['sci']) == 1:
        fn = obs['sci'][0]
        in_fn = os.path.join(gargs['out_dir_arm'], '%s.p%s.fits' % (fn, prev_suffix))
        out_fn = os.path.join(gargs['out_dir_arm'], '%s.p%s.fits' % (fn, curr_suffix))
        if gargs['skip_done'] and os.path.isfile(out_fn) \
                and os.path.getmtime(in_fn) < os.path.getmtime(out_fn):
            return
        print(f"Copying image {os.path.basename(in_fn)}")
        pywifes.imcopy(in_fn, out_fn)
    # coadd sci frames!
    else:
        in_fn_list = [os.path.join(gargs['out_dir_arm'], '%s.p%s.fits' % (fn, prev_suffix))
                      for fn in obs['sci']]
        out_fn = os.path.join(gargs['out_dir_arm'], '%s.p%s.fits' % (obs['sci'][0], curr_suffix))
        print(f"Coadding images for {os.path.basename(in_fn_list[0])}")
        if gargs['skip_done'] and os.path.isfile(out_fn) \
                and os.path.getmtime(in_fn_list[0]) < os.path.getmtime(out_fn):
            return
        pywifes.imcombine_mef(in_fn_list, out_fn, scale=scale, method=method)


@wifes_recipe
def _run_obs_coadd(metadata, gargs, prev_suffix, curr_suffix, method="sum", scale=None):
    """
    Coadd science and standard frames.

    Parameters
    ----------
    metadata : dict
        Dictionary containing metadata information of the FITS files.
    gargs : dict
        A dictionary containing global arguments used by the processing steps.
    prev_suffix : str
        Previous suffix of the file name (input).
    curr_suffix : str
        Current suffix of the file name (output).
    method : str, optional
        Method used for coadding the frames.
        Options: 'median', 'sum', 'nansafesum'.
        Default: 'sum'.
    scale : float, optional
        Scale factor applied to the frames during coaddition.
        Options: 'exptime', 'per_slice_median', None.
        Default: None.

    Returns
    -------
    None
    """
    nworkers = get_num_proc()
    obslist = metadata['sci'] + metadata['std']
    nobs = len(obslist)
    for worker in range(0, nobs, nworkers):
        nmax = worker + nworkers
        if nmax > nobs:
            nmax = nobs
        jobs = []
        for fidx in range(worker, nmax):
            obs = obslist[fidx]
            p = multiprocessing.Process(target=_run_obs_coadd_indiv,
                                        args=(obs, gargs, prev_suffix, curr_suffix,
                                              method, scale))
            jobs.append(p)
            p.start()

        for proc in jobs:
            proc.join()
    return
