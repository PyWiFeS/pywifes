from astropy.io import fits as pyfits
import os
import gc
from pywifes.lacosmic import lacos_wifes
from pywifes.wifes_utils import (
    get_sci_obs_list, get_sky_obs_list, get_std_obs_list,
    is_nodshuffle, is_subnodshuffle, wifes_recipe
)


# ------------------------------------------------------
# Cosmic Rays
# ------------------------------------------------------
@wifes_recipe
def _run_cosmic_rays(
    metadata,
    gargs,
    prev_suffix,
    curr_suffix,
    multithread=False,
    max_processes=-1,
):
    """
    Clean cosmic rays on all science and standard frames.

    Parameters
    ----------
    metadata : dict
        Metadata containing information about the observations.
    gargs : dict
        A dictionary containing global arguments used by the processing steps.
    prev_suffix : str
        Previous suffix of the FITS file to apply the correction (input).
    curr_suffix : str
        Current suffix of the FITS file with the correction alredy applied (output).
    multithread : bool, optional
        Flag indicating whether to use multithreading for cosmic ray cleaning.
        Default: False.
    max_processes : int, optional
        Maximum number of processes to use for multithreading (-1 uses all
        available processes).
        Default: -1.

    Returns
    -------
    None
    """
    # now run ONLY ON SCIENCE TARGETS AND STANDARDS
    sci_obs_list = get_sci_obs_list(metadata)
    sky_obs_list = get_sky_obs_list(metadata)
    std_obs_list = get_std_obs_list(metadata)
    for fn in sci_obs_list + sky_obs_list:
        in_fn = os.path.join(gargs['out_dir_arm'], "%s.p%s.fits" % (fn, prev_suffix))
        out_fn = os.path.join(gargs['out_dir_arm'], "%s.p%s.fits" % (fn, curr_suffix))
        if gargs['skip_done'] and os.path.isfile(out_fn) \
                and os.path.getmtime(in_fn) < os.path.getmtime(out_fn):
            continue
        print(f"Cleaning cosmics in {os.path.basename(in_fn)}")
        in_hdr = pyfits.getheader(in_fn)
        rdnoise = 5.0 if 'RDNOISE' not in in_hdr else in_hdr['RDNOISE']
        lacos_wifes(
            in_fn,
            out_fn,
            rdnoise=rdnoise,
            wsol_fn=gargs['wsol_out_fn'],
            niter=3,
            sig_clip=10.0,
            obj_lim=10.0,
            sig_frac=0.2,
            is_multithread=multithread,
            max_processes=max_processes,
        )
        if is_nodshuffle(in_fn) or is_subnodshuffle(in_fn):
            # Also process extracted sky slitlets.
            in_fn = os.path.join(gargs['out_dir_arm'], "%s.s%s.fits" % (fn, prev_suffix))
            out_fn = os.path.join(gargs['out_dir_arm'], "%s.s%s.fits" % (fn, curr_suffix))
            print(f"Cleaning cosmics in {os.path.basename(in_fn)}")
            in_hdr = pyfits.getheader(in_fn)
            rdnoise = 5.0 if 'RDNOISE' not in in_hdr else in_hdr['RDNOISE']
            lacos_wifes(
                in_fn,
                out_fn,
                rdnoise=rdnoise,
                wsol_fn=gargs['wsol_out_fn'],
                niter=3,
                sig_clip=10.0,
                obj_lim=10.0,
                sig_frac=0.2,
                is_multithread=multithread,
                max_processes=max_processes,
            )
        gc.collect()
    for fn in std_obs_list:
        in_fn = os.path.join(gargs['out_dir_arm'], "%s.p%s.fits" % (fn, prev_suffix))
        out_fn = os.path.join(gargs['out_dir_arm'], "%s.p%s.fits" % (fn, curr_suffix))
        if gargs['skip_done'] and os.path.isfile(out_fn) \
                and os.path.getmtime(in_fn) < os.path.getmtime(out_fn):
            continue
        print(f"Cleaning cosmics in standard star {os.path.basename(in_fn)}")
        in_hdr = pyfits.getheader(in_fn)
        rdnoise = 5.0 if 'RDNOISE' not in in_hdr else in_hdr['RDNOISE']
        lacos_wifes(
            in_fn,
            out_fn,
            rdnoise=rdnoise,
            wsol_fn=gargs['wsol_out_fn'],
            niter=3,
            sig_clip=10.0,
            obj_lim=10.0,
            sig_frac=0.2,
            is_multithread=multithread,
            max_processes=max_processes,
        )
        if is_nodshuffle(in_fn) or is_subnodshuffle(in_fn):
            # Also process extracted sky slitlets.
            in_fn = os.path.join(gargs['out_dir_arm'], "%s.s%s.fits" % (fn, prev_suffix))
            out_fn = os.path.join(gargs['out_dir_arm'], "%s.s%s.fits" % (fn, curr_suffix))
            print(f"Cleaning cosmics in standard star {os.path.basename(in_fn)}")
            in_hdr = pyfits.getheader(in_fn)
            rdnoise = 5.0 if 'RDNOISE' not in in_hdr else in_hdr['RDNOISE']
            lacos_wifes(
                in_fn,
                out_fn,
                rdnoise=rdnoise,
                wsol_fn=gargs['wsol_out_fn'],
                niter=3,
                sig_clip=10.0,
                obj_lim=10.0,
                sig_frac=0.2,
                is_multithread=multithread,
                max_processes=max_processes,
            )
        gc.collect()
    return
