from astropy.io import fits as pyfits
import os
from pywifes import pywifes
from pywifes.wifes_utils import get_primary_sci_obs_list, get_primary_std_obs_list, wifes_recipe


# ------------------------------------------------------
# Flatfield: Division
# ------------------------------------------------------
@wifes_recipe
def _run_flatfield(metadata, gargs, prev_suffix, curr_suffix):
    """
    Apply flat-field correction (division) to science and standard frames.

    Parameters
    ----------
    metadata : dict
        Metadata containing information about the FITS files of the observations.
    gargs : dict
        A dictionary containing global arguments used by the processing steps.
    prev_suffix : str
        Previous suffix of the file names (input).
    curr_suffix : str
        Current suffix of the file names (output).

    Returns
    -------
    None
    """
    sci_obs_list = get_primary_sci_obs_list(metadata)
    std_obs_list = get_primary_std_obs_list(metadata)
    print(f"Primary science observation list: {sci_obs_list}")
    print(f"Primary standard observation list: {std_obs_list}")
    for fn in sci_obs_list + std_obs_list:
        in_fn = os.path.join(gargs['out_dir_arm'], "%s.p%s.fits" % (fn, prev_suffix))
        out_fn = os.path.join(gargs['out_dir_arm'], "%s.p%s.fits" % (fn, curr_suffix))
        if gargs['skip_done'] and os.path.isfile(out_fn) \
                and os.path.getmtime(in_fn) < os.path.getmtime(out_fn) \
                and os.path.getmtime(gargs['flat_resp_fn']) < os.path.getmtime(out_fn):
            continue
        print(f"Flat-fielding image {os.path.basename(in_fn)}")
        pywifes.imarith_mef(in_fn, "/", gargs['flat_resp_fn'], out_fn)
        ffh = pyfits.getheader(gargs['flat_resp_fn'])
        ffin = ffh.get("PYWRESIN", default="Unknown")
        dfnum = ffh.get("PYWFLATN", default="Unknown")
        if "twi" in ffin:
            tfnum = ffh.get("PYWTWIN", default="Unknown")
        else:
            tfnum = 0
        of = pyfits.open(out_fn, mode="update")
        of[0].header.set("PYWRESIN", ffin, "PyWiFeS: flatfield inputs")
        of[0].header.set("PYWFLATN", dfnum, "PyWiFeS: number lamp flat images combined")
        of[0].header.set("PYWTWIN", tfnum, "PyWiFeS: number twilight flat images combined")
        of.close()
    return
