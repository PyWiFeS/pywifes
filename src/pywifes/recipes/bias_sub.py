from astropy.io import fits as pyfits
import os
from pywifes import pywifes
from pywifes.wifes_utils import get_associated_calib, get_full_obs_list, wifes_recipe


# ----------------------------------------------------
# Subtract bias
# ----------------------------------------------------
@wifes_recipe
def _run_bias_sub(metadata, gargs, prev_suffix, curr_suffix, method="subtract"):
    """
    Subtract bias from all the input data FITS files.

    Parameters
    ----------
    metadata : dict
        Metadata dictionary containing information about the FITS files of the observations.
    gargs : dict
        A dictionary containing global arguments used by the processing steps.
    prev_suffix : str
        Previous suffix of the data files.
    curr_suffix : str
        Current suffix of the data files.
    method : str, optional
        Method for bias subtraction.
        Options: 'subtract', 'copy'.
        Default: 'subtract'.

    Returns
    -------
    None
    """
    full_obs_list = get_full_obs_list(metadata)
    for fn in full_obs_list:
        in_fn = os.path.join(gargs['out_dir_arm'], "%s.p%s.fits" % (fn, prev_suffix))
        out_fn = os.path.join(gargs['out_dir_arm'], "%s.p%s.fits" % (fn, curr_suffix))
        if gargs['skip_done'] and os.path.isfile(out_fn) \
                and os.path.getmtime(in_fn) < os.path.getmtime(out_fn):
            continue
        local_biases = get_associated_calib(metadata, fn, "bias")
        if local_biases:
            local_bias_fn = get_associated_calib(metadata, fn, "bias")[0]
            bias_fit_fn = os.path.join(
                gargs['out_dir_arm'], "%s.fits" % (local_bias_fn + ".lsb_fit")
            )
            bias_type = "local"
        else:
            bias_fit_fn = gargs['superbias_fit_fn']
            bias_type = "global"

        # subtract it!
        print(f"Subtracting {bias_type} superbias for {os.path.basename(in_fn)}")
        if method == "copy":
            pywifes.imcopy(in_fn, out_fn)
        elif method == "subtract":
            pywifes.imarith(in_fn, "-", bias_fit_fn, out_fn, data_hdu=gargs['my_data_hdu'])

            bh = pyfits.getheader(bias_fit_fn)
            bnum = bh.get("PYWBIASN", default="Unknown")
            bmode = bh.get("PYWBMTHD", default="Unknown")
            of = pyfits.open(out_fn, mode="update")
            of[0].header.set("PYWBIASN", bnum, "PyWiFeS: number of bias images combined")
            of[0].header.set("PYWBMTHD", bmode, "PyWiFeS: bias fit method")
            of.close()

        else:
            raise ValueError(f"Unknown bias_sub method '{method}'. Options: 'subtract', 'copy'.")
    return
