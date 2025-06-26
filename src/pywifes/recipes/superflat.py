import os
from pywifes import pywifes
from pywifes.wifes_utils import wifes_recipe


# ------------------------------------------------------
# Generate super-flat/wire/arc
# ------------------------------------------------------
@wifes_recipe
def _run_superflat(
    metadata, gargs, prev_suffix, curr_suffix, source, **args
):
    """
    Generate a co-add calibration for a given source type.

    Parameters
    ----------
    metadata : dict
        Metadata containing information about the FITS files from which extract the
        inputs.
    gargs : dict
        A dictionary containing global arguments used by the processing steps.
    prev_suffix : str
        Previous suffix of the input files.
    curr_suffix : str
        Current suffix of the output files.
    source : str
        Type of frame.
        Options: 'dome', 'twi', 'wire', 'arc'.

    Optional Function Arguments
    ---------------------------
    method : str
        The method to be used for combining the images.
        Options: 'median', 'sum', 'mean'.
        Default: 'median'.
    scale : str
        The scaling method to be used before combining. The 'median_nonzero' option
        scales by pixels above 'nonzero_thresh'. The 'percentileN' option scales by
        the Nth percentile of the pixels in 'sregion' (e.g., 'percentile90').
        Options: 'median', 'median_nonzero', 'exptime', 'percentileN', None.
        Default: None.
    sregion : list
        List defining image x-axis region in which to compute scaling (if a scaling
        method is defined). If None, calculates the scaling over the whole image.
        Example: [1000, 3000].
        Default: None.
    nonzero_thresh : float
        Threshold for counting pixels to "median_nonzero" scaling.
        Default: 100.0.
    plot : bool
        Whether to output a diagnostic plot.
        Default: False.
    save_prefix : str
        Prefix for plot (if requested).
        Default: 'imcombine_inputs'.
    interactive_plot : bool
        Whether to interrupt processing to provide interactive plot to user.
        Default: False.
    debug : bool
        Whether to report the parameters used in this function call.
        Default: False.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the type is not recognized.
    """
    kwstring = None
    commstring = None
    outvarimg = None
    save_prefix = 'imcombine_inputs'
    if source == "dome":
        out_fn = gargs['super_dflat_raw']
        flat_list = [
            os.path.join(gargs['out_dir_arm'], "%s.p%s.fits" % (x, prev_suffix))
            for x in metadata["domeflat"]
        ]
        if gargs['skip_done'] and os.path.isfile(out_fn) \
                and os.path.getmtime(flat_list[0]) < os.path.getmtime(out_fn):
            return
        print(f"List of {source} flats: {flat_list}")
        kwstring = "FLATN"
        commstring = "lamp flats"
        outvarimg = os.path.join(gargs['master_dir'], f"wifes_{gargs['arm']}_super_domeflat_raw_var.fits")
        save_prefix = f"dflat_{save_prefix}"
    elif source == "twi":
        out_fn = gargs['super_tflat_raw']
        flat_list = [
            os.path.join(gargs['out_dir_arm'], "%s.p%s.fits" % (x, prev_suffix))
            for x in metadata["twiflat"]
        ]
        if gargs['skip_done'] and os.path.isfile(out_fn) \
                and os.path.getmtime(flat_list[0]) < os.path.getmtime(out_fn):
            return
        print(f"List of {source} flats: {flat_list}")
        kwstring = "TWIN"
        commstring = "twilight flats"
        save_prefix = f"twi_{save_prefix}"
    elif source == "wire":
        out_fn = gargs['super_wire_raw']
        flat_list = [
            os.path.join(gargs['out_dir_arm'], "%s.p%s.fits" % (x, prev_suffix))
            for x in metadata["wire"]
        ]
        if gargs['skip_done'] and os.path.isfile(out_fn) \
                and os.path.getmtime(flat_list[0]) < os.path.getmtime(out_fn):
            return
        print(f"List of wire frames: {flat_list}")
        kwstring = "WIREN"
        commstring = "wires"
    elif source == "arc":
        out_fn = gargs['super_arc_raw']
        flat_list = [
            os.path.join(gargs['out_dir_arm'], "%s.p%s.fits" % (x, prev_suffix))
            for x in metadata["arc"]
        ]
        if gargs['skip_done'] and os.path.isfile(out_fn) \
                and os.path.getmtime(flat_list[0]) < os.path.getmtime(out_fn):
            return
        print(f"List of arc frames: {flat_list}")
        kwstring = "ARCN"
        commstring = "arcs"
    else:
        print(f"Calibration type '{source}' not recognised")
        raise ValueError(f"Calibration type '{source}' not recognised")
    if not flat_list:
        print(f"No {source} flats found. Skipping the superflat generation for {source}.")
        return
    print(f"Generating co-add {source} flat")
    pywifes.imcombine(
        flat_list, out_fn, data_hdu=gargs['my_data_hdu'], kwstring=kwstring, commstring=commstring,
        outvarimg=outvarimg, plot_dir=gargs['plot_dir_arm'], save_prefix=save_prefix, **args
    )
    return
