import os
from pywifes import pywifes
from pywifes.wifes_utils import get_associated_calib, get_sci_obs_list, get_std_obs_list, wifes_recipe


# ------------------------------------------------------
# Generate super-bias
# ------------------------------------------------------
@wifes_recipe
def _run_superbias(metadata, gargs, prev_suffix, curr_suffix, method="row_med", **args):
    """
    Generate superbias for the entire dataset and for each science frame.
    Fit a smart surface to the bias or take the median of each row.

    Parameters
    ----------
    metadata: dict
        Metadata containing information about the dataset.
    gargs : dict
        A dictionary containing global arguments used by the processing steps.
    prev_suffix: str
        Previous suffix used in the filenames.
    curr_suffix: str
        Current suffix to be used in the filenames.
    method: str, optional
        Method to generate the superbias. Fit a smart surface to the bias or
        take the median of each row.
        Options: 'row_med', 'fit'.
        Default: 'row_med'.

    Optional Function Arguments
    ---------------------------
    plot : bool
        Whether to create the diagnotic plot.
        Default: True.
    save_prefix : str
        Prefix for the diagnostic plot.
        Default: 'bias'.
    verbose : bool
        Whether to output extra messages.
        Default: False.

    Returns
    -------
    None
    """
    bias_list = [os.path.join(gargs['out_dir_arm'], "%s.p%s.fits" % (x, prev_suffix))
                 for x in metadata["bias"]
                 ]
    if not (gargs['skip_done'] and os.path.isfile(gargs['superbias_fn'])
            and os.path.isfile(gargs['superbias_fit_fn'])
            and os.path.getmtime(gargs['superbias_fn']) < os.path.getmtime(gargs['superbias_fit_fn'])):
        print("Calculating Global Superbias")
        pywifes.imcombine(bias_list, gargs['superbias_fn'], data_hdu=gargs['my_data_hdu'],
                          kwstring="BIASN", commstring="bias")
        if method == "fit" or method == "row_med":
            pywifes.generate_wifes_bias_fit(
                gargs['superbias_fn'],
                gargs['superbias_fit_fn'],
                data_hdu=gargs['my_data_hdu'],
                method=method,
                plot_dir=gargs['plot_dir_arm'],
                **args,
            )
        else:
            pywifes.imcopy(gargs['superbias_fn'], gargs['superbias_fit_fn'])
    # generate local superbiases for any science frames
    sci_obs_list = get_sci_obs_list(metadata)
    std_obs_list = get_std_obs_list(metadata)
    print('***************************************************')
    print(f"Science observation list: {sci_obs_list}")
    print(f"Standard observation list: {std_obs_list}")
    print('***************************************************')
    for fn in sci_obs_list + std_obs_list:
        local_biases = get_associated_calib(metadata, fn, "bias")
        if local_biases:
            local_bias_fn = get_associated_calib(metadata, fn, "bias")[0]
            local_superbias = os.path.join(
                gargs['out_dir_arm'], "%s.fits" % (local_bias_fn + ".lsb")
            )
            local_superbias_fit = os.path.join(
                gargs['out_dir_arm'], "%s.fits" % (local_bias_fn + ".lsb_fit")
            )
            if gargs['skip_done'] and os.path.isfile(local_superbias) and os.path.isfile(local_superbias_fit) \
                    and os.path.getmtime(local_superbias) < os.path.getmtime(local_superbias_fit):
                continue
            print(f"Calculating Local Superbias for {local_bias_fn}")
            # step 1 - coadd biases
            local_biases_filename = [
                os.path.join(gargs['out_dir_arm'], "%s.p%s.fits" % (x, prev_suffix))
                for x in local_biases
            ]
            pywifes.imcombine(
                local_biases_filename, local_superbias, data_hdu=gargs['my_data_hdu'],
                kwstring="LOCBN", commstring="local bias"
            )
            # step 2 - generate fit
            if method == "fit" or method == "row_med":
                pywifes.generate_wifes_bias_fit(
                    local_superbias,
                    local_superbias_fit,
                    data_hdu=gargs['my_data_hdu'],
                    method=method,
                    **args,
                )
            else:
                pywifes.imcopy(local_superbias, local_superbias_fit)
    return
