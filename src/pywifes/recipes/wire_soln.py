import os
from pywifes import pywifes
from pywifes.wifes_utils import get_associated_calib, get_sci_obs_list, get_std_obs_list, wifes_recipe


# ------------------------------------------------------
# Wire solution
# ------------------------------------------------------
@wifes_recipe
def _run_wire_soln(metadata, gargs, prev_suffix, curr_suffix, **args):
    """
    Global wire solution first, then local wire solutions for any specific
    observations.

    Parameters
    ----------
    metadata : dict
        Metadata containing information about the observations.
    gargs : dict
        A dictionary containing global arguments used by the processing steps.
    prev_suffix : str
        Previous suffix used in the file names.
    curr_suffix : str
        Current suffix to be used in the file names.

    Optional Function Arguments
    ---------------------------
    bin_x : int
        If specified, override the x-axis binning defined in the header.
        Default: None.
    bin_y : int
        If specified, override the y-axis binning defined in the header.
        Default: None.
    fit_zones : list
        Per-slitlet y-axis region limits to define background against which to
        locate wire. Unbinned pixels.
        Default: [16, 26, 54, 70].
    flux_threshold : float
        Minimum flux difference for wire relative to background. Default value
        assumes a superwire scaled by 'percentile90'.
        Default: 0.001.
    wire_polydeg : int
        Order of polynomial to fit wire position.
        Default: 1.
    xlims : str or list
        Either 'default' or a 2-element list defining the x-axis range for the wire
        fitting.
        Default: 'default'.
    plot : bool
        Whether to create the diagnotic plot.
        Default: True.
    save_prefix : str
        Prefix for the diagnostic plot.
        Default: 'wire_fit_params'.
    interactive_plot : bool
        Whether to interrupt processing to provide interactive plot to user.
        Default: False.
    debug : bool
        Whether to report the parameters used in this function call.
        Default: False.

    Returns
    -------
    None
    """
    # Global wire solution
    if os.path.isfile(gargs['super_wire_mef']):
        wire_in_fn = gargs['super_wire_mef']
    else:
        wire_in_fn = os.path.join(
            gargs['out_dir_arm'], "%s.p%s.fits" % (metadata["wire"][0], prev_suffix)
        )
        try:
            wire_in_fn = os.path.join(
                gargs['out_dir_arm'], "%s.p%s.fits" % (metadata["wire"][0], prev_suffix)
            )
        except IndexError:
            raise FileNotFoundError("No wire spectra available to fit.")
    if not (gargs['skip_done'] and os.path.isfile(gargs['wire_out_fn'])
            and os.path.getmtime(wire_in_fn) < os.path.getmtime(gargs['wire_out_fn'])):
        print(f"Deriving global wire solution from {os.path.basename(wire_in_fn)}")
        pywifes.derive_wifes_wire_solution(wire_in_fn,
                                           gargs['wire_out_fn'],
                                           plot_dir=gargs['plot_dir_arm'],
                                           **args)

    # Wire solutions for any specific obsevations
    sci_obs_list = get_sci_obs_list(metadata)
    std_obs_list = get_std_obs_list(metadata)
    for fn in sci_obs_list + std_obs_list:
        # Check if the file has a dedicated wire associated with it ...
        # Only for Science and Std stars (sky not required at this stage)
        local_wires = get_associated_calib(metadata, fn, "wire")
        if local_wires:
            local_wire_fn = os.path.join(
                gargs['out_dir_arm'], "%s.p%s.fits" % (local_wires[0], prev_suffix)
            )
            local_wire_out_fn = os.path.join(
                gargs['out_dir_arm'], "%s.wire.fits" % (local_wires[0])
            )
            if gargs['skip_done'] and os.path.isfile(local_wire_out_fn) \
                    and os.path.getmtime(local_wire_fn) < os.path.getmtime(local_wire_out_fn):
                continue
            print(f"Deriving local wire solution for {local_wires[0]}")
            pywifes.derive_wifes_wire_solution(local_wire_fn,
                                               local_wire_out_fn,
                                               plot_dir=gargs['plot_dir_arm'],
                                               save_prefix='local_wire_fit_params',
                                               **args)
    return
