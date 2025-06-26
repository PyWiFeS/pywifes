import numpy
import os
from pywifes import wifes_wsol
from pywifes.wifes_utils import get_associated_calib, get_sci_obs_list, get_std_obs_list, wifes_recipe


# ------------------------------------------------------
# Wavelength solution
# ------------------------------------------------------
# N.B. The multiprocessing for the wavelength calibration
# is performed in src/pywifes/wifes_wsol.py, not here.
@wifes_recipe
def _run_wave_soln(metadata, gargs, prev_suffix, curr_suffix, **args):
    """
    Wavelength Solution:
    Generate the master arc solution, based on generic arcs at first. Then looks
    for the local wavelength solutions for science or standards (sky not required
    at this stage). Check if the file has a dedicated arc associated with it. If
    two arcs are present, find a solution for both to later interpolate between
    them. Restrict it to the first two arcs in the list (in case the feature is
    being unknowingly used).

    Parameters
    ----------
    metadata : dict
        Metadata information for the data.
    gargs : dict
        A dictionary containing global arguments used by the processing steps.
    prev_suffix : str
        Previous suffix of the file.
    curr_suffix : str
        Current suffix of the file.

    Optional Function Arguments
    ---------------------------
    method : str
        Method for fitting wavelength solution: optical model of spectrograph or
        polynomial method.
        Options: 'optical', 'poly'
        Default: 'optical'.

        Options for 'method'='optical':
            shift_method : str
                Method to shift model to match reference line list: shifting each
                row, just the central row of each slitlet, sampling a grid of rows
                in the y-axis, or guessing from the optical model.
                Options: 'xcorr_all', 'xcorr_single', 'xcorr_grid', None.
                Default: 'xcorr_all'.
            find_method : str
                Method for fitting the arc lines, using MPFIT, numpy fitter (using
                logFlux), or scipy fitter.
                Options: 'mpfit', 'loggauss', 'least_squares'.
                Default: 'mpfit'.
            doalphapfit : bool
                Whether to fit each slitlet angle of incidence.
                Default: True.
            dlam_cut_start : float
                Initial threshold in Angstroms for the derivative calculation
                portion of the list matching.
                Default: 5.0.
            flux_threshold_nsig : float
                Number of sigma above the background for line detection.
                Default: 3.0.
            epsilon : float
                Threshold for excluding lines when matching to exclusion list.
                Default: 0.005.
            automatic : bool
                Whether to exclude lines with large residuals.
                Default: False.
            sigma : float
                RMS threshold offset from mean for excluding lines if
                'automatic'=True.
                Default: 1.0.
            decimate : bool
                Whether to perform initial fit with 10% of data.
                Default: False.
            multithread : bool
                Whether to run step using "multiprocessing" module.
                Default: true.

        Options for 'method'='poly':
            dlam_cut_start : float
                Initial threshold in Angstroms for the derivative calculation
                portion of the list matching.
                Default: 7.0.
            dlam_cut : float
                Subsequent threshold for matching to line lists.
                Default: 3.0.
            x_polydeg : int
                Order of x-axis polynomial.
                Default: 4.
            y_polydeg : int
                Order of y-axis polynomial.
                Default: 2.
            flux_threshold_nsig : float
                Number of sigma above the background for line detection.
                Default: 3.0.
            deriv_threshold_nsig : float
                Threshold for number of sigma different from median flux derivative
                per x pixel.
                Default: 1.0.

    Returns
    -------
    None
    """
    # Global arc solution
    if os.path.isfile(gargs['super_arc_mef']):
        wsol_in_fn = gargs['super_arc_mef']
    else:
        try:
            wsol_in_fn = os.path.join(
                gargs['out_dir_arm'], "%s.p%s.fits" % (metadata["arc"][0], prev_suffix)
            )
        except IndexError:
            raise FileNotFoundError("No arc spectra available to fit.")
    if not (gargs['skip_done'] and os.path.isfile(gargs['wsol_out_fn'])
            and os.path.getmtime(wsol_in_fn) < os.path.getmtime(gargs['wsol_out_fn'])):
        print(f"Deriving master wavelength solution from {os.path.basename(wsol_in_fn)}")
        wifes_wsol.derive_wifes_wave_solution(wsol_in_fn, gargs['wsol_out_fn'],
                                              plot_dir=gargs['plot_dir_arm'], **args)

    # Arc solutions for any specific obsevations
    sci_obs_list = get_sci_obs_list(metadata)
    std_obs_list = get_std_obs_list(metadata)

    for fn in sci_obs_list + std_obs_list:
        local_arcs = get_associated_calib(metadata, fn, "arc")

        if local_arcs:
            for i in range(numpy.min([2, numpy.size(local_arcs)])):
                local_arc_fn = os.path.join(
                    gargs['out_dir_arm'], "%s.p%s.fits" % (local_arcs[i], prev_suffix)
                )

                local_wsol_out_fn = os.path.join(
                    gargs['out_dir_arm'], "%s.wsol.fits" % (local_arcs[i])
                )

                if gargs['skip_done'] and os.path.isfile(local_wsol_out_fn) \
                        and os.path.getmtime(local_arc_fn) < os.path.getmtime(local_wsol_out_fn):
                    continue
                print(f"Deriving local wavelength solution for {local_arcs[i]}")

                wifes_wsol.derive_wifes_wave_solution(
                    local_arc_fn,
                    local_wsol_out_fn,
                    plot_dir=gargs['plot_dir_arm'],
                    **args
                )

    return
