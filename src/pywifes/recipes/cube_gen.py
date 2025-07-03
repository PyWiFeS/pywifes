from astropy.io import fits as pyfits
import os
import datetime
import pickle
from pywifes import pywifes
from pywifes.wifes_utils import get_associated_calib, get_primary_sci_obs_list, get_primary_std_obs_list, wifes_recipe


# ------------------------------------------------------
# Data Cube Generation
# ------------------------------------------------------
@wifes_recipe
def _run_cube_gen(metadata, gargs, prev_suffix, curr_suffix, **args):
    '''
    Generate data cubes for science and standard frames.

    Parameters
    ----------
    metadata : dict
        Metadata dictionary containing information about the FITS of the
        observations.
    gargs : dict
        A dictionary containing global arguments used by the processing steps.
    prev_suffix : str
        Previous suffix used in the file names (input).
    curr_suffix : str
        Current suffix to be used in the file names (output).

    Optional Function Arguments
    ---------------------------
    wmin_set : float
        Minimum wavelength of output cube in Angstroms. If None, uses the largest
        minimum wavelength from the set of slitlets.
        Default: None.
    wmax_set : float
        Maximum wavelength of output cube in Angstroms. If None, uses the smallest
        maximum wavelength from the set of slitlets.
        Default: None.
    dw_set : float
        Wavelength step of output cube in Anstroms. If None, uses the mean of the
        per-slitlet mean pixel spacings.
        Default: None.
    wavelength_ref : str
        Whether to output cube with wavelengths calibrated to "AIR" or "VACUUM".
        Default: "AIR".
    adr : bool
        Apply atmospheric differential refraction correction.
        Default: False.
    subsample : int
        Divide each spatial dimension into 'subsample' components. Value of 1 means
        no subsampling. Minimises effects of intergerisation of pixel shifts, but
        increases processing time by subsample**2.
        Default: 1.
    offset_orig : int
        Number of (unbinned) y-axis pixels that the wire is offset from the field
        centre.
        Default: 2.
    ny_orig : int
        Number of (unbinned) y-axis pixels that should exist in output cube.
        Default: 76.
    bin_x : int
        If specified, override the x-axis binning defined in the header.
        Default: None.
    bin_y : int
        If specified, override the y-axis binning defined in the header.
        Default: None.
    verbose : bool
        Whether to output extra messages.
        Default: True.
    multithread : bool
        Flag indicating whether to use multithreading for cosmic ray cleaning.
        Default: False.
    max_processes : int
        Maximum number of processes to use for multithreading (-1 uses all
        available processes).
        Default: -1.
    debug : bool
        Whether to report the parameters used in this function call.
        Default: False.

    Returns
    -------
    None
    '''
    sci_obs_list = get_primary_sci_obs_list(metadata)
    std_obs_list = get_primary_std_obs_list(metadata)
    for fn in sci_obs_list + std_obs_list:
        in_fn = os.path.join(gargs['out_dir_arm'], "%s.p%s.fits" % (fn, prev_suffix))
        out_fn = os.path.join(gargs['out_dir_arm'], "%s.p%s.fits" % (fn, curr_suffix))
        # decide whether to use global or local wsol and wire files
        local_wires = get_associated_calib(metadata, fn, "wire")
        if local_wires:
            wire_fn = os.path.join(gargs['out_dir_arm'], "%s.wire.fits" % (local_wires[0]))
            print(f"(Note: using {os.path.basename(wire_fn)} as wire file)")
        else:
            wire_fn = gargs['wire_out_fn']
        local_arcs = get_associated_calib(metadata, fn, "arc")
        if local_arcs:
            # Do I have two arcs? Do they surround the Science file?
            # Implement linear interpolation:
            if len(local_arcs) == 2:
                # First, get the Science time
                f = pyfits.open(in_fn)
                sci_header = f[0].header
                sci_time = sci_header["DATE-OBS"]
                # Now get the arc times
                arc_times = ["", ""]
                for i in range(2):
                    local_wsol_out_fn_extra = os.path.join(
                        gargs['out_dir_arm'], f"{local_arcs[i]}.wsol.fits_extra.pkl")
                    with open(local_wsol_out_fn_extra, "rb") as f:
                        try:
                            f_pickled = pickle.load(f, protocol=2)
                        except Exception:
                            f_pickled = pickle.load(f)
                    f.close()
                    arc_times[i] = f_pickled[-1][0]

                # Make sure the Science is between the arcs:
                t0 = datetime.datetime(
                    int(arc_times[0].split("-")[0]),
                    int(arc_times[0].split("-")[1]),
                    int(arc_times[0].split("-")[2].split("T")[0]),
                    int(arc_times[0].split("T")[1].split(":")[0]),
                    int(arc_times[0].split(":")[1]),
                    int(arc_times[0].split(":")[2].split(".")[0]),
                )
                t1 = datetime.datetime(
                    int(sci_time.split("-")[0]),
                    int(sci_time.split("-")[1]),
                    int(sci_time.split("-")[2].split("T")[0]),
                    int(sci_time.split("T")[1].split(":")[0]),
                    int(sci_time.split(":")[1]),
                    int(sci_time.split(":")[2].split(".")[0]),
                )
                t2 = datetime.datetime(
                    int(arc_times[1].split("-")[0]),
                    int(arc_times[1].split("-")[1]),
                    int(arc_times[1].split("-")[2].split("T")[0]),
                    int(arc_times[1].split("T")[1].split(":")[0]),
                    int(arc_times[1].split(":")[1]),
                    int(arc_times[1].split(":")[2].split(".")[0]),
                )
                ds1 = (t1 - t0).total_seconds()
                ds2 = (t2 - t1).total_seconds()
                if ds1 > 0 and ds2 > 0:
                    # Interpolate betweent the two arcs
                    file_camera = sci_header["CAMERA"]
                    if file_camera == "WiFeSRed":
                        w1 = ds1 / (ds1 + ds2)
                        w2 = ds2 / (ds1 + ds2)
                    else:  # file_camera == 'WiFeSBlue'
                        w1 = ds2 / (ds1 + ds2)
                        w2 = ds1 / (ds1 + ds2)

                    # Open the arc solution files
                    fn0 = os.path.join(gargs['out_dir_arm'], f"{local_arcs[0]}.wsol.fits")
                    fn1 = os.path.join(gargs['out_dir_arm'], f"{local_arcs[1]}.wsol.fits")
                    fits0 = pyfits.open(fn0)
                    fits1 = pyfits.open(fn1)

                    for i in range(1, len(fits0)):
                        fits0[i].data = w1 * fits0[i].data + w2 * fits1[i].data

                    wsol_fn = os.path.join(gargs['out_dir_arm'], "%s.wsol.fits" % (fn))
                    fits0.writeto(wsol_fn, overwrite=True)

                    print("(2 arcs found)")
                    print(f"(Note: using {w1:.2f}x{local_arcs[0]}.wsol.fits + {w2:.2f}x{local_arcs[1]}.wsol.fits as wsol file)")

                else:
                    # Arcs do not surround the Science frame
                    # Revert to using the first one instead
                    wsol_fn = os.path.join(gargs['out_dir_arm'], f"{local_arcs[0]}.wsol.fits")
                    print("(2 arcs found, but they do not bracket the Science frame!)")
                    print(f"(Note: using {os.path.basename(wsol_fn)} as wsol file)")
            else:
                # IF Either 1 or more than two arcs present, only use the first one.
                wsol_fn = os.path.join(gargs['out_dir_arm'], f"{local_arcs[0]}.wsol.fits")
                print(f"(Note: using {os.path.basename(wsol_fn)} as wsol file)")
        else:
            wsol_fn = gargs['wsol_out_fn']
        if gargs['skip_done'] and os.path.isfile(out_fn) \
                and os.path.getmtime(in_fn) < os.path.getmtime(out_fn) \
                and os.path.getmtime(wire_fn) < os.path.getmtime(out_fn) \
                and os.path.getmtime(wsol_fn) < os.path.getmtime(out_fn):
            continue

        print(f"Generating Data Cube for {os.path.basename(in_fn)}")
        # All done, let's generate the cube
        pywifes.generate_wifes_cube(
            in_fn,
            out_fn,
            wire_fn=wire_fn,
            wsol_fn=wsol_fn,
            **args
        )
    return
