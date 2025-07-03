import astropy.io.fits as fits
import os
from pywifes import pywifes
from pywifes.wifes_utils import (
    get_std_obs_list, is_nodshuffle, is_subnodshuffle, wifes_recipe
)


# ------------------------------------------------------
# Sky subtraction
# ------------------------------------------------------
@wifes_recipe
def _run_sky_sub(metadata, gargs, prev_suffix, curr_suffix, separate_ns=False):
    """
    Subtract sky frames from science objects in nod-and-shuffle mode.

    Parameters
    ----------
    metadata : dict
        Metadata containing information about the observations.
    gargs : dict
        A dictionary containing global arguments used by the processing steps.
    prev_suffix : str
        Previous suffix of the file names (input).
    curr_suffix : str
        Current suffix of the file names (output).

    Optional Function Arguments
    ---------------------------
    separate_ns : bool
        Whether to separate the two N&S positions and progress them independently.
        Default: False.

    Returns
    -------
    None
    """
    # Subtract sky frames from science objects.

    # Create a stable copy to enable additions without repetitions
    this_metadata_sci = metadata["sci"].copy()
    for obs in this_metadata_sci:
        if len(obs["sky"]) > 0:
            # Prepare the separate sky exposure, if defined.
            if len(obs["sky"]) > 1:
                # If multiple sky frames, scale by exposure time and
                # median-combine. List will be recombined for every science
                # frame in case the first sky (which defines the output
                # filename) was associated with multiple science frames. Thus,
                # no consideration of skip_done.
                in_fn_list = [
                    os.path.join(gargs['out_dir_arm'], "%s.p%s.fits" % (fn, prev_suffix))
                    for fn in obs["sky"]
                ]
                sky_proc_fn = os.path.join(
                    gargs['out_dir_arm'], "%s.p%s.fits" % (obs["sky"][0], curr_suffix)
                )
                print(f"Coadding sky frames into {os.path.basename(sky_proc_fn)}")
                pywifes.imcombine_mef(in_fn_list, sky_proc_fn, scale="exptime", method="median")
            else:
                # A single sky exposure.
                sky_fn = obs["sky"][0]
                sky_proc_fn = os.path.join(
                    gargs['out_dir_arm'], "%s.p%s.fits" % (sky_fn, prev_suffix)
                )
            for fn in obs["sci"]:
                # Subtract scaled sky frame from each science frame for a single object.
                in_fn = os.path.join(gargs['out_dir_arm'], "%s.p%s.fits" % (fn, prev_suffix))
                out_fn = os.path.join(
                    gargs['out_dir_arm'], "%s.p%s.fits" % (fn, curr_suffix)
                )
                if gargs['skip_done'] and os.path.isfile(out_fn) \
                        and os.path.getmtime(in_fn) < os.path.getmtime(out_fn):
                    continue
                print(f"Subtracting sky frame for {os.path.basename(in_fn)}")
                pywifes.scaled_imarith_mef(
                    in_fn, "-", sky_proc_fn, out_fn, scale="exptime"
                )
        else:
            # No separate sky frames defined.
            for fn in obs["sci"]:
                in_fn = os.path.join(gargs['out_dir_arm'], "%s.p%s.fits" % (fn, prev_suffix))
                out_fn = os.path.join(
                    gargs['out_dir_arm'], "%s.p%s.fits" % (fn, curr_suffix)
                )
                ns_input = is_nodshuffle(in_fn)
                if ns_input or is_subnodshuffle(in_fn):
                    # For Nod and Shuffle, locate the file containing the second position.
                    sky_fn = os.path.join(gargs['out_dir_arm'], "%s.s%s.fits" % (fn, prev_suffix))
                    if separate_ns:
                        # Treat the two offset positions as independent exposures.
                        if gargs['skip_done'] and os.path.isfile(out_fn) \
                                and os.path.getmtime(in_fn) < os.path.getmtime(out_fn):
                            pass
                        else:
                            print(f"Copying N&S science image {os.path.basename(in_fn)}")
                            pywifes.imcopy(in_fn, out_fn)

                        out_sky_fn = os.path.join(gargs['out_dir_arm'], "%ss.p%s.fits" % (fn, curr_suffix))
                        if gargs['skip_done'] and os.path.isfile(out_sky_fn) \
                                and os.path.getmtime(sky_fn) < os.path.getmtime(out_sky_fn):
                            pass
                        else:
                            print(f"Copying N&S sky image {os.path.basename(sky_fn)}")
                            pywifes.imcopy(sky_fn, out_sky_fn)
                            # Add 'sky' from N&S to science metadata
                            metadata['sci'].append({"sci": [os.path.basename(out_sky_fn).replace(f".p{curr_suffix}.fits", "")], "sky": []})
                            # Update headers
                            fh = fits.open(out_fn, mode="update")
                            fh_sky = fits.open(out_sky_fn, mode="update")
                            for extnum in range(len(fh_sky)):
                                # Remove N&S label
                                fh[extnum].header["WIFESOBS"] = "ClassicalEqual"
                                fh_sky[extnum].header["WIFESOBS"] = "ClassicalEqual"
                                # If not sub-frame Nod and Shuffle, update sky exposure time
                                if ns_input and "SEXPTIME" in fh_sky[extnum].header:
                                    fh_sky[extnum].header["EXPTIME"] = fh_sky[extnum].header["SEXPTIME"]
                                # Set header flag to aid separate extraction
                                fh[extnum].header["SPLIT_NS"] = (1.0, "First position of split N&S obs")
                                fh_sky[extnum].header["SPLIT_NS"] = (-1.0, "Second position of split N&S obs")
                            fh.close()
                            fh_sky.close()
                    else:
                        # Subtract Nod and Shuffle positions.
                        if gargs['skip_done'] and os.path.isfile(out_fn) \
                                and os.path.getmtime(in_fn) < os.path.getmtime(out_fn) \
                                and os.path.getmtime(sky_fn) < os.path.getmtime(out_fn):
                            continue
                        print(f"Subtracting N+S sky frame for {os.path.basename(in_fn)}")
                        # If not sub-frame Nod and Shuffle, update sky exposure time
                        if ns_input:
                            fh_sky = fits.open(sky_fn, mode="update")
                            for extnum in range(len(fh_sky)):
                                if "SEXPTIME" in fh_sky[extnum].header:
                                    fh_sky[extnum].header["EXPTIME"] = fh_sky[extnum].header["SEXPTIME"]
                            fh_sky.close()
                        pywifes.scaled_imarith_mef(in_fn, "-", sky_fn, out_fn, scale="exptime")
                else:
                    # For Classical observations, copy the science image.
                    if gargs['skip_done'] and os.path.isfile(out_fn) \
                            and os.path.getmtime(in_fn) < os.path.getmtime(out_fn):
                        continue
                    print(f"Copying science image {os.path.basename(in_fn)}")
                    pywifes.imcopy(in_fn, out_fn)
    # copy stdstar frames
    std_obs_list = get_std_obs_list(metadata)
    for fn in std_obs_list:
        in_fn = os.path.join(gargs['out_dir_arm'], "%s.p%s.fits" % (fn, prev_suffix))
        out_fn = os.path.join(gargs['out_dir_arm'], "%s.p%s.fits" % (fn, curr_suffix))
        if gargs['skip_done'] and os.path.isfile(out_fn) \
                and os.path.getmtime(in_fn) < os.path.getmtime(out_fn):
            continue
        print(f"Copying standard star image {os.path.basename(in_fn)}")
        pywifes.imcopy(in_fn, out_fn)

    return
