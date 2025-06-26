#! /usr/bin/env python3

import argparse
import astropy.io.fits as pyfits
import contextlib
import datetime
import glob
import multiprocessing
import os
from pathlib import Path
import re
import shutil
import sys
import traceback

from pywifes import pywifes
from pywifes.data_classifier import classify, cube_matcher
from pywifes.extract_spec import detect_extract_and_save, plot_1D_spectrum
from pywifes.quality_plots import flatfield_plot
from pywifes.splice import splice_spectra, splice_cubes
from pywifes.wifes_utils import (
    is_halfframe, is_nodshuffle, is_subnodshuffle, is_taros,
    copy_files, get_file_names, load_config_file, move_files
)
import pywifes.recipes as recipes


def run_arm_indiv(temp_data_dir, obs_metadatas, arm, master_dir, output_master_dir,
                  output_dir, params_path, grism_key, just_calib, plot_dir,
                  from_master, extra_skip_steps, return_dict, skip_done):
    # Reduces the data for an individual arm.

    try:
        # ------------------------------------------------------------------------
        #      LOAD JSON FILE WITH USER DATA REDUCTION SETUP
        # ------------------------------------------------------------------------
        obs_metadata = obs_metadatas[arm]

        # global args to keep filenames and other global info
        gargs = {}
        gargs['data_dir'] = temp_data_dir
        gargs['arm'] = arm
        gargs['master_dir'] = master_dir
        gargs['from_master'] = from_master
        gargs['output_master_dir'] = output_master_dir
        gargs['output_dir'] = output_dir

        # Determine the grism and observing mode used in the first image of science,
        # standard, or arc of the respective arm.
        # Skip reduction steps if no objects or standard stars are present.

        if obs_metadata["sci"]:
            reference_filename = obs_metadata["sci"][0]["sci"][0] + ".fits"
        elif obs_metadata["std"]:
            reference_filename = obs_metadata["std"][0]["sci"][0] + ".fits"
        elif obs_metadata["arc"]:
            reference_filename = obs_metadata["arc"][0] + ".fits"
        else:
            print("No science, standard, or arc files found in metadata.")
            raise ValueError("No science, standard, or arc files found in metadata.")

        # Check if reference image was taken with TAROS
        taros = is_taros(temp_data_dir + reference_filename)
        gargs['taros'] = taros

        # Check if reference image is half-frame
        halfframe = is_halfframe(temp_data_dir + reference_filename)
        gargs['halfframe'] = halfframe
        if halfframe:
            print(f"Cutting data to half-frame with to_taros = {taros}")
            obs_metadata = pywifes.calib_to_half_frame(obs_metadata, temp_data_dir,
                                                       to_taros=taros)

        # Grism
        grism = pyfits.getheader(temp_data_dir + reference_filename)[grism_key[arm]]

        # Set the JSON file path and read it.
        if params_path[arm] is None:
            json_path = str(Path(pywifes.__file__).parent / f"pipeline_params/{arm}/params_{grism}.json5")
            # json_path = f"./pipeline_params/{arm}/params_{grism}.json5"
        else:
            json_path = params_path[arm]

        # Load the JSON file
        proc_steps = load_config_file(json_path)

        # Remove irrelevant steps for just_calib usage
        if just_calib:
            json_idx = []
            for this_snum, this_step in enumerate(proc_steps[arm]):
                if this_step['step'] not in ['sky_sub', 'obs_coadd', 'flatfield',
                                             'cube_gen', 'extract_stars', 'derive_calib',
                                             'flux_calib', 'derive_telluric', 'telluric_corr',
                                             'save_3dcube']:
                    json_idx.append(this_snum)
            proc_steps[arm] = [proc_steps[arm][i] for i in json_idx]

        # Create data products directory structure
        gargs['out_dir_arm'] = os.path.join(output_dir, f"intermediate/{arm}")
        os.makedirs(gargs['out_dir_arm'], exist_ok=True)

        calib_prefix = f"wifes_{arm}"

        # Some WiFeS specific things
        gargs['my_data_hdu'] = 0

        # SET SKIP ALREADY DONE FILES ?
        gargs['skip_done'] = skip_done

        # Creates a directory for diagnisis plots (one per arm).
        gargs['plot_dir_arm'] = os.path.join(plot_dir, arm)
        os.makedirs(gargs['plot_dir_arm'], exist_ok=True)

        # ------------------------------------------------------------------------
        # Define names for master calibration files and set their path.
        # ------------------------------------------------------------------------
        # Bias Master Files
        gargs['overscanmask_fn'] = os.path.join(master_dir, f"{calib_prefix}_overscanmask.fits")
        gargs['superbias_fn'] = os.path.join(master_dir, f"{calib_prefix}_superbias.fits")
        gargs['superbias_fit_fn'] = os.path.join(master_dir, f"{calib_prefix}_superbias_fit.fits")

        # Dome Master Files
        gargs['super_dflat_raw'] = os.path.join(master_dir, f"{calib_prefix}_super_domeflat_raw.fits")
        gargs['super_dflat_fn'] = os.path.join(master_dir, f"{calib_prefix}_super_domeflat.fits")
        gargs['super_dflat_mef'] = os.path.join(master_dir, f"{calib_prefix}_super_domeflat_mef.fits")
        gargs['smooth_shape_fn'] = os.path.join(master_dir, f"{calib_prefix}_smooth_shape.dat")

        # Twilight Master Files
        gargs['super_tflat_raw'] = os.path.join(master_dir, f"{calib_prefix}_super_twiflat_raw.fits")
        gargs['super_tflat_fn'] = os.path.join(master_dir, f"{calib_prefix}_super_twiflat.fits")
        gargs['super_tflat_mef'] = os.path.join(master_dir, f"{calib_prefix}_super_twiflat_mef.fits")

        # Wire Master Files
        gargs['super_wire_raw'] = os.path.join(master_dir, f"{calib_prefix}_super_wire_raw.fits")
        gargs['super_wire_mef'] = os.path.join(master_dir, f"{calib_prefix}_super_wire_mef.fits")

        # Arc Master Files
        gargs['super_arc_raw'] = os.path.join(master_dir, f"{calib_prefix}_super_arc_raw.fits")
        gargs['super_arc_mef'] = os.path.join(master_dir, f"{calib_prefix}_super_arc_mef.fits")

        # Slitlet definition
        gargs['slitlet_def_fn'] = os.path.join(master_dir, f"{calib_prefix}_slitlet_defs.pkl")
        gargs['wsol_out_fn'] = os.path.join(master_dir, f"{calib_prefix}_wave_soln.fits")
        gargs['wire_out_fn'] = os.path.join(master_dir, f"{calib_prefix}_wire_soln.fits")
        gargs['flat_resp_fn'] = os.path.join(master_dir, f"{calib_prefix}_resp_mef.fits")
        gargs['calib_fn'] = os.path.join(master_dir, f"{calib_prefix}_calib.pkl")
        gargs['tellcorr_fn'] = os.path.join(master_dir, f"{calib_prefix}_tellcorr.pkl")

        # When reducing from master calibration files, if calibration files
        # are already among the master calibrations, skip their generation.
        if from_master:
            if os.path.exists(gargs['calib_fn']):
                extra_skip_steps.append("derive_calib")
            if os.path.exists(gargs['tellcorr_fn']):
                extra_skip_steps.append("derive_telluric")

        # ------------------------------------------------------------------------
        # Run proccessing steps
        # ------------------------------------------------------------------------
        flog_filename = os.path.join(gargs['output_dir'], f"{arm}.log")
        print("")
        print(f"Starting processing of {arm} arm")
        print(f"See {flog_filename} for detailed output.")
        print("")

        prev_suffix = None

        # keep a reference to current stdout so we can use it to print progress
        old_stdout = sys.stdout

        # get the current time to track how long each arm's reductions take
        start_time = datetime.datetime.now()

        # open a separate log file to direct all recipe stdout/stderr to
        with open(flog_filename, 'a') as flog:
            with contextlib.redirect_stdout(flog), contextlib.redirect_stderr(flog):
                counter = 1
                nsteps = len(proc_steps[arm])
                for step in proc_steps[arm]:
                    step_name = step["step"]
                    step_run = step["run"]
                    step_suffix = step["suffix"]
                    step_args = step["args"]
                    func_name = "run_" + step_name

                    print(f"Running {arm}/{func_name} ({counter}/{nsteps})", file=old_stdout)
                    func = getattr(recipes, func_name)

                    # When master calibrations are in use, the steps listed in skip_steps
                    # will be skipped.
                    if step_name in extra_skip_steps:
                        print('======================')
                        print(f"Skipping step: {step_name}")
                        print('======================')
                        continue

                    if step_run:
                        print('======================')
                        print(step_name)
                        print('======================')

                        func(
                            obs_metadata,
                            gargs,
                            prev_suffix=prev_suffix,
                            curr_suffix=step_suffix,
                            **step_args,
                        )
                        if step_suffix is not None:
                            prev_suffix = step_suffix

                    else:
                        pass
                    counter = counter + 1
                duration = datetime.datetime.now() - start_time
                # this print goes to the log file, since we don't specify file=old_stdout
                print(f"WiFeS {arm} arm reductions took a total of {duration.total_seconds()} seconds.")

        # Extra Data Reduction Quality Plots:
        # Dome Flats
        try:
            title = 'Dome Flatfield'
            output_plot = os.path.join(gargs['plot_dir_arm'], "raw_domeflat_check.png")

            flat_image_path = os.path.join(master_dir, f"wifes_{arm}_super_domeflat_raw.fits")
            slitlet_path = os.path.join(master_dir, f"wifes_{arm}_slitlet_defs.pkl")
            flatfield_plot(flat_image_path, slitlet_path, title, output_plot)
        except Exception:
            pass

        # Twilight Flats
        try:
            title = 'Twilight Flatfield'
            output_plot = os.path.join(gargs['plot_dir_arm'], "raw_twiflat_check.png")
            flat_image_path = os.path.join(master_dir, f"wifes_{arm}_super_twiflat_raw.fits")
            slitlet_path = os.path.join(master_dir, f"wifes_{arm}_slitlet_defs.pkl")
            flatfield_plot(flat_image_path, slitlet_path, title, output_plot)
        except Exception:
            pass

        print(f"Successfully completed {arm} arm\n")

    except Exception:
        print("")
        print(f"{arm} arm skipped, an error occurred during processing.")
        traceback.print_exc()
        print("")

    return_dict.update(gargs)
# end of run_arm_indiv


def main():
    # Main script
    start_time = datetime.datetime.now()
    print(f"Pipeline started at {start_time}")

    # ------------------------------------------------------------------------
    # DEFINE THE PROCESSING STEPS
    # ------------------------------------------------------------------------

    # --------------------------------------------
    # INITIATE THE SCRIPT
    # --------------------------------------------

    # Initialize ArgumentParser with a description
    parser = argparse.ArgumentParser(
        description="The Python data reduction pipeline for WiFeS."
    )

    # The raw data directory is a required positional argument
    parser.add_argument("user_data_dir", type=str, help="Path to the raw data directory.")

    # Option for specifying the path for the output files
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data_products",
        help="Optional: Path to output folder.",
    )

    # Option for specifying the path to the red parameters JSON file
    parser.add_argument(
        "--red-params",
        type=str,
        help="Optional: Path to the configuration JSON file containing parameters for "
             "reducing the blue arm.",
    )

    # Option for specifying the path to the blue parameters JSON file
    parser.add_argument(
        "--blue-params",
        type=str,
        help="Optional: Path to the configuration JSON file containing parameters for "
             "reducing the blue arm.",
    )

    # Option for triggering the reduction from master calibrations
    parser.add_argument(
        "--from-master",
        type=str,
        const="./master_calib",
        nargs="?",
        help="Optional: Path to the master calibrations directory. If not provided, the"
             " default path will be used: .",
    )

    # Option for specifying to only produce the master calibration files
    parser.add_argument(
        "--just-calib",
        action="store_true",
        help="Optional: Only basics master calibration files will produced.",
    )

    # Option for specifying to skip already completed steps
    parser.add_argument(
        "--skip-done",
        action="store_true",
        help="Optional: Skip already completed steps.",
    )

    # Option for treating OBJECT images near standard stars as standards,
    # even if IMAGETYP != STANDARD
    parser.add_argument(
        "--greedy-stds",
        action="store_true",
        help="Optional: Treat OBJECT as STANDARD if near known standard.",
    )

    # Option to avoid coadding all science frames with the same OBJECT name.
    # Useful for time-series, position shifts, etc.
    parser.add_argument(
        "--coadd-mode",
        choices=["all", "none", "prompt"],
        default="all",
        help="Optional: coadd 'all' (default), 'none', 'prompt' (user-selected, "
             "blue/red separate) frames of a given OBJECT",
        required=False,
        type=str,
    )

    # Option to reduce both blue and red arms simultaneously
    parser.add_argument(
        "--run-both",
        action="store_true",
        help="Optional: Reduce Red and Blue data simultaneously using multiprocessing.",
    )

    # Option for specifying to auto-extract the output datacubes
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Optional: Auto-extract the datacubes.",
    )

    # Option for specifying to auto-extract AND splice the output datacubes
    parser.add_argument(
        "--extract-and-splice",
        action="store_true",
        help="Optional: Auto-extract AND splice the datacubes.",
    )

    # Option for specifying the path to the extraction JSON file
    parser.add_argument(
        "--extract-params",
        type=str,
        help="Optional: Path to the configuration JSON file containing parameters for "
             "extracting the spectra.",
    )

    # Option to skip processing (and only extract or extract-and-splice)
    parser.add_argument(
        "--no-processing",
        action="store_true",
        help="Optional: Skip processing and only extract or extract-and-splice",
    )

    args = parser.parse_args()

    # Validate and process the user_data_dir
    user_data_dir = os.path.abspath(args.user_data_dir)
    if not os.path.exists(user_data_dir):
        raise ValueError(f"No such raw data directory {user_data_dir}")

    if not user_data_dir.endswith("/"):
        user_data_dir += "/"

    # Validate and process the output_dir
    output_dir = os.path.abspath(args.output_dir)
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
        except PermissionError as e:
            raise PermissionError(e)
        except Exception as e:
            print(f"ERROR: {e}")
            raise

    if not output_dir.endswith("/"):
        output_dir += "/"
    print(f"Processing data in directory: {user_data_dir}. Outputting to directory: {output_dir}")

    # Handling reduction parameters.
    params_path = {
        "blue": None,
        "red": None,
    }

    # Red
    if args.red_params:
        params_path["red"] = os.path.abspath(args.red_params)
        print(f"Using red parameters from: {params_path['red']}")

    # Blue
    if args.blue_params:
        params_path["blue"] = os.path.abspath(args.blue_params)
        print(f"Using blue parameters from: {params_path['blue']}")

    # Extraction
    if args.extract_params:
        extract_params_path = os.path.abspath(args.extract_params)
        print(f"Using extraction parameters from: {extract_params_path}")
    else:
        extract_params_path = "./pipeline_params/params_extract.json5"

    # Reduction from master calibration frames
    from_master = args.from_master

    # Only basics master calibration
    just_calib = args.just_calib

    # Set to skip already done files
    skip_done = args.skip_done

    # Auto-extract the datacubes
    extract = args.extract

    # Auto-extract AND splice the datacubes
    extract_and_splice = args.extract_and_splice

    # Skip processing and only extract or extract-and-splice
    no_processing = args.no_processing

    # Creates a directory for plot.
    plot_dir = os.path.join(output_dir, "plots/")
    os.makedirs(plot_dir, exist_ok=True)

    # Creates a temporary data directory containning all raw data for reduction.
    if not no_processing:
        temp_data_dir = os.path.join(output_dir, "intermediate/raw_data_temp/")
        os.makedirs(temp_data_dir, exist_ok=True)

        all_fits_names = get_file_names(user_data_dir, "*.fits*")
        # Copy raw data from user's direcory into temporaty raw directory.
        copy_files(user_data_dir, temp_data_dir, all_fits_names)

        # Classify all raw data (red and blue arm)
        mode_save_fn = os.path.join(output_dir, "coadd_mode.json5")
        obs_metadatas = classify(temp_data_dir,
                                 greedy_stds=args.greedy_stds,
                                 coadd_mode=args.coadd_mode,
                                 mode_save_fn=mode_save_fn)

        # Set grism_key dictionary due to different keyword names for red and blue arms.
        grism_key = {
            "blue": "GRATINGB",
            "red": "GRATINGR",
        }

        # Set the directory for master calibration files (default is ./data_products/master_calib/)
        # Define a list of steps to skip if the reduction is being performed
        # using master calibration files
        if from_master:
            master_dir = os.path.abspath(from_master)
            output_master_dir = os.path.join(output_dir, "master_calib/")
            os.makedirs(output_master_dir, exist_ok=True)
            extra_skip_steps = [
                "superbias",
                "superflat",
                "slitlet_profile",
                "flat_cleanup",
                "superflat_mef",
                "wave_soln",
                "wire_soln",
                "flat_response",
            ]

        else:
            # Master calibration files firectory
            master_dir = os.path.join(output_dir, "master_calib/")
            os.makedirs(master_dir, exist_ok=True)
            output_master_dir = ""
            # No extra skiped steps in principal.
            extra_skip_steps = []

        print(f"Processing using master calibrations {'to' if just_calib else 'from'}: '{master_dir}'.")

        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        jobs = []
        if args.run_both:
            # Try and reduce both arms at the same time
            for arm in obs_metadatas.keys():
                p = multiprocessing.Process(target=run_arm_indiv, args=(temp_data_dir, obs_metadatas, arm, master_dir, output_master_dir, output_dir, params_path, grism_key, just_calib, plot_dir, from_master, extra_skip_steps, return_dict, skip_done))
                jobs.append(p)
                p.start()

            for proc in jobs:
                proc.join()
        else:
            # Otherwise reduce them sequentially
            for arm in obs_metadatas.keys():
                p = multiprocessing.Process(target=run_arm_indiv, args=(temp_data_dir, obs_metadatas, arm, master_dir, output_master_dir, output_dir, params_path, grism_key, just_calib, plot_dir, from_master, extra_skip_steps, return_dict, skip_done))
                jobs.append(p)
                p.start()

                for proc in jobs:
                    proc.join()

        # Delete temporary directory containing raw data.
        shutil.rmtree(temp_data_dir)

    # ----------------------------------------------------------
    # Move reduce cube to the output directory
    # ----------------------------------------------------------
    if just_calib:
        print("Only basics master calibration files have been produced.")
    else:
        if not no_processing:
            print(f"Moving reduced 3D cubes to {output_dir}.")

            # Red
            red_cubes_path = os.path.join(output_dir, "intermediate/red/")
            red_cubes_file_name = get_file_names(red_cubes_path, "*.cube.fits")
            # Move reduced cubes to the output directory
            move_files(red_cubes_path, output_dir, red_cubes_file_name)

            # Blue
            blue_cubes_path = os.path.join(output_dir, "intermediate/blue/")
            blue_cubes_file_name = get_file_names(blue_cubes_path, "*.cube.fits")
            # Move reduced cubes to the output directory
            move_files(blue_cubes_path, output_dir, blue_cubes_file_name)

        if extract or extract_and_splice:
            # ----------------------------------------------------------
            # Find all reduced cubes in the output directory (omit spliced cubes)
            # ----------------------------------------------------------
            reduced_cubes_paths = [os.path.join(output_dir, fname) for fname
                                   in get_file_names(output_dir, "*.cube.fits")
                                   if "Splice" not in fname]

            # ----------------------------------------------------------
            # Match cubes from the same observation based on DATE-OBS
            # ----------------------------------------------------------
            matched_cubes = cube_matcher(reduced_cubes_paths)

            if matched_cubes is not None:
                # ----------------------------------------------------------
                # Read extraction parameters from JSON file
                # ----------------------------------------------------------
                extract_args = load_config_file(extract_params_path)

                # ----------------------------------------------------------
                # Loop over matched cubes list
                # ----------------------------------------------------------
                for match_cubes in matched_cubes:
                    # ----------
                    # Extraction
                    # ----------
                    print('======================')
                    print(f'Extracting spectra for {match_cubes["file_name"].replace(".cube", "")}')
                    print('======================')
                    blue_cube_path = match_cubes["Blue"]
                    red_cube_path = match_cubes["Red"]
                    print(f'Found blue={blue_cube_path} and red={red_cube_path}')
                    plot_name = match_cubes["file_name"].replace(".cube", "_detection_plot.png")
                    plot_path = os.path.join(plot_dir, plot_name)
                    if blue_cube_path is None:
                        taros = is_taros(red_cube_path)
                        ns = is_nodshuffle(red_cube_path)
                        subns = is_subnodshuffle(red_cube_path)
                    else:
                        taros = is_taros(blue_cube_path)
                        ns = is_nodshuffle(blue_cube_path)
                        subns = is_subnodshuffle(blue_cube_path)

                    # Run auto-extraction
                    detect_extract_and_save(
                        blue_cube_path,
                        red_cube_path,
                        output_dir,
                        ns=ns,
                        subns=subns,
                        plot_path=plot_path,
                        **extract_args['detext_args'],
                    )

                    # Plot extracted spectra:
                    if extract_args["plot"]:
                        if blue_cube_path is not None:

                            blue_pattern = blue_cube_path.replace("cube.fits", "spec.det*")
                            blue_specs = sorted(glob.glob(blue_pattern))

                            for blue_spec in blue_specs:
                                plot_1D_spectrum(blue_spec, plot_dir=plot_dir)

                        if red_cube_path is not None:

                            red_pattern = red_cube_path.replace("cube.fits", "spec.det*")
                            red_specs = sorted(glob.glob(red_pattern))

                            for red_spec in red_specs:
                                plot_1D_spectrum(red_spec, plot_dir=plot_dir)

                    if not extract_and_splice:
                        continue

                    # ------------------------------------
                    # Splice only paired cubes and spectra
                    # ------------------------------------

                    spliced_output = []
                    if blue_cube_path is not None and red_cube_path is not None:
                        print('======================')
                        print('Splicing blue and red cubes')
                        print('======================')

                        blue_cube_name = os.path.basename(blue_cube_path)
                        red_cube_name = os.path.basename(red_cube_path)

                        # Get filename of form `xxx-Splice-UTxxx.cube.fits`
                        if taros:
                            if re.search("T2m3wb", blue_cube_name):
                                spliced_cube_name = blue_cube_name.replace("T2m3wb", "T2m3wSplice")
                            else:
                                spliced_cube_name = "Splice_" + blue_cube_name
                        else:
                            spliced_cube_name = blue_cube_name.replace("Blue", "Splice")
                        spliced_cube_path = os.path.join(output_dir, spliced_cube_name)

                        # Splice cubes
                        if os.path.isfile(spliced_cube_path) and \
                            os.path.getmtime(blue_cube_path) < os.path.getmtime(spliced_cube_path) and \
                                os.path.getmtime(red_cube_path) < os.path.getmtime(spliced_cube_path):
                            print(f"Skipping splicing of cube {spliced_cube_name}. Output newer than input.")
                        else:
                            splice_cubes(match_cubes["Blue"], match_cubes["Red"], spliced_cube_path, **extract_args['splice_args'])

                        # Find blue spectra files matching the pattern 'xxx-Blue-UTxxx.spec.det*'
                        pattern_blue = os.path.join(
                            output_dir, blue_cube_name.replace("cube", "spec.det*")
                        )
                        blue_specs = sorted(glob.glob(pattern_blue))

                        # Find red spectra files matching the pattern 'xxx-Red-UTxxx.spec.det*'
                        pattern_red = os.path.join(
                            output_dir, red_cube_name.replace("cube", "spec.det*")
                        )
                        red_specs = sorted(glob.glob(pattern_red))

                        # Splice spectra
                        for blue_spec, red_spec in zip(blue_specs, red_specs):
                            # Generate filename for spliced spectrum 'xxx-Splice-UTxxx.spec.det*.fits'
                            if taros:
                                if re.search("T2m3wb", blue_spec):
                                    spliced_spectrum_name = os.path.basename(blue_spec).replace(
                                        "T2m3wb", "T2m3wSplice"
                                    )
                                else:
                                    spliced_spectrum_name = "Splice_" + os.path.basename(blue_spec)
                            else:
                                spliced_spectrum_name = os.path.basename(blue_spec).replace(
                                    "Blue", "Splice"
                                )
                            spliced_output.append(os.path.join(
                                output_dir, spliced_spectrum_name
                            ))
                            if os.path.isfile(spliced_output[-1]) and \
                                os.path.getmtime(blue_spec) < os.path.getmtime(spliced_output[-1]) and \
                                    os.path.getmtime(red_spec) < os.path.getmtime(spliced_output[-1]):
                                print(f"Skipping splicing of spectrum {spliced_output[-1]}. Output newer than input.")
                            else:
                                splice_spectra(blue_spec, red_spec, spliced_output[-1], **extract_args['splice_args'])

                    # Plot extracted spectra:
                    if extract_args["plot"]:
                        for spliced_spec in spliced_output:
                            if os.path.isfile(spliced_spec):
                                plot_1D_spectrum(spliced_spec, plot_dir=plot_dir)

    # ----------------------------------------------------------
    # Print total running time
    # ----------------------------------------------------------
    end_time = datetime.datetime.now()
    duration = end_time - start_time
    message = "All done in %.01f seconds." % duration.total_seconds()
    print(message)
    print('\U0001F52D', message, '\u2B50')


if __name__ == "__main__":
    main()
