.. _usage:

Running the Pipeline
====================

Quick Start
-----------

The pipeline is designed to be user-friendly and easy to use. The main routine is called `reduce_data.py` and is used to reduce the data. The pipeline will automatically determine the observing mode and the arm used (blue or red) by checking the headers of the raw data. The pipeline will then run the corresponding reduction script for the arm and observing mode.

1. Put all raw data and calibration files in a dedicated directory, e.g., `/.../my_raw_data`.
2. Run the main reduction routine, giving the raw data directory path as an input parameter. The pipeline will run both arms automatically and choose the observing mode by checking the headers.
    .. code-block:: bash

        ./reduce_data.py /.../my_raw_data


User-Defined Reduction Parameters
---------------------------------

**Specify an output directory**

The default output directory is a `data_products` folder created in whatever directory the pipeline is launched from. To specify an alternative directory, use the `--output-dir` flag as follows:
    .. code-block::bash

        ./reduce_data.py /.../my_raw_data --output_dir /.../data_products_YYYYMMDD

**Set reduction steps**

To specify the reduction steps for blue and red data, users can provide the paths to the respective JSON or JSON5 files using the `--red-params` and `--blue-params` flags as follows:
   .. code-block:: bash

        ./reduce_data.py /.../my_raw_data --red-params /.../user_red_param_file.json5 --blue-params /.../user_blue_param_file.json5

The default parameters (annotated, with various options) are normally read by the pipeline from pip's site-packages/pywifes/pipeline_params/[blue,red]/ directory. The files in the installation directory are copied by pip and changes in the installation directory will not be seen by the pipeline without reinstalling.

**Reduce Data Using Master Calibration Files**

To perform data reduction using master calibration files from previous reductions, use the `--from-master` flag along with the path to the directory containing all calibration files. Both blue and red data should be stored together in the same directory for coherence. If no path is specified after the `--from-master` flag, the default directory `./data_products/master_calib` is assumed.
   .. code-block:: bash
      
       ./reduce_data.py /.../my_raw_data --from-master /.../master_calibrations_directory

**Controlling How OBJECT Frames Are Coadded**

The default pipeline behaviour is to coadd exposures that share the same OBJECT header keyword, which is unhelpful when there are position shifts of the source in the IFU (due to tracking offsets, dithering, mosaicking, PA changes, etc.) or the observations form a time series. With the `--coadd-mode` command line argument, one may choose:

    - `all`: (default) coadd all observations with the same OBJECT name;
    - `none`: treat all IMAGETYP='OBJECT' exposures separately;
    - `prompt`: prompt the user to select which exposures to coadd together.

With the `prompt` option, the user makes the choice independently for each arm. The choices are saved for the next time the pipeline is run with that dataset (as long as the `--coadd-mode prompt` option is used each time).

Example:
   .. code-block:: bash

        ./reduce_data.py /.../my_raw_data --coadd-mode prompt

**Association of SKY Frames with OBJECT Frames**

While Nod & Shuffle observations automatically subtract the sky in the `run_sky_sub` step, other OBJECT frames need to have one or more SKY frames associated with it. At present, this relies on a manual modification of the FITS headers of such SKY images: set IMAGETYP = 'SKY'. 

Under the default `--coadd-mode all` option, associations will be made if IMAGETYP='SKY' and the OBJECT keyword matches that of the science (IMAGETYP='OBJECT') frame. 

If the `--coadd-mode prompt` option is used, the user can associate one or more SKY images with the OBJECT images of their choice. As above, this is done independently for each arm, and the choices are saved in case the pipeline is interrupted.

When `--coadd-mode none` is used, a single sky frame is associated with each science frame having the same OBJECT keyword -- the first sky with the first science, the second sky with the second science, etc.

When multiple sky frames are associated with a science image, they are scaled by their exposure time (to an equivalent EXPTIME of 1 second) and median-combined. The sky frame is scaled to the science exposure time and subtracted.

**Other parameters**

- `--skip-done`: Skip already completed steps. This will check for the existence of the output files from each step, as well as whether the output files are newer than the input files -- if either part fails, the step will be executed.
- `--just-calib`: Triggers the data reduction in the absence of on-sky data (both science and calibration). It only produces basic calibration files.
- `--greedy-stds`: Treat observations vaguely near known standard stars as STANDARD frames even if IMAGETYP = 'OBJECT'. If this option is not set, only IMAGETYP = 'STANDARD' frames are used as standards.
- `--extract`: Automatically locate sources in the output datacubes and extract spectra. By default, uses the parameters defined in JSON5 file (normally in pip's site-packages/pywifes directory rather than the install directory):

   .. code-block:: bash

        /.../pipeline_params/params_extract.json5

- `--extract-params`: Specify path to alternative JSON or JSON5 file with extraction parameters.

   .. code-block:: bash

        ./reduce_data.py /.../my_raw_data --extract-params /.../user_extract_param_file.json5

- `--extract-and-splice`: Automatically locate sources in the output datacubes, extract spectra, and splice the datacubes and spectra, using parameters defined in the JSON5 file above. The pipeline uses 2nd-order Lanczos (sinc) interpolation to map the red arm onto the finer wavelength spacing of the blue arm (the red arm wavelength spacing is 60% coarser in the default JSON5 setup). If the inputs are Nod & Shuffle frames, the sky has already been subtracted.
- `--no-processing`: Skip processing of files and use existing datacubes to extract or extract-and-splice.
- `--run-both`: Process both blue and red arms simultaneously (with associated increase in resource usage).

Extra Usabilities
-----------------

**Multiprocessing**

When multiprocessing is enabled, the pipeline *may* do the job faster. This will depend on the operating system used to run the pipeline. The multiprocessing setup is recommended for **Linux** users, as they will see a significant improvement in computation time. On the other hand, Mac OS users might get a similar running time (or just slightly faster) than in one-process mode. 

To enable the multithreading option, please follow these steps:

1. Open the `.json5` file that corresponds to your grating in pip's site-packages/pywifes folder (or in the installation folder and then reinstall with pip). That is, `/pipeline_parms/<arm>/params_<grating>.json5`.
2. Set `"multithread": true` in all the cases. There should be a total of 6 `"multithread"`, 3 for each of the blue and red arms in the following steps: `"step": "wave_soln"`, `"step": "cosmic_rays"`, and `"step": "cube_gen"`.
3. [Optional] Set `max_processes` to the *maximum* number of sub-processes you would like to launch for `"step": "cosmic_rays"`, and `"step": "cube_gen"`. If `-1`, the pipeline will use as many processes as there are hardware & logical cores on your device, which may be larger than the number of *available* cores, e.g., for Slurm users. Limiting the number of sub-processes can improve the efficiency and availability of your device.
4. Run the pipeline following the instructions below.

**Skip steps** 

Some steps in the data reduction process can be skipped by setting `"run": false` in the corresponding step in the `.json5` files. However, in some cases, the step cannot be skipped as it is required for the pipeline to continue reducing the data. For example, the wavelength solution is always required for a successful data reduction. Other steps such as the flux calibration, the extraction of the standard star, or the telluric correction can be skipped in case of, for example, missing calibration files.
