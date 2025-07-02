# PyWiFeS 

PyWiFeS is the automated Python data reduction pipeline for WiFeS, the Wide Field Spectrograph, an optical integral field spectrograph for the ANU 2.3m telescope at Siding Spring Observatory. 

WiFeS has a field of view of 25x38 arcseconds, with R=3000 VPH gratings that cover the full optical wavelength range in a single exposure, as well as R=7000 VPH gratings that offer higher spectral resolution for smaller wavelength ranges. WiFeS was described in two papers led by the Principal Investigator, the late Michael Dopita, in [2007](https://ui.adsabs.harvard.edu/abs/2007Ap%26SS.310..255D/abstract) and [2010](https://ui.adsabs.harvard.edu/abs/2010Ap%26SS.327..245D/abstract).

The current version of PyWiFeS is compatible with data from both the automated 2.3m operations and the previous manual (in-person or remote) operations through TAROS. The upgrade of the 2.3m telescope for automated observing was described by [Price et al. (2024)](https://ui.adsabs.harvard.edu/abs/2024PASA...41...57P/abstract). The original version of PyWiFeS was described by [Childress et al. (2014)](https://ui.adsabs.harvard.edu/abs/2014Ap%26SS.349..617C/abstract). A publication describing PyWiFeS version 2 is in preparation.

The development of PyWiFeS version 2 was made possible by grant LE230100063 from the Australian Research Council.

## User Manual
Documentation for the current version of the pipeline has been deployed to [this ReadTheDocs site](https://pywifes-pipeline.readthedocs.io/en/latest/).

For even more information, we refer the users to the [old PyWiFeS User Manual](https://www.mso.anu.edu.au/pywifes/doku.php?id=documentation). That manual explains the general structure of the original version of the pipeline, the steps of the data reduction as performed then, and technical details about the Python modules and functions, many of which are still applicable. **However, please follow the installation and usage instructions here, rather than the old instructions there.**

## Installation

### MacOS/Linux/Unix

1. Download or clone this branch of the `pipeline` repository:
    ```sh
    git clone -b main https://github.com/PyWiFeS/pywifes.git
    ```
2. Set up a python environment (via conda, for example) with:

    -python >= 3.10

    -scipy >= 1.15.1

    -numpy >= 2.0

    -pip

3. From the pipeline root directory, run:
    ```sh
    pip install .
    ```
    If you've installed the pipeline this way, you may need to unset your `PYTHONPATH` environment variable (or at least remove the path to the download directory from `PYTHONPATH`).
4. Point the environment variable `PYWIFES_DIR` to your reference data directory. There are a few possible ways to do that:
    1. In your conda env, run the following:

    ```sh
    conda env config vars set PYWIFES_DIR=/your/path/to/pywifes/reference_data

    conda env config vars unset PYTHONPATH
    ```

    Then deactivate and reactivate the conda env.
    2. Add the following line to your `~/.bashrc` so it will run on login:

    ```sh
    export PYWIFES_DIR=/your/path/to/pywifes/reference_data
    ```
    3. Or run the command manually before 'Running the Pipeline'.
    4. Alternatively, if `PYWIFES_DIR` is not set, the pipeline searches the program's *install* directory.
    For this approach to work, you would instead need to install with `pip install -e .`
5.  If desired, set up an alias for the main reduction routine `reduce_data.py`:

    ```sh
    alias pywifes-reduce='/.../pywifes/reduction_scripts/reduce_data.py'
    ```    

### Windows

Testing of installation on Windows 11 has been minimal, but the following approach has worked for at least one user.

1. Download and install Anaconda/Miniconda.
2. Within the Anaconda/Miniconda app, install the Powershell app.
3. Launch Powershell and create the conda environment, e.g.:

    ```sh
    conda create -n pywifes python=3.13 pip git setuptools wheel numpy=2 scipy=1.15.1 photutils astropy matplotlib pandas
    conda activate pywifes
    ```
4. Retrieve the pipeline from the repository (one can also download a zip file from GitHub). The `git clone` command will create a `pywifes` directory in your working directory, then complete the environment setup:

    ```sh
    git clone -b main https://github.com/PyWiFeS/pywifes.git
    conda env config vars set PYWIFES_DIR=C:\your\path\to\pywifes\reference_data
    conda env config vars unset PYTHONPATH
    conda deactivate
    conda activate pywifes
    cd pywifes
    pip install .
    ```

## Running the Pipeline
Put all raw data and calibration files in a dedicated directory, e.g. `/.../working_directory/my_raw_data`. Then launch the pipeline as below.

### MacOS/Linux/Unix
Run the main reduction routine, giving the raw data directory path as an input parameter. The pipeline will run both arms automatically and choose the observing mode by checking the headers.

```sh
pywifes-reduce my_raw_data [other options]
```

### Windows
Run the main reduction routine, giving the raw data directory path as an input parameter. The pipeline will run both arms automatically and choose the observing mode by checking the headers.

```sh
cd reduction_scripts
python -m reduce_data my_raw_data [other options]
```

## Important Considerations
The pipeline cannot handle mixed instrument configurations. Input raw_data folders should contain only a single combination of blue grating + beam splitter + red grating (bias frames are the exception, which may use any setup). 

All data should use the same CCD binning, with the exception of standard star observations. Following overscan subtraction and the conversion from ADU to electron units, standard star observations (usually taken in 1x2 binning in the automated observations of the 2.3m telescope) will be converted to the same binning as the science data by block-summing or distributing the electrons over the appropriate number of pixels in each axis (depending on whether the science data is binned more than or less than the standards).

In addition, the *science* data should all use either Full or Stellar (half-frame) readout regions. Full field calibration data (including standard star observations) will automatically be cut to Stellar size if the science data is Stellar. 

The pipeline will appropriately handle mixed Classical and Nod & Shuffle (or Sub Nod & Shuffle) science observations, as long as the gratings and beam splitter match the available calibrations.

A rule-of-thumb calibration dataset would include:
1. bias >= 8 frames
2. flat >= 9 frames (including from the night of the science data; showing no signs of shifts in the fringing pattern in the red arm)
3. skyflat >= 6 frames
4. arc >= 3 frames (from the night of the science data)
5. wire >= 3 frames (from the night of the science data)
6. standard >= 2 frames of one star (from the night of the science data, if possible) having IS_TELLURIC = 1 in reference_data/stdstar_lookup_table.dat

Dark frames are available in the data archive, but the dark current is ~1 ADU/hour/pixel and so any correction estimated from a limited set of darks would likely only add nose. Use of the darks is thus not recommended.

### User-Defined Reduction Parameters

**Specify an output directory**

The default output directory is a `data_products` folder created in whatever directory the pipeline is launched from. To specify an alternative directory, use the `--output-dir` flag as follows:

    pywifes-reduce my_raw_data --output-dir /.../data_products_YYYYMMDD

**Set reduction steps**

To specify the reduction steps for blue and red data other than the defaults, users can provide the paths to the respective JSON or JSON5 files using the `--red-params` and `--blue-params` flags as follows:
   
    pywifes-reduce my_raw_data --red-params /.../user_red_param_file.json5 --blue-params /.../user_blue_param_file.json5

The default parameters (annotated, with various options) are normally read by the pipeline from pip's site-packages/pywifes/pipeline_params/[blue,red]/ directory. The files in the installation directory are copied by pip and changes in the installation directory will not be seen by the pipeline without reinstalling.

**Reduce Both Arms in Parallel**

Processing may be sped up by processing both arms simultaneously. Obviously, this entails utilising more of the machine's resources. To enable, use the `--run-both` flag as follows:

    pywifes-reduce --run-both my_raw_data

    
**Reduce Data Using Master Calibration Files**

To perform data reduction using master calibration files from previous reductions, use the `--from-master` flag along with the path to the directory containing all calibration files. Both blue and red data should be stored together in the same directory for coherence. If no path is specified after the `--from-master` flag, the default directory `./data_products/master_calib` is assumed.

    pywifes-reduce my_raw_data --from-master /.../.../master_calibrations_directory


**Controlling How OBJECT Frames Are Coadded**

The default pipeline behaviour is to coadd exposures that share the same OBJECT header keyword, which is unhelpful when there are position shifts of the source in the IFU (due to tracking offsets, dithering, mosaicking, PA changes, etc.) or the observations form a time series. With the `--coadd-mode` command line argument, one may choose:

    `all`: (default) coadd all observations with the same OBJECT name;
    `none`: treat all IMAGETYP='OBJECT' exposures separately;
    `prompt`: prompt the user to select which exposures to coadd together.

With the `prompt` option, the user makes the choice independently for each arm. The choices are saved for the next time the pipeline is run with that dataset (as long as the `--coadd-mode prompt` option is used each time).

Example:

    pywifes-reduce my_raw_data --coadd-mode prompt

**Association of SKY Frames with OBJECT Frames**

While Nod & Shuffle observations automatically subtract the sky in the `run_sky_sub` step, other OBJECT frames need to have one or more SKY frames associated with it. At present, this relies on a manual modification of the FITS headers of such SKY images: set IMAGETYP = 'SKY'. 

Under the default `--coadd-mode all` option, associations will be made if IMAGETYP='SKY' and the OBJECT keyword matches that of the science (IMAGETYP='OBJECT') frame. 

If the `--coadd-mode prompt` option is used, the user can associate one or more SKY images with the OBJECT images of their choice. As above, this is done independently for each arm, and the choices are saved in case the pipeline is interrupted.

When `--coadd-mode none` is used, a single sky frame is associated with each science frame having the same OBJECT keyword -- the first sky with the first science, the second sky with the second science, etc.

When multiple sky frames are associated with a science image, they are scaled by their exposure time (to an equivalent EXPTIME of 1 second) and median-combined. The sky frame is scaled to the science exposure time and subtracted.

**Other parameters**

`--skip-done`: Skip already completed steps. This will check for the existence of the output files from each step, as well as whether the output files are newer than the input files -- if either part fails, the step will be executed. Note that this will not skip the extraction or splicing steps, but will overwrite existing files if `--extract` or `--extract-and-splice` are used.

`--just-calib`: Triggers the data reduction in the absence of on-sky data (both science and calibration). It only produces basic calibration files.

`--greedy-stds`: Treat observations vaguely near known standard stars as STANDARD frames even if IMAGETYP = 'OBJECT'. If this option is not set, only IMAGETYP = 'STANDARD' frames are used as standards.

`--extract`: Automatically locate sources in the output datacubes and extract sources. Default parameters defined in JSON5 file (normally in pip's site-packages/pywifes directory):

    /.../pipeline_params/params_extract.json5

The subtracted sky spectrum is included in the output. Users may choose whether to propagate the data quality (DQ), applied telluric correction (TELLURICMODEL), and corrected atmospheric extinction (EXTINCTION) to the output spectrum. Users may choose whether to subtract the residual sky from Nod & Shuffle observations

`--extract-params`: To specify extraction parameters other than the defaults, users can provide the path to the JSON or JSON5 file using the `--extract-params` flag as follows:

    pywifes-reduce my_raw_data --extract --extract-params /.../user_extract_param_file.json5

`--extract-and-splice`: In addition to the extraction described above, splice together the datacubes and extracted spectra. By default, the pipeline uses 2nd-order Lanczos (sinc) interpolation to map the red arm onto the finer wavelength spacing of the blue arm (the red arm wavelength spacing is 60% coarser in the default pipeline setup). The user may specify an alternative wavelength spacing in the extraction JSON5 file, and both arms will be interpolated to that scale.

`--no-processing`: Skip data processing (if, for example, intermediate files have been deleted) and only extract or extract-and-splice the existing datacubes.

### Extra usabilities
#### Multiprocessing
When multiprocessing is enabled, the pipeline *may* do the job faster. This will depend on the operative system used to run the pipeline. The multiprocessing setup is recommended for **Linux** users, as they will see a significant improvement in the computation time. On the other side, Mac OS users might get a similar running time (or just slightly faster) than in one-process mode. 
To enable the multithreading option, please follow these steps:

1. Open the `.json5` file that corresponds to your grating in pip's site-packages/pywifes folder (or in the installation folder and then reinstall with pip). That is, `/pipeline_parms/<arm>/params_<grating>.json5`.
2. Set `"multithread": true` in all the cases. There should be a total of 6 `"multithread"`, 3 for each of the blue and red arms in the following steps: `"step": "wave_soln"`, `"step": "cosmic_rays"`, and `"step": "cube_gen"`.
3. [Optional] Set `max_processes` to the *maximum* number of sub-processes you would like to launch for `"step": "cosmic_rays"`, and `"step": "cube_gen"`. If `-1`, the pipeline will use as many processes as there are hardware & logical cores on your device, which may be larger than the number of *available* cores, e.g. for Slurm users. Limiting the number of sub-processes can improve the efficiency and availability of your device.
4. Run the pipeline following the instructions above.

#### Skip steps 

Some steps in the data reduction process can be skipped by setting `"run": false` in the corresponding step in the `.json` files. However, in some cases, the step cannot be skipped as it is required for the pipeline to continue reducing the data. For example, the wavelength solution is always required for a successful data reduction. Other steps such as the flux calibration, the extraction of the standard star, or the telluric correction can be skipped in case of, for example, missing calibration files. 


### DATA REDUCED
The pipeline will generate the `data_products` directory within the working directory 
`/.../.../working_directory` containing the reduced data, a logger file to track the information from the reduction process, and the following structure (the Splice files only appearing if the `--extract-and-splice` flag is used): 

- data_products
    - `pywifes_logger.log` 
    - `xxx-Blue-UTxxx.cube.fits`
    - `xxx-Red--UTxxx.cube.fits`
    - `xxx-Splice-UTxxx.cube.fits`
    - ... 
    - `xxx-Blue-UTxxx.spec.detN.fits`
    - `xxx-Red--UTxxx.spec.detN.fits`
    - `xxx-Splice-UTxxx.spec.detN.fits`
    
    - plots
        - `UTxxx_detection_plot.png`
        - `xxx-Blue-UTxxx_.spec.detN.png`
        - `xxx-Red-UTxxx_.spec.detN.png`
        - ...
        - `xxx-Splice-UTxxx_.spec.detN.png`
        - ... 
        - blue
            - `bias.png`
            - `flat_response.png`
            - ...
        - red
            - `bias.png`
            - `flat_response.png`
            - ...

    - intermediate
        - blue
            - `xxx-Blue-UTxxx.p00.fits`
            - `xxx-Blue-UTxxx.p01.fits`
            - ...
            - `xxx-Blue-UTxxx.p10.fits`
        - red
            - `xxx-Red--UTxxx.p00.fits`
            - `xxx-Red--UTxxx.p01.fits`
            - ...
            - `xxx-Red--UTxxx.p10.fits`
        - raw_data_temp
            - `xxx-Blue-UTxxx.fits`
            - ...
            - `xxx-Red--UTxxx.fits`
            - ...

    - master_calib 
        - `wifes_blue_<master_calibration>_files.fits`
        - ...
        - `wifes_red_<master_calibration>_files.fits`
        - ...

`data_products` contains the `plots` directory with the final figures of the data reduction: the 2D extracted spectra, the spliced spectra, and the sources detection plots. The figures generated during the calibration steps are saved in a different directory for each arm (`data_products/plots/arm`). 
Then, the `data_products/intermediate` directory with the calibration files generated during the data reduction process and saves the data separately for each red and blue arm. Also in `intermediate` there is the temporary directory `raw_data_temp` aimed to store the raw data and any pre-treated images (e.g. cut down calibration frames to stellar mode size, when needed) during the data reduction process. `raw_data_temp` is automatically removed when the pipeline is successfully completed. 

Finally, we find `data_products/master_calib`, which is a directory with all master calibration files produced in the data reduction. They are stored to be used in further reductions if required.


## Reporting Issues or Suggestions
If you encounter any issues or have suggestions for improving the pipeline, please [**open a new issue**](https://github.com/PyWiFeS/pywifes) in the `issues` tab and fill out the provided template. Your feedback is very valuable!
