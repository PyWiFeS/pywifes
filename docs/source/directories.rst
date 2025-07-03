.. _directories:

Directory Structure
===================

The pipeline generates a directory structure to store the **reduced data**. The directory structure is created within the working directory specified by the user. The working directory is the directory where the pipeline is executed. 

Unless otherwise specified by the user, the pipeline will generate the `data_products` directory within the working directory `/.../working_directory` containing the reduced data, a logger file to track the information from the reduction process, and the following structure: 

- data_products
    - `blue.log` 
    - `red.log` 
    - `xxx-Blue-UTxxx.cube.fits`
    - `xxx-Red-UTxxx.cube.fits`
    - `xxx-Splice-UTxxx.cube.fits`
    - ... 
    - `xxx-Blue-UTxxx.spec.detx.fits`
    - `xxx-Red-UTxxx.spec.detx.fits`
    - `xxx-Splice-UTxxx.spec.detx.fits`
    
    - plots
        - `UTxxx_detection_plot.png`
        - `xxx-Blue-UTxxx_.spec.det1.png`
        - `xxx-Red-UTxxx_.spec.det1.png`
        - ...
        - `xxx-Splice-UTxxx_.spec.det1.png`
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
            - `xxx-Red-UTxxx.p00.fits`
            - `xxx-Red-UTxxx.p01.fits`
            - ...
            - `xxx-Red-UTxxx.p10.fits`
        - raw_data_temp
            - `xxx-Blue-UTxxx.fits`
            - ...
            - `xxx-Red-UTxxx.fits`
            - ...

    - master_calib 
        - `wifes_blue_<master_calibration>_files.fits`
        - ...
        - `wifes_red_<master_calibration>_files.fits`
        - ...

`data_products` contains the `plots` directory with the final figures of the data reduction: the 2D extracted spectra, the spliced spectra, and the sources detection plots. The figures generated during the calibration steps are saved in a different directory for each arm (`data_products/plots/[arm]`). 
Then, the `data_products/intermediate` directory with the calibration files generated during the data reduction process and saves the data separately for each red and blue arm. Also in `intermediate` there is the temporary directory `raw_data_temp` aimed to store the raw data and any pre-treated images (e.g. cut down calibration frames to stellar mode size, when needed) during the data reduction process. `raw_data_temp` is automatically removed when the pipeline is successfully completed. 

Finally, we find `data_products/master_calib`, which is a directory with all master calibration files produced in the data reduction. They are stored to be used in further reductions if required.
