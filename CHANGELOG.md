# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [main - 2.1.0] - 2025-06-26

Minor enhancements for added flexibility.

### Changed

- Installation
  - Added Windows instructions

- User controls
  - Option to specify output directory as command-line argument

- Input file compatibility
  - Allow standard star inputs of a different binning to the science data

## [main - 2.0] - 2025-05-22

Public release back into the main branch with a range of upgrades. With contributions from Christopher Onken, Brent Miszalski, Felipe Jimenez-Ibarra, and more.

### Changed

- Requirements
  - Update dependencies to photutils >= 2 (requiring scipy requirement update) and numpy >= 2.

- User controls
  - Simplify JSON5 files by removing separate nod & shuffle variants
  - Option to reduce arms simultaneously
  - Add option to avoid coadding all science frames with the same OBJECT name (useful for time-series, position shifts, mosaics, etc.)
  - Associate any sky exposures with science exposures if IMAGETYP = 'SKY' (currently a manual modification of the image header by the user) and OBJECT keywords match. Use of `--coadd-mode prompt` allows manual association of SKY and OBJECT exposures. Use of `--coadd-mode none` blindly associates one SKY and one OBJECT using the order they appear in the image list. Multiple sky exposures are scaled by their exposure times and median-combined. Subtraction from science frame scales effective sky exposure time to that of science frame.
  - Allow user to separate Nod & Shuffle observations and process independently rather than subtracting
  - Make greedy definition of STANDARD stars rely on command-line argument to enable (rough) sky proximity to known standard, otherwise (new default) require IMAGETYP keyword to be 'STANDARD'

- Data handling
  - Handle compressed raw data inputs (fpacked [.fz] or gzipped [.gz], just using astropy.io.fits capabilities)
  - Does not search for raw files in subfolders (making it easier to hide away data that shouldn't be processed)
  - Accommodate Full-Frame and Stellar (half-frame) readout regions for both Automated 2.3m data and older (Taros) data
  - Accommodate nights of mixed Classical and Nod & Shuffle data
  - Updated bad pixel mask
  - Handle saturated pixels in raw frames (flag in VAR and DQ frames for SCIENCE, STANDARD, SKY images; interpolate over for other image types).
  - Add option to set flux pixels to NaN if masked in DQ extenstion
  - Improve handling of coadded images, especially when exposure times differ between arms
  - Propagate various calibration keywords into science data
  - Make skip-done option more comprehensive in avoiding repetition, and check if input is newer than output to also trigger running of step

- Calibration
  - Optional masking of Littrow ghosts in dome and sky flats
  - Avoid elevated overscan levels for rows with high counts, and estimate the overcan for those rows with Chebyshev polynomial determined from interslice regions
  - Coadd arc and wire frames
  - Chunk the imcombine process into x-axis sections if equivalent of > 5 unbinned images
  - Use enhanced flatfield algorithm when lacking twilight flats
  - New NeAr and CuAr line list from NIST (Ritz wavelengths in air)
  - Option to output datacubes in vacuum wavelengths instead of air
  - Improve handling of flux vs. telluric standards
  - Update standard stars to use Nov 2024 edition of CALSPEC, where available
  - Save smooth fit to flatfield (reflecting CCD, grating, beamsplitter throughput) and temporarily remove when fitting flux standard to reduce amplitude of fluctuations
  - Save sky spectrum from telluric standard to allow wavelength refinement when correcting science spectra
  - Correct atmospheric differential refraction calculation to be valid for y-binning options x2, and update standard weather parameters based on SkyMapper records

- Spectral extraction
  - Make automatic extraction and splicing rely on the presence of a command-line argument
  - Updated the extraction parameters JSON5 file (`./pipeline_params/params_extract.json5`), making it more structured and including the parameters required for the new routine.
  - Introduced the `--extract-params` command-line argument, allowing users to pass a tailored extraction configuration file. This is particularly useful for local users who need to adjust detection and extraction parameters (e.g., for detecting faint sources, changing the default aperture size for extraction).
  - Improved source detection routine by implementing DAOStarFinder (`photutils`), which identifies sources based on local density maxima and fits a 2D Gaussian model.
  - Allow user to propagate new extensions into datacubes and extracted spectra: data quality, sky spectrum, wavelength, applied telluric spectrum, extinction correction
  - New options for sky spectrum creation (science-sized aperture offset along the same IFU slice vs. annulus; median vs. weighted mean), and allow user to subtract sky from Nod & Shuffle frames
  - Allow user to just extract or extract-and-splice existing datacubes without regenerating intermediate files
  - Allow user to only extract spectra from datacubes without also splicing the cubes and spectra
  - Allow user to splice cubes and spectra that do not overlap in wavelength

## [automation - 0.7.3] - 2024-08-07

A major revision for the new WiFeS data being produced by the automated 2.3m telescope. With contributions from Cristina Martinez-Lombilla, Felipe Jimenez-Ibarra, Brent Miszalski, Tim Davies, and more.

### Changed

- Updated to Python 3, with bug fixes and headers for the new telescope setup.
- Cleaner repository (defunct scripts and directories were removed).
- Pip installable.
- Adjusted to read and reduce data automatically all at once in both arms and for all the observing configurations, including:
    - `classic` mode.
    - `nod-and-shuffle` mode.
    - `sub-nod-and-shuffle` mode.
    - Stellar frame configuration (that is, only the middle part of the detector is used).
    - Binning along the spatial axis.
- New `.JSON` config files provided for each grating. 
  - The pipeline chooses the template automatically.
  - Users don't need to set anything in generate metadata or reduction scripts anymore.
  - Users can create their own `.JSON` file following the same structure as their preferred setup.
- Logger file to track the data usage and pipeline performance.
- Astrometry is now implemented in the data cubes although the accuracy is uncertain.
- Extraction and splice of the spectra and splice of the 3D astrometrised cubes are now implemented.
- Multiprocessing can be enabled so the pipeline can run faster (see caveats below). 
- Multiple quality plots are automatically generated and saved.
- Outputs organised in a new `/data_products` directory.

## [master - 0.7.3+] - 2021-10-20

Many changes by many people, including Frederic Vogt, Jon Nielsen, Aaron Rizzuto, Mike Ireland, Adam Rains, and Marusa Zerjal.

## [master - 0.7.0] - 2015-02-27

Initial commit by Frederic Vogt, with contributions from Mike Childress, Jon Nielsen, Rob Sharp, Mike Bessell, Raul Cacho, Rebecca Davies, Catherine de Burgh-Day, I-Ting Ho, Wolfgang Kerzendorf, Sarah Leslie, Melissa Ness, David Nicholls, Chris Owen, Sinem Ozbilgen, Matthew Satterthwaite, Marja Seidel, Tiantian Yuan, and Jabran Zahid.

