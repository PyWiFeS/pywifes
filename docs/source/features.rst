.. _features:

What's New in PyWiFeS
=====================

This pipeline automates many of the routine tasks involved in data reduction. It is written in Python and is open-source, allowing users to modify and extend the code to suit their needs.
PyWiFeS is an automated and upgraded version of the original PyWiFeS pipeline (see documentation `here <https://www.mso.anu.edu.au/pywifes/doku.php?id=documentation>`_), developed by `Childress et al. (2014) <http://adsabs.harvard.edu/abs/2014Ap%26SS.349..617C>`_


Upgrades from the Previous Version
----------------------------------

Main upgrades from the previous PyWiFeS version:

- Updated to Python 3, with bug fixes and headers for the new telescope setup.
- Pip installable.
- Supports various observing configurations automatically:
    - `classic` mode.
    - `nod-and-shuffle` mode.
    - `sub-nod-and-shuffle` mode.
    - Stellar frame configuration (that is, only half of the detector is used).
    - Any single binning mode, plus standards binned differently from science data.
    - Backwards compatible with data obtained by TAROS.
- New JSON5 config files for each grating, with comments for each user option.
  - The pipeline will choose a template automatically, if not specified by the user.
  - Users don't need to set anything to generate metadata or reduction scripts anymore.
  - Users can create their own JSON5 file following the same structure as their preferred setup.
- Logger file to track data usage and pipeline performance.
- Implemented astrometry in the data cubes. The accuracy could be low (> 2 arcsec) in some cases. 
- Extraction and splicing of the spectra and splicing of the 3D astrometrised cubes are now implemented.
- Added multiprocessing for faster execution.
- Multiple quality plots are automatically generated and saved.
- Organized output directory (`/data_products`), which can be modified by command-line option.


Key Features
------------

- Fully automated data reduction processes.
- Provides tools for quality control and analysis.
- Supports batch processing of multiple objects in a single night.
- Extensible and customizable.
