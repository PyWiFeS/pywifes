.. _installation:

Installation
============

PyWiFeS requires an installation environment with python >=3.10, scipy >=1.15.1, numpy >=2, photutils >=2, and pip.

** MacOS/Linux/Unix

To install PyWiFeS, follow these steps:

1. Clone the repository (main branch):

   .. code-block:: bash

      git clone -b main https://github.com/PyWiFeS/pywifes.git

2. Navigate to the project directory and install dependencies:

   .. code-block:: bash

      pip install .

If you've installed the pipeline this way, you may need to unset your `PYTHONPATH` environment variable (or at least remove the path to the download directory from `PYTHONPATH`).

3. Set the `PYWIFES_DIR` environment variable to the reference data directory in your installation folder:

   .. code-block:: bash

      export PYWIFES_DIR=/.../pywifes/reference_data

4. If desired, set up an alias for the main reduction routine:

   .. code-block:: bash

      alias pywifes-reduce='/.../pywifes/reduction_scripts/reduce_data.py'

** Windows

Testing of installation on Windows 11 has been minimal, but the following approach has worked for at least one user.

1. Download and install Anaconda/Miniconda.
2. Within the Anaconda/Miniconda app, install the Powershell app.
3. Set up the conda environment in Powershell:

   .. code-block:: bash

      conda create -n pywifes python=3.13 pip git setuptools wheel numpy=2 scipy=1.15.1 photutils astropy matplotlib pandas
      conda activate pywifes

4. Retrieve the pipeline from the repository (one can also download a zip file from GitHub). The `git clone` command will create a `pywifes` directory in your working directory, then complete the environment setup:

   .. code-block:: bash

      git clone -b main https://github.com/PyWiFeS/pywifes.git
      conda env config vars set PYWIFES_DIR=C:\your\path\to\pywifes\reference_data
      conda env config vars unset PYTHONPATH
      conda deactivate
      conda activate pywifes
      cd pywifes
      pip install .
