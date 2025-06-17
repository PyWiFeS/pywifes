.. _installation:

Installation
============

PyWiFeS requires an installation environment with python >=3.10, scipy >=1.15.1, numpy >=2, photutils >=2, and pip.

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
