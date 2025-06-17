.. _main-page:

.. image:: _static/pywifes_logo.jpeg
   :alt: PyWiFeS Logo
   :align: center

Welcome to the PyWiFeS User Manual!
===================================

PyWiFeS is an automated Python data reduction pipeline for the Wide Field Spectrograph (WiFeS). WiFeS is an optical integral field spectrograph for the ANU 2.3m telescope at Siding Spring Observatory.

An overview of the WiFeS instrument itself can be found on its `RSAA webpage <https://rsaa.anu.edu.au/observatories/instruments/wide-field-spectrograph-wifes>`_.
WiFeS has a field of view of 25x38 arcseconds, with R=3000 VPH gratings that cover the full optical wavelength range in a single exposure, as well as R=7000 VPH gratings that offer higher spectral resolution for smaller wavelength ranges. 

WiFeS was described in two papers led by the Principal Investigator, the late Michael Dopita, in `2007 <https://ui.adsabs.harvard.edu/abs/2007Ap%26SS.310..255D/abstract>`_ and `2010 <https://ui.adsabs.harvard.edu/abs/2010Ap%26SS.327..245D/abstract>`_.

PyWiFeS is written in Python and is open-source, allowing users to modify and extend the code to suit their needs. PyWiFeS is compatible with data from the automated 2.3m telescope as well as the previous manual telescope operations through TAROS. The upgrade of the 2.3m for automated observations was described by `Price et al. (2024) <https://ui.adsabs.harvard.edu/abs/2024PASA...41...57P/abstract>`_.

The original version of PyWiFeS was described by `Childress et al. (2014) <https://ui.adsabs.harvard.edu/abs/2014Ap%26SS.349..617C/abstract>`_.
A publication describing PyWiFeS version 2 is in preparation.

Documentation for the current version of the pipeline has been deployed to `this ReadTheDocs site <https://pywifes-pipeline.readthedocs.io/en/latest/>`_.
   
For even more information, we refer the users to the `old PyWiFeS User Manual <https://www.mso.anu.edu.au/pywifes/doku.php?id=documentation>`_. That manual explains the general structure of the original version of the pipeline, the steps of the data reduction as performed then, and technical details about the Python modules and functions, many of which are still applicable. **However, please follow the installation and usage instructions here, rather than the old instructions there.**

The development of PyWiFeS version 2 was made possible by grant LE230100063 from the Australian Research Council.

This documentation will guide you through installation, usage, main features and general information of the pipeline, and will provide you details about the different modules available.

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   installation
   usage
   features
   directories
   data_quality
   modules/modules

Reporting Issues or Suggestions
===============================

If you encounter any issues or have suggestions for improving the pipeline, please `open a new issue on the GitHub repository <https://github.com/PyWiFeS/pywifes/issues>`_.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
