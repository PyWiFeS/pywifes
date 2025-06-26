import os

__version__ = '2.0.1'

# If you set the environment variable 'PYWIFES_DIR' then it will be found
mdir = os.getenv('PYWIFES_DIR')

# Otherwise, just look where we are right now, and go from there
if mdir is None:
    # Where are we located ?
    src_dir = os.path.dirname(__file__)
    # Where are the reference data ?
    pywifes_dir = os.path.join(src_dir, '../../')
    metadata_dir = os.path.join(pywifes_dir, 'reference_data/')
else:
    metadata_dir = mdir
