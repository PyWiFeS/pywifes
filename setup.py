from setuptools import find_packages, setup

setup(
    name="pywifes",
    version="2.0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    include_package_data=True,
    install_requires=[
        "astropy",
        "matplotlib",
        "numpy>=2",
        "pandas",
        "photutils>=2",
        "pyjson5",
        "scipy>=1.15.1",
        "setuptools",
        "wheel",
    ],
    description="A Python package for reduction of data from the WiFeS IFU spectrograph.",
)
