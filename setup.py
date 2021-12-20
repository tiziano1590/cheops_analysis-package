#!/usr/bin/env python
from setuptools import find_packages
from distutils.core import setup

packages = find_packages(exclude=("tests", "docs"))

provides = [
    "cheope",
]


requires = []


install_requires = [
    "numpy==1.21.4",
    "matplotlib>=3.5.1",
    "ipython==7.30.1",
    "emcee==3.1.0",
    "pycheops-ultra==1.0.1",
    "ultranest==3.3.0",
    "cython==0.29.24",
    "pathos==0.2.8",
    "mpi4py>=3.0.0",
    "scikit-learn==0.24.2",
    "statsmodels==0.12.2",
    "cryptography==3.4.7",
    "h5py==3.5.0",
    "selenium==3.141.0",
]

console_scripts = ["cheope=cheope.cheope:main"]

entry_points = {
    "console_scripts": console_scripts,
}

classifiers = [
    "Development Status :: 4 - Beta",
    "Environment :: Console",
    "Environment :: Win32 (MS Windows)",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: BSD License",
    "Operating System :: Microsoft :: Windows",
    "Operating System :: POSIX",
    "Operating System :: POSIX :: Linux",
    "Operating System :: Unix",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Topic :: Software Development :: Libraries",
]

# Handle versioning
version = "0.3.0"

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="cheope",
    version=version,
    packages=packages,
    author="Tiziano Zingales",
    author_email="tiziano.zingales@gmail.com",
    license="LICENSE",
    description="CHEOPE: studying transiting exoplanets",
    classifiers=classifiers,
    long_description=long_description,
    url="https://github.com/tiziano1590/cheops_analysis-package/cheope",
    long_description_content_type="text/markdown",
    keywords=["astrophysics", "exoplanets", "photometry", "transit"],
    package_data={"": ["*.txt", "*.rst", "*.dat", "*.csv"]},
    # include_package_data=True,
    entry_points=entry_points,
    provides=provides,
    requires=requires,
    install_requires=install_requires,
)
