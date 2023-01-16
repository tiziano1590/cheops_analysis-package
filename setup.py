#!/usr/bin/env python
import os
from setuptools import find_packages
from distutils.core import setup

packages = find_packages(exclude=("tests", "docs"))

provides = [
    "cheope",
]


requires = []


install_requires = [
    "numpy",
    "matplotlib",
    "ipython",
    "pyyaml",
    "emcee",
    "pycheops-ultra",
    "ultranest",
    "cython",
    "pathos",
    "mpi4py",
    "scikit-learn",
    "statsmodels",
    "cryptography",
    "h5py",
    "selenium",
    "sphinx",
    "sphinx-autoapi",
    "renku-sphinx-theme",
    "pyGTC",
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
# version = "0.3.1"
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, "./cheope/VERSION")) as version_file:
    version = version_file.read().strip()

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
