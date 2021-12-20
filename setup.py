#!/usr/bin/env python
from setuptools import find_packages
from distutils.core import setup

# packages = find_packages(exclude=('tests', 'doc'))

provides = ['cheope', ]


requires = []


install_requires = []

console_scripts = ['cheope=cheope.cheope:main']

entry_points = {'console_scripts': console_scripts, }

classifiers = [
    'Development Status :: 4 - Beta',
    'Environment :: Console',
    'Environment :: Win32 (MS Windows)',
    'Intended Audienc~/opte :: Developers',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: BSD License',
    'Operating System :: Microsoft :: Windows',
    'Operating System :: POSIX',
    'Operating System :: POSIX :: Linux',
    'Operating System :: Unix',
    'Operating System :: OS Independent',
    'Programming Language :: Python',
    'Topic :: Astrophysics/Exoplanets/Transit Photometry',
    'Topic :: Software Development :: Libraries',
]

# Handle versioning
version = '0.1.0'

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='cheope',
    version=version,
    packages=["cheope"],
    author='Tiziano Zingales',
    author_email='tiziano.zingales@gmail.com',
    license="LICENSE",
    version=version,
    description='CHEOPE: studying transiting exoplanets',
    classifiers=classifiers,
    long_description=long_description,
    url='https://github.com/tiziano1590/cheops_analysis-package/cheope',
    long_description_content_type="text/markdown",
    keywords = ['astrophysics', 'exoplanets', 'photometry', 'transit'],
    package_data={"": ["*.txt", "*.rst", "*.dat", "*.csv"]},
    # include_package_data=True,
    entry_points=entry_points,
    provides=provides,
    requires=requires,
    install_requires=install_requires
    )
