# Download and Install

Download from GitHub:

```
git clone https://github.com/tiziano1590/cheops_analysis_package
```

go to your local Cheope repository and install it with the following command:

```
pip install -e .
```

IMPORTANT: For the correct usage of the parallel version of pycheops.
To do so install pycheops tiziano190 repository:

```
git clone https://github.com/tiziano1590/pycheops

cd pycheops
```

switch to the parallel branch:

```
git checkout parallel
```

and install it:

```
pip install -e .
```

# Cheops

In this section we regroup all the commands inherent to the CHEOPS space mission dataset analysis. Here we include some visualisation and analysis options.

### Usage

To use it, simply digit the command:

```
cheope -i path/to/parameters/file.yml
```

### Run initial check of a dataset

Cheope will run a basic analysis of the input dataset, checking the lightcurve and providing some basic statistics about it.
The command to run the basic check is:

```
cheope -i path/to/parameters/file.yml -sc
```

### Run analysis for a Single Visit observation and model selection with Bayes Factor

Cheope can run a single visit analysis for a transit observation, compares several models with different
parameters and computes a Bayes factor for each of them, comparing them with the simple transit model without parameters.

To run Cheope in this configuration use the command:

```
cheope -i path/to/parameters/file.yml -sb
```

### Multivisit run

In this mode, if folds all the input observations and runs a multivisit analysis.
To activate the multivisit mode, run:

```
cheope -i path/to/parameters/file.yml -m
```

### User-defined light curve

`cheope` can run also user-precomputer light curves stored in an ascii file, the minimum file should have three columns with: time, flux and the error on the flux.
Once reformatted the lightcurve into a `.txt` or `.dat` file, it is possible to fit the user-defined lightcurve by using the command:

```
cheope -i path/to/parameters/file.yml -a
```

# TESS

In this section we explore the possible commands to analyise TESS-like datasets

### Run analysis for a Single Visit including also your Kepler/TESS points

A normal Single visit run, including Kepler/TESS observation.

The command is:

```
cheope -i path/to/parameters/file.yml -skt
```

# Use of Selenium

`cheope` incorporates a web-browser bot able to download all the datasets related to a particular target.

## The CHEOPS dataset

We bypass the official API (will be included in a future version) and use a human-simulated behaviour to log into the DACE platform and download the target's dataset. To download and run a preliminary check on a planetary system, run:

```
cheope -i path/to/parameters/file.yml --selenium-dace --add-single-check
```

## The TESS dataset

Here there is a list of command to check and analyse some TESS lightcurves.

### download TESS lighcurves and run preliminary check

To run the latest sectors' light curves and run a preliminary check on them:

```
cheope -i path/to/parameters/file.yml --selenium-tess --add-single-kepler-tess --download
```

### Only display the TESS' lighcurves

If you want only display the TESS' lightcurve withough running any check nor analysis, run:

```
cheope -i path/to/parameters/file.yml --selenium-tess --read-fits
```
