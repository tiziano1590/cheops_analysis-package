### Download and Install

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
