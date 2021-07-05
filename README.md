### Download and Install
Download from GitHub:
```
git clone https://github.com/tiziano1590/cheops_analysis_package
```

go to your local Cheope repository and install it with the following command:
```
pip install -e .
```

### Usage
To use it, simply digit the command:
```
cheope -i path/to/parameters/file.yml
```

### Run analysis for a Single Visit observation and model selection with Bayes Factor
Cheope can run a single visit analysis for a transit observation, compares several models with different
parameters and computes a Bayes factor for each of them, comparing them with the simple transit model without parameters.

To run Cheope in this configuration use the command:

```
cheope -i path/to/parameters/file.yml -sb
```