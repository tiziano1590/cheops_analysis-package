.. _installation:

Installation
============

Before proceeding with the installation, please download and install a python 3 Anaconda distribution on their `Anaconda website <https://www.anaconda.com/>`_.

Create the appropriate environment
----------------------------------


It is suggestted to create a separate anaconda environment to proceed with the installation:

.. code-block::

        conda create -n cheope python==3.8 numpy scipy matplotlib pandas


After creating a conda environmnent called `cheope` and installed the basic libraries `numpy`, `scipy`, `matplotlib` and `pandas`, activate the environment:

.. code-block::

        conda activate cheope


Before installing cheope, install `cython` and `mpi4py` using conda:

.. code-block::

        conda install cython mpi4py


Download and Install
--------------------

Download with PyPI:
^^^^^^^^^^^^^^^^^^^

Simply:

Before installing ``cheope``, install pycheops-ultra with

.. code-block::

        pip install pycheops-ultra

And, then:

.. code-block::

        pip install cheope


Download from GitHub:
^^^^^^^^^^^^^^^^^^^^^

Before Installing ``cheope`` install pycheops-ultra from its GitHub repository:

.. code-block::

        git clone https://github.com/tiziano1590/pycheops

        cd pycheops


switch to the parallel branch:

.. code-block::

        git checkout ultranest


and install it:

.. code-block::

        pip install -e .


.. code-block::

        git clone https://github.com/tiziano1590/cheops_analysis_package


go to your local Cheope repository and install it with the following command:

.. code-block::

        pip install -e .
