.. cheope documentation master file, created by
   sphinx-quickstart on Fri Dec 10 12:59:21 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to cheope's documentation!
==================================

.. toctree::
    :hidden:
    :maxdepth: 2
    :caption: Contents:

    Home <self>
    User Guide <user/index>
    Development Guide <devel/index>
    Plugin Catalogue <plugins_cata>
    Full API <api/modules>
    Citation <citation>
    Project Page <https://github.com/tiziano1590/cheops_analysis-package>


cheope - interpreting lightcurves from any dataset
=======================================================

    |image1|

``cheope`` is an open-source fully bayesian code able to interpret 
light-curve datasets licensed under BSDv3. It runs using all the pycheops functionalities.

It allows to use CHEOPS, TESS and Kepler/K2 dataset using built-in function.
In alternative, one can use a owned ascii file with a light-curve and fit it
with cheope's ascii module.

``cheope`` can be easily modified thanks to its object-oriented structure. Can be expanded
and one day it will be conscious.



If you use ``cheope`` in you work, please see the guide to :ref:`Citations`.


-----------------------------------

Installation :ref:`installation`

``cheope`` quickstart: :ref:`quickstart`

----------------------------------

:Release: |release|

.. |image1| image::  _static/cheope_logo.jpg
    :align: middle
    :height: 180


.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
