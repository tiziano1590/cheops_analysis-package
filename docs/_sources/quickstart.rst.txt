.. _quickstart:

Basic usage
===========

In these section we show the possible parametrs file to be used in different configurations. 
Use these tmaplates as example and modify with your destination folder and physical parameters:


Single Visit
^^^^^^^^^^^^

.. code-block::

        main_folder: /absolute/path/to/analysis/folder
        visit_number: 1 # starting from 1
        file_key: name of your dace file key # example CH_PR100015_TG006701_V0200
        aperture: default # default, optimal, rinf, rsup
        shape: fit # fit or fix, default fit
        seed: 42 # a int number
        optimizer: emcee # emcee or ultranest
        star:
          star_name: WASP-106
          Rstar: [1.418, 0.0190]
          Mstar: [1.262, 0.052]
          teff: [6265, 36] # None
          logg: [4.38, 0.04]
          feh : [0.15, 0.03]
        planet:
          P: [9.289715, 0.000010] # period in days
          D: [0.00642, 0.00018] # flux depth
          #k: None # = Rp/Rstar, None, omitted or as D, D stronger than k
          # or Rp in Rearth
          # provide:
          # {inc, aRs, b}
          # or
          # {inc, aRs} => b
          # or
          # {b, aRs} ==> inc
          # or
          # {b, inc} ==> aRs
          #b: None
          inc: [89.49, 0.64] # degrees
          aRs: [14.20, 0.43]
          # if total duration T14 in days provided use it for W = T14/P, 
          # otherwise computed from
          # {k, b, aRs}
          #T14: None
          ecc: None
          w: None # degrees
          T_0: 
            fit: True
            value: [0.7, 0] # CHANGE IT ACCORDINGLY TO YOUR VISIT
            bounds: [0.57, 0.85] # CHANGE IT ACCORDINGLY TO YOUR VISIT
          Kms: [165.3, 4.3] # K_RV in m/s
        # emcee parameters: NOT USED IF optimizer is ultranest
        emcee:
          nwalkers: 256
          nprerun : 2
          nsteps  : 7
          nburn   : 0
          nthin   : 1
          nthreads: 2
          progress: True
        # ultranest parameters: NOT USED IF optimizer is emcee
        ultranest:
          live_points: 120
          cluster_num_live_points: 40
          tol: 2.0
          adaptive_nsteps: False   #  False, 'proposal-distance', 'move-distance'
          resume: overwrite ## resume should be one of 'overwrite' 'subfolder', 'resume' or 'resume-similar'
    
The T_0 parameters, initially, is assumed and it can be checked using the SingleCheck ``cheope``'s class. To check the dataset and using a T_0 that actually makes sense,
launche the ``cheope``'s check using the following command:

.. code-block::

        cheope -i path/to/parameters/file.yml -sc
        
After it finishes, ``cheope`` created a folder inside the main folder defined in the parameters' file. Looking at the plots that it generated, you can decide what's the best value for the 
central transit time and modify the T_0 value accordingly. 

Once you modified the T_0 value and boundaries, you can launch ``cheope`` again to fit the light curve using the following command.

.. code-block::

        cheope -i path/to/parameters/file.yml -sb


Multivisit
^^^^^^^^^^
In the following we followed an example of Multivisit usage after analysed separately three data sets in Single visit using the previous section.

.. code-block::
        
        main_folder: /absolute/path/to/analysis/folder
        datasets:
          v1:
            file_name: /absolute/path/to/visit_01_your_first_dataset_single_visit_analysis_folder/your_planet.tgz.dataset
          v2:
            file_name: /absolute/path/to/visit_02_your_second_dataset_single_visit_analysis_folder/your_planet.tgz.dataset
          v3:
            file_name: /absolute/path/to/visit_03_your_third_dataset_single_visit_analysis_folder/your_planet.tgz.dataset
        seed: 42
        #aperture: default # default, optimal, rinf, rsup
        shape: fit # fit or fix, default fit
        GP: False
        nroll: 3
        unwrap: False
        optimizer: emcee
        star:
          star_name: WASP-47
          dace: False
          Rstar: [1.13, 0.03] # Dai et al., 2019
          Mstar: [1.01, 0.05] # Dai et al., 2019
          teff: [5552, 75] # None # Dai et al., 2019
          logg: [4.34, 0.03] # None # Dai et al., 2019
          feh : [0.38, 0.05] # None # Dai et al., 2019
          h_1: 
            fit: False
            value: [0.714, 0.011]
            bounds: [0.0, 1.0]

          h_2: 
            fit: False
            value: [0.438, 0.050]
            bounds: [0.0, 1.0]
        planet:
          T_ref: [2459124.94060303, 0.0005711284] # value +/- error
          P_ref: [4.16071, 0.00038] # period in days # Almenara et al., 2016
          Kms: [142.0, 1.7 ] # K_RV in m/s # Almenara et al., 2016
        # emcee parameters: NOT USED IF optimizer is ultranest
        emcee:
          nwalkers: 256
          nprerun : 2
          nsteps  : 7
          nburn   : 0
          nthin   : 1
          nthreads: 2
          progress: True
        # ultranest parameters: NOT USED IF optimizer is emcee
        ultranest:
          live_points: 120
          cluster_num_live_points: 40
          tol: 2.0
          adaptive_nsteps: False   #  False, 'proposal-distance', 'move-distance'
          resume: overwrite ## resume should be one of 'overwrite' 'subfolder', 'resume' or 'resume-similar'

Currently, it runs only with MCMC, the nested sampling version with ``ultranest`` is still under development, so it will currently use ``emcee`` as default optimizer.

After modified the above file, you can use the ``cheope``'s multivisit class by digiting:

.. code-block::

        cheope -i path/to/parameters/file.yml -m

TESS & Kepler/K2
^^^^^^^^^^^^^^^^
``cheope`` can analyse also datasets different from the ones of the CHEOPS space mission. In this example we show how to set up the parameters file to analyse the light curves from the TESS data sets.

It can also automatically search, download and analyse different sectors's lightcurves of TESS using the ``selenium`` functionalities (see below for its usage).

.. code-block::
        
        main_folder: /absolute/path/to/analysis/folder
        firefox_driver_path: /path/to/firefox/geckodriver
        download_path: /your/download/folder
        file_fits: /path/to/TESS/lightcurve_lc.fits
        object_name: 102264230 # TIC number
        passband: TESS
        aperture: pdc # sap or pdc
        seed: 42
        single_duration_hour: 13.7267
        optimizer: emcee
        dace: True
        shape: fix
        star:
          star_name: WASP-47
          Rstar: [1.13, 0.03] # Dai et al., 2019
          Mstar: [1.01, 0.05] # Dai et al., 2019
          teff: [5552, 75] # None # Dai et al., 2019
          logg: [4.34, 0.03] # None # Dai et al., 2019
          feh : [0.38, 0.05] # None # Dai et al., 2019
        planet:
          P: 
            value: [4.1591289, 0.0000042] # period in days
            fit: False
          k: [0.10193, 0.00021]
          b: [0.173, 0.032]
          W:
            value: [3.5722, 0.003]
            fit: False
          # D: [0.003430, 0.000286] # flux depth
          #k: None # = Rp/Rstar, None, omitted or as D, D stronger than k
          # or Rp in Rearth
          # k: [0.1019, 0.0002] # Almenara et al., 2016
          # provide:
          # {inc, aRs, b}
          # or
          # {inc, aRs} => b
          # or
          # {b, aRs} ==> inc
          # or
          # {b, inc} ==> aRs
          #b: None
          # inc: 
          #   fit: True
          #   value: [89.49, 0.64] # degrees
          #   bounds: [70, 90]
          inc: [88.98, 0.2]
          aRs: [9.702, 0.044]
          # if total duration T14 in days provided use it for W = T14/P,
          # otherwise computed from
          # {k, b, aRs}
          #T14: None
          T_ref: [2459470.147334, 0.0011402]
        # emcee parameters: NOT USED IF optimizer is ultranest
        emcee:
          nwalkers: 256
          nprerun : 2
          nsteps  : 7
          nburn   : 0
          nthin   : 1
          nthreads: 2
          progress: True
        # ultranest parameters: NOT USED IF optimizer is emcee
        ultranest:
          live_points: 120
          cluster_num_live_points: 40
          tol: 2.0
          adaptive_nsteps: False   #  False, 'proposal-distance', 'move-distance'
          resume: overwrite ## resume should be one of 'overwrite' 'subfolder', 'resume' or 'resume-similar'

To run ``cheope`` simply run:

.. code-block::

        cheope -i path/to/parameters/file.yml -skt

If you installed the ``firefox`` geckodriver from their `GitHub repository <https://github.com/mozilla/geckodriver/releases>`_, you can specify in the parameters file the location of the driver and download the TESS lightcurves using the following command

.. code-block::

        cheope -i path/to/parameters/file.yml --selenium-tess --download


Custom light curve
^^^^^^^^^^^^^^^^^^

``cheope`` allows you to fit for an input lightcurve from an ASCII file (e.g. extension .dat, .txt etc.)
The input lightcurve should have at least three columns with: time, flux and the error on the flux

An example parameters file would be:

.. code-block::

		main_folder: /absolute/path/to/analysis/folder
		file_ascii: /absolute/path/to/custom_lightcurve.dat
		file_columns: [time, flux, flux_err] # allowed: [time, flux, flux_err, bg, contam, smear, centroid_x, centroid_y, xoff, yoff]
		normalise_flux: False
		input_LD:
		  type: quad
		  coeff: [0.714, 0.438]
		seed: 42
		dace: True
		optimizer: emcee
		visit_number: 1
		#to_detrend: ['all']
		star:
		  star_name: WASP-47
		  Rstar: [1.13, 0.03] # Dai et al., 2019
		  Mstar: [1.01, 0.05] # Dai et al., 2019
		  teff: [5552, 75] # None # Dai et al., 2019
		  logg: [4.34, 0.03] # None # Dai et al., 2019
		  feh : [0.38, 0.05] # None # Dai et al., 2019
		  h_1: 
		    fit: False
		    value: [0.714, 0.011]
		    bounds: [0.0, 1.0]

		  h_2: 
		    fit: False
		    value: [0.438, 0.050]
		    bounds: [0.0, 1.0]
		planet:
		  P: 
		    value: [9.0307784240, 0.00015] # period in days
		    fit: False
		  aRs: [16.268,  0.074]
		  b: [0.192, 0.065]
		  Rp: [3.58, 0.04] # Planetary Radiu in Rearth
		  #D: [0.00642, 0.00018] # flux depth
		  #k: None # = Rp/Rstar, None, omitted or as D, D stronger than k
		  # or Rp in Rearth
		  # k: [0.1019, 0.0002] # Almenara et al., 2016
		  # provide:
		  # {inc, aRs, b}
		  # or
		  # {inc, aRs} => b
		  # or
		  # {b, aRs} ==> inc
		  # or
		  # {b, inc} ==> aRs
		  #b: None
		  inc: [89.32, 0.23] # degrees  # Almenara et al., 2016
		  # aRs: [9.705, 0.047] # Almenara et al., 2016
		  # if total duration T14 in days provided use it for W = T14/P, 
		  # otherwise computed from
		  # {k, b, aRs}
		  #T14: None
		  T_ref: [2173.3700844621, 0.0003633881]
		# emcee parameters:
		emcee:
		  nwalkers: 256
		  nprerun : 512
		  nsteps  : 1024
		  nburn   : 0
		  nthin   : 1
		  progress: True
		ultranest:
		  tol: 0.5

To fit the lightcurve launch the following command:

.. code-block::

		cheope -i path/to/parameters/file.yml -a