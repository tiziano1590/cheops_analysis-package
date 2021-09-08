import numpy as np
import matplotlib.pyplot as plt
import pylightcurve as plc
from astropy.io import fits
from cheope.parameters import ReadFile
import time


class TasteLC:
    def __init__(self, input_file):
        self.input_file = input_file

    def fitLC(self):
        inpars = ReadFile(
            self.input_file, multivisit=True
        )  # Multivisits flag prevent the applycation of planetary conditions

        start_time = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())

        # ======================================================================
        # CONFIGURATION
        # ======================================================================

        (visit_args, star_args, planet_args, emcee_args, read_file_status,) = (
            inpars.visit_args,
            inpars.star_args,
            inpars.planet_args,
            inpars.emcee_args,
            inpars.read_file_status,
        )

        main_fold = visit_args["main_folder"] + "/"

        inlc = np.genfromtxt(visit_args["photometry"])
        timex, flux, flux_err = inlc[:, 0], inlc[:, 1], inlc[:, 2]
        plt.errorbar(timex, flux, yerr=flux_err, fmt="o", color="k")
        plt.savefig(main_fold + "test_lightcurve.pdf")

        # define a planet manually
        planet = plc.Planet(
            name=planet_args["name"],
            ra=planet_args["ra"],  # float values are assumed to be in degrees,
            # alternatively, you can provide a plc.Hours or plc.Degrees object
            # here it would be plc.Hours('16:39:14.54')
            dec=planet_args["dec"],  # float values are assumed to be in degrees,
            # alternatively, you can provide a plc.Hours or plc.Degrees object
            # here it would be plc.Degrees('+19:13:33.2')
            stellar_logg=planet_args["stellar_logg"],  # float, in log(cm/s^2)
            stellar_temperature=planet_args["stellar_temperature"],  # float, in Kelvin
            stellar_metallicity=planet_args[
                "stellar_metallicity"
            ],  # float, in dex(Fe/H) or dex(M/H)
            rp_over_rs=planet_args["rp_over_rs"],  # float, no units 1.37*Re
            period=planet_args["period"],  # float, in days
            sma_over_rs=planet_args["sma_over_rs"],  # float, no units
            eccentricity=planet_args["eccentricity"],  # float, no units
            inclination=planet_args[
                "inclination"
            ],  # float values are assumed to be in degrees,
            # alternatively, you can provide a plc.Hours or plc.Degrees object
            # here it would be plc.Degrees(86.71)
            periastron=planet_args[
                "periastron"
            ],  # float values are assumed to be in degrees,
            # alternatively, you can provide a plc.Hours or plc.Degrees object
            # here it would be plc.Degrees(0.0)
            mid_time=planet_args["mid_time"],  # float, in days
            mid_time_format=planet_args[
                "mid_time_format"
            ],  # str, available formats are JD_UTC, MJD_UTC, HJD_UTC, HJD_TDB, BJD_UTC, BJD_TDB
            ldc_method=planet_args[
                "ldc_method"
            ],  # str, default = claret, the other methods are: linear, quad, sqrt
            ldc_stellar_model=planet_args[
                "ldc_stellar_model"
            ],  # str, default = phoenix, the other model is atlas
            albedo=planet_args["albedo"],  # float, default = 0.15, no units
            emissivity=planet_args["emissivity"],  # float, default = 1.0, no units
        )

        planet.add_observation(
            time=timex,
            time_format=planet_args["mid_time_format"],
            exp_time=planet_args["exp_time"],
            time_stamp=planet_args["time_stamp"],
            flux=flux,
            flux_unc=flux_err,
            flux_format=planet_args["flux_format"],
            filter_name=planet_args["filter_name"],
        )

        planet.transit_fitting(
            main_fold + f"MCMC_results_taste_{start_time}",
            iterations=emcee_args["nsteps"],
            walkers=emcee_args["nwalkers"],
            burn_in=emcee_args["nprerun"],
        )
