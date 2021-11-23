#!/usr/bin/env python
# coding: utf-8

# WG-P3 EXPLORE/TTV

# # TEST BASED ON
# KELT-6 b VISIT 1


import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt

# import pycheops
from pycheops import Dataset, StarProperties
from pycheops.dataset import _kw_to_Parameter, _log_prior
from pycheops.funcs import massradius, rhostar
from lmfit import Parameter, Parameters
from uncertainties import ufloat, UFloat

# import uncertainties.umath as um
from uncertainties import umath as um
import corner

import argparse
import numpy as np
import os
import sys
import time
from pathlib import Path

import yaml

import cheope.pyconstants as cst
import cheope.pycheops_analysis as pyca
from cheope.parameters import ReadFile

from pycheops.instrument import CHEOPS_ORBIT_MINUTES

# matplotlib rc params
my_dpi = 192
plt.rcParams["text.usetex"] = False
# plt.rcParams['font.family']       = 'sans-serif'
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Computer Modern Roman", "Palatino", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["figure.figsize"] = [5, 5]
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"
plt.rcParams["figure.dpi"] = my_dpi
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 12
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["animation.html"] = "jshtml"

print_pycheops_pars = pyca.print_pycheops_pars
print_analysis_stats = pyca.print_analysis_stats
multi_detrending_model = pyca.multi_detrending_model
printlog = pyca.printlog

fig_ext = ["png", "pdf"]  # ", eps"]


class SingleCheck:
    def __init__(self, input_file):
        self.input_file = input_file

    # ======================================================================
    # ======================================================================

    def run(self):

        start_time = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())

        # ======================================================================
        # CONFIGURATION
        # ======================================================================

        inpars = ReadFile(self.input_file)

        (visit_args, star_args, planet_args, emcee_args, read_file_status,) = (
            inpars.visit_args,
            inpars.star_args,
            inpars.planet_args,
            inpars.emcee_args,
            inpars.read_file_status,
        )

        # seed = 42
        seed = visit_args["seed"]
        np.random.seed(seed)

        # =====================
        # aperture type to use:
        # choose from:
        # “OPTIMAL”, “RSUP”, “RINF”, or “DEFAULT”
        aperture = visit_args["aperture"]

        # =====================
        # name of the file in pycheops_data
        file_key = visit_args["file_key"]

        # =====================
        # visit_folder
        main_folder = visit_args["main_folder"]
        visit_number = visit_args["visit_number"]
        # shape        = visit_args['shape']

        # visit_folder = Path('/home/borsato/Dropbox/Research/exoplanets/objects/KELT/KELT-6/data/CHEOPS_DATA/pycheops_analysis/visit_01/')
        visit_name = "visit_{:02d}_{:s}".format(visit_number, file_key)

        logs_folder = os.path.join(main_folder, "logs")
        if not os.path.isdir(logs_folder):
            os.makedirs(logs_folder, exist_ok=True)
        log_file = os.path.join(logs_folder, "{}_{}.log".format(start_time, visit_name))
        olog = open(log_file, "w")

        printlog("", olog=olog)
        printlog(
            "################################################################",
            olog=olog,
        )
        printlog(" CHECK OF A SINGLE VISIT OF CHEOPS OBSERVATION", olog=olog)
        printlog(
            "################################################################",
            olog=olog,
        )

        # =====================
        # TARGET STAR
        # =====================
        # target name, without planet or something else. It has to be a name in simbad
        star_name = star_args["star_name"]

        printlog("TARGET: {}".format(star_name), olog=olog)
        printlog("FILE_KEY: {}".format(file_key), olog=olog)
        printlog("APERTURE: {}".format(aperture), olog=olog)

        # CHECK INPUT FILE
        error_read = False
        for l in read_file_status:
            if len(l) > 0:
                printlog(l, olog=olog)
            if "ERROR" in l:
                error_read = True
        if error_read:
            olog.close()
            sys.exit()

        visit_folder = Path(os.path.join(main_folder, visit_name))
        if not visit_folder.is_dir():
            visit_folder.mkdir(parents=True, exist_ok=True)

        printlog("SAVING OUTPUT INTO FOLDER {}".format(visit_folder), olog=olog)

        # stellar parameters
        # from WG TS3
        Rstar = star_args["Rstar"]
        Mstar = star_args["Mstar"]
        teff = star_args["teff"]
        logg = star_args["logg"]
        feh = star_args["feh"]

        star = StarProperties(
            star_name,
            match_arcsec=5,
            teff=teff,
            logg=logg,
            metal=feh,
            dace=visit_args["dace"],
        )

        printlog("STAR INFORMATION", olog=olog)
        printlog(star, olog=olog)
        if star.logrho is None:
            printlog("logrho not available from sweetcat...computed:", olog=olog)
            rho_star = Mstar / (Rstar ** 3)
            logrho = um.log10(rho_star)
            star.logrho = logrho
            printlog("logrho = {}".format(logrho), olog=olog)

        printlog("rho  = {} rho_sun".format(10 ** star.logrho), olog=olog)

        # ======================================================================
        # Load data
        printlog("Loading dataset", olog=olog)

        dataset = Dataset(
            file_key=file_key,
            target=star_name,
            download_all=True,
            view_report_on_download=False,
        )

        # copy DRP pdf report in output folder
        tgzfile = os.path.abspath(dataset.tgzfile)
        pdfFile = "{}_DataReduction.pdf".format(file_key)
        printlog("", olog=olog)
        printlog(tgzfile, olog=olog)
        printlog(Path(dataset.tgzfile).parent, olog=olog)
        printlog(Path(dataset.tgzfile).parent / pdfFile, olog=olog)
        pyca.copy_DRP_report(dataset, visit_folder, olog=olog)
        printlog("", olog=olog)

        # extract all the apertures and plot them all
        printlog("Plotting visit for all the apertures", olog=olog)
        pyca.plot_all_apertures(dataset, visit_folder)

        # Get original light-curve
        printlog("Getting light curve for selected aperture", olog=olog)
        t, f, ef = dataset.get_lightcurve(
            aperture=aperture,
            reject_highpoints=False,
            decontaminate=False,
        )

        printlog(
            "Selected aperture = {} px ({})".format(dataset.ap_rad, aperture), olog=olog
        )

        printlog("Plot raw light curve", olog=olog)
        fig = pyca.plot_single_lc(t, f, ef, dataset.lc["bjd_ref"])
        for ext in fig_ext:
            fig.savefig(
                visit_folder.joinpath("00_lc_raw.{:s}".format(ext)), bbox_inches="tight"
            )
        plt.close(fig)

        # Clip outliers
        printlog("Clip outliers", olog=olog)
        t, f, ef = dataset.clip_outliers(verbose=True)

        printlog("Plot clipped light curve", olog=olog)
        fig = pyca.plot_single_lc(t, f, ef, dataset.lc["bjd_ref"])
        for ext in fig_ext:
            fig.savefig(
                visit_folder.joinpath("01_lc_clipped_outliers.{:s}".format(ext)),
                bbox_inches="tight",
            )
        plt.close(fig)

        printlog("Diagnostic plots", olog=olog)

        # Diagnostic plot
        try:
            for ext in fig_ext:
                dataset.diagnostic_plot(
                    fname=visit_folder.joinpath("02_diagnostic_plot.{:s}".format(ext))
                )
        except:
            printlog(
                "Still issues with the dataset.diagnostic_plot function eh...",
                olog=olog,
            )

        # Roll angle vs contam | Roll angle vs background | background vs contam
        lc = dataset.lc
        fig = pyca.plot_custom_diagnostics(lc)
        for ext in fig_ext:
            fig.savefig(
                visit_folder.joinpath("03_roll_angle_vs_bg_vs_contam.{:s}".format(ext)),
                bbox_inches="tight",
            )
        plt.close(fig)

        # Corner plot of diagnostics
        fig = pyca.plot_corner_diagnostics(dataset)
        for ext in fig_ext:
            fig.savefig(
                visit_folder.joinpath("03_diagnostics_corner.{:s}".format(ext)),
                bbox_inches="tight",
            )
        plt.close(fig)
        # ======================================================================

        printlog(
            "Checking glint - Remember to check this table and also the residuals vs roll angle plot.",
            olog=olog,
        )
        # planet check for glint
        # dataset.planet_check()
        pyca.planet_check(dataset, olog=olog)

        printlog("\n===\nEND\n===\n", olog=olog)

        olog.close()


class CheckEphemerids:
    def __init__(self, input_file):
        self.input_file = input_file

        inpars = ReadFile(self.input_file)

        # ======================================================================
        # CONFIGURATION
        # ======================================================================

        (
            self.visit_args,
            self.star_args,
            self.planet_args,
            self.emcee_args,
            self.read_file_status,
        ) = (
            inpars.visit_args,
            inpars.star_args,
            inpars.planet_args,
            inpars.emcee_args,
            inpars.read_file_status,
        )

    # ======================================================================
    # ======================================================================

    def plot_lightcurve(self):

        mpl.use("TkAgg")

        dataset = Dataset(
            file_key=self.visit_args["file_key"],
            target=self.star_args["star_name"],
            download_all=True,
            view_report_on_download=False,
            n_threads=self.emcee_args["nthreads"],
        )

        dataset.get_lightcurve("DEFAULT", decontaminate=False)

        T_ref = self.planet_args["T_ref"]
        P = self.planet_args["P_user_data"]
        Wd = self.planet_args["W"] * P

        btjd = dataset.lc["bjd_ref"]

        t = dataset.lc["time"] + btjd
        emin = np.rint((np.min(t) - T_ref.n) / P.n)
        x = T_ref.n + P.n * emin
        if x < np.min(t):
            emin -= 1
        emax = np.rint((np.max(t) - T_ref.n) / P.n)
        x = T_ref.n + P.n * emax
        if x > np.max(t):
            emax += 1
        epochs = np.arange(emin, emax + 1, 1).astype(np.int32)

        lc = dataset.lc

        vdurh = dataset.lc.get("vdurh")

        if vdurh is None:
            vdurh = 1.5 * Wd.n * cst.day2hour + 3.0 * CHEOPS_ORBIT_MINUTES * cst.min2day
        vdur = vdurh / cst.day2hour
        hdur = 0.5 * vdur
        vdur_co = vdur * cst.day2min / CHEOPS_ORBIT_MINUTES

        # printlog("t min: {:.5f}".format(np.min(t)))
        # printlog("t max: {:.5f}".format(np.max(t)))

        transits = []
        t0s = []

        median_flux = np.median(dataset.lc["flux"])
        std_flux = np.std(dataset.lc["flux"])

        dataset.lc["flux"][
            np.where(np.abs(dataset.lc["flux"] - median_flux) > 2 * std_flux)
        ] = median_flux

        # for i_epo, epo in enumerate(epochs):

        #     bjd_lin = T_ref.n + P.n * epo

        #     sel = np.logical_and(t >= bjd_lin - hdur, t < bjd_lin + hdur)
        #     nsel = np.sum(sel)
        #     wsel = np.logical_and(
        #         t >= bjd_lin - (Wd.n * 0.5), t < bjd_lin + (Wd.n * 0.5)
        #     )
        #     nwsel = np.sum(wsel)
        #     if nsel > 0 and nwsel > 3:
        #         tra = {}
        #         tra["epoch"] = epo
        #         tra["data"] = {}
        #         for k, v in lc.items():
        #             tra["data"][k] = v[sel]
        #         bjdref = int(np.min(tra["data"]["time"]) + btjd)
        #         tra["bjdref"] = bjdref
        #         Tlin = bjd_lin - bjdref
        #         tra["T_0"] = Tlin
        #         tra["T_0_bounds"] = [Tlin - 0.5 * Wd.n, Tlin + 0.5 * Wd.n]
        #         tra["T_0_user_data"] = ufloat(Tlin, 0.5 * Wd.n)
        #         transits.append(tra)
        #         t0s.append(tra["bjdref"])
        #     else:
        #         # pass

        t0s = [T_ref.n + P.n * epo for epo in range(min(epochs) - 1, max(epochs) + 1)]

        t0s_maxs = [
            T_ref.n + P.n * epo + np.sqrt(T_ref.s ** 2 + P.s ** 2)
            for epo in range(min(epochs) - 1, max(epochs) + 1)
        ]

        t0s_mins = [
            T_ref.n + P.n * epo - np.sqrt(T_ref.s ** 2 + P.s ** 2)
            for epo in range(min(epochs) - 1, max(epochs) + 1)
        ]

        t0s = np.array(t0s) - btjd
        t0s_maxs = np.array(t0s_maxs) - btjd
        t0s_mins = np.array(t0s_mins) - btjd

        print(f"Aperture is: {self.visit_args['aperture']}")
        print(t0s)

        if self.visit_args["aperture"].lower() == "sap":
            flux_lab = "SAP_FLUX"
        elif self.visit_args["aperture"].lower() == "pdc":
            flux_lab = "PDCSAP_FLUX"

        markers, caps, bars = plt.errorbar(
            dataset.lc["time"],
            dataset.lc["flux"],
            yerr=dataset.lc["flux_err"],
            fmt="o",
            markersize=0.3,
            capsize=0.2,
            color="k",
            ecolor="gray",
            elinewidth=0.2,
        )

        plt.vlines(
            t0s,
            ymin=0,
            ymax=10 * max(dataset.lc["flux"]),
            color="firebrick",
            linewidth=0.5,
            linestyles="dashed",
        )

        for i in range(len(t0s_maxs)):
            plt.axvspan(t0s_mins[i], t0s_maxs[i], alpha=0.5, color="gray")

        for cap in caps:
            cap.set_markeredgewidth(0.3)

        [bar.set_alpha(0.5) for bar in bars]
        [cap.set_alpha(0.5) for cap in caps]
        plt.ylim(
            min(dataset.lc["flux"]) - max(dataset.lc["flux_err"]),
            max(dataset.lc["flux"]) + max(dataset.lc["flux_err"]),
        )
        plt.xlabel(f"Time (BTJD - {btjd})")
        plt.ylabel("Flux")
        plt.show()


# ======================================================================
# ======================================================================
if __name__ == "__main__":
    sc = SingleCheck()
    sc.run()
