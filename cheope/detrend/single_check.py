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

import pyconstants as cst
import pycheops_analysis as pyca

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

    def check_yaml_keyword(self, keyword, yaml_input, olog=None):

        l = ""
        if keyword not in yaml_input:
            l = "ERROR: needed keyword {} not in input file".format(keyword)
        return l

    def read_file(self, yaml_file_in):

        read_file_status = []

        visit_args = {}
        star_args = {}
        # planet_args = {}
        # emcee_args  = {}

        if os.path.exists(yaml_file_in) and os.path.isfile(yaml_file_in):
            with open(yaml_file_in) as in_f:
                yaml_input = yaml.load(in_f, Loader=yaml.FullLoader)

            # -- visit_args
            key = "main_folder"
            ck = self.check_yaml_keyword(key, yaml_input)
            if "ERROR" in ck:
                visit_args[key] = None
                sys.exit(ck)
            else:
                visit_args[key] = os.path.abspath(yaml_input[key])
            read_file_status.append(ck)

            key = "visit_number"
            ck = self.check_yaml_keyword(key, yaml_input)
            if "ERROR" in ck:
                visit_args[key] = None
            else:
                visit_args[key] = yaml_input[key]
            read_file_status.append(ck)

            key = "file_key"
            ck = self.check_yaml_keyword(key, yaml_input)
            if "ERROR" in ck:
                visit_args[key] = None
            else:
                visit_args[key] = yaml_input[key].strip()
            read_file_status.append(ck)

            key = "aperture"
            visit_args[key] = "DEFAULT"
            if key in yaml_input:
                tmp = yaml_input[key].strip().upper()
                if tmp in ["DEFAULT", "OPTIMAL", "RINF", "RSUP"]:
                    visit_args[key] = tmp
                else:
                    read_file_status.append("{} set to default: DEFAULT".format(key))

            key = "seed"
            visit_args[key] = 42
            if key in yaml_input:
                tmp = yaml_input[key]
                try:
                    visit_args[key] = int(tmp)
                except:
                    read_file_status.append("{} must be a positive integer".format(key))

            # -- star_args
            star_yaml = yaml_input["star"]

            key = "star_name"
            ck = self.check_yaml_keyword(key, star_yaml)
            if "ERROR" in ck:
                star_args[key] = None
            else:
                star_args[key] = star_yaml[key]
            read_file_status.append(ck)

            key = "Rstar"
            ck = self.check_yaml_keyword(key, star_yaml)
            if "ERROR" in ck:
                star_args[key] = None
            else:
                star_args[key] = ufloat(star_yaml[key][0], star_yaml[key][1])
            read_file_status.append(ck)

            key = "Mstar"
            ck = self.check_yaml_keyword(key, star_yaml)
            if "ERROR" in ck:
                star_args[key] = None
            else:
                star_args[key] = ufloat(star_yaml[key][0], star_yaml[key][1])
            read_file_status.append(ck)

            key = "teff"
            star_args[key] = None
            if key in star_yaml:
                star_args[key] = ufloat(star_yaml[key][0], star_yaml[key][1])

            key = "logg"
            star_args[key] = None
            if key in star_yaml:
                star_args[key] = ufloat(star_yaml[key][0], star_yaml[key][1])

            key = "feh"
            star_args[key] = None
            if key in star_yaml:
                star_args[key] = ufloat(star_yaml[key][0], star_yaml[key][1])

        else:
            read_file_status.append("NOT VALID INPUT FILE:\n{}".format(yaml_file_in))
            # sys.exit()

        return visit_args, star_args, read_file_status

    # ======================================================================
    # ======================================================================

    def run(self):

        start_time = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())

        # ======================================================================
        # CONFIGURATION
        # ======================================================================

        visit_args, star_args, read_file_status = self.read_file(self.input_file)

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
            star_name, match_arcsec=None, teff=teff, logg=logg, metal=feh, dace=False
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


# ======================================================================
# ======================================================================
if __name__ == "__main__":
    sc = SingleCheck()
    sc.run()
