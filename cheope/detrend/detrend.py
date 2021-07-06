#!/usr/bin/env python
# coding: utf-8

# WG-P3 EXPLORE/TTV


import emcee
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

# matplotlib rc params
my_dpi = 192
plt.rcParams["text.usetex"] = False
# plt.rcParams['font.family']                         = 'sans-serif'
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
printlog = pyca.printlog

fig_ext = ["png", "pdf"]  # ", eps"]


class SingleBayes:
    def __init__(self, input_file):
        self.input_file = input_file

    def check_yaml_keyword(self, keyword, yaml_input, olog=None):

        l = ""
        if keyword not in yaml_input:
            l = "ERROR: needed keyword {} not in input file".format(keyword)

        return l

    # ======================================================================
    def read_file(self):

        yaml_file_in = self.input_file

        read_file_status = []

        visit_args = {}
        star_args = {}
        planet_args = {}
        emcee_args = {}

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

            key = "shape"
            visit_args[key] = "fit"
            if key in yaml_input:
                tmp = yaml_input[key].strip().lower()
                if tmp in ["fit", "fix"]:
                    visit_args[key] = tmp
                else:
                    read_file_status.append("{} set to default: fit".format(key))

            key = "seed"
            visit_args[key] = 42
            if key in yaml_input:
                tmp = yaml_input[key]
                try:
                    visit_args[key] = int(tmp)
                except:
                    read_file_status.append("{} must be a positive integer".format(key))

            key = "glint_type"
            visit_args[key] = False
            if key in yaml_input:
                tmp = yaml_input[key]
                if isinstance(tmp, str) and tmp.lower() in ["moon", "glint"]:
                    visit_args[key] = tmp.lower()

            key = "clipping"
            visit_args[key] = False
            if key in yaml_input:
                tmp = yaml_input[key]
                if isinstance(tmp, list):
                    # visit_args[key] = {tmp[0]: tmp[1]}
                    visit_args[key] = tmp

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

            # -- planet_args
            planet_yaml = yaml_input["planet"]

            key = "P"
            ck = self.check_yaml_keyword(key, planet_yaml)
            if "ERROR" in ck:
                planet_args[key] = None
            else:
                planet_args[key] = ufloat(planet_yaml[key][0], planet_yaml[key][1])
            read_file_status.append(ck)

            if "D" in planet_yaml:
                D = ufloat(planet_yaml["D"][0], planet_yaml["D"][1])
                k = um.sqrt(D)
            elif "k" in planet_yaml:
                k = ufloat(planet_yaml["k"][0], planet_yaml["k"][1])
                D = k ** 2
            elif "Rp" in planet_yaml:
                Rp = ufloat(planet_yaml["Rp"][0], planet_yaml["Rp"][1]) * cst.Rears
                k = Rp / star_args["Rstar"]
                D = k ** 2
            else:
                read_file_status.append(
                    "ERROR: missing needed planet keyword: D or k or Rp (Rearth)"
                )
                # sys.exit()
            planet_args["D"] = D
            planet_args["k"] = k

            if "inc" in planet_yaml and "aRs" in planet_yaml and "b" in planet_yaml:
                inc = ufloat(planet_yaml["inc"][0], planet_yaml["inc"][1])
                aRs = ufloat(planet_yaml["aRs"][0], planet_yaml["aRs"][1])
                b = ufloat(planet_yaml["b"][0], planet_yaml["b"][1])
            elif "inc" in planet_yaml and "aRs" in planet_yaml:
                inc = ufloat(planet_yaml["inc"][0], planet_yaml["inc"][1])
                aRs = ufloat(planet_yaml["aRs"][0], planet_yaml["aRs"][1])
                b = aRs * um.cos(inc * cst.deg2rad)
            elif "b" in planet_yaml and "aRs" in planet_yaml:
                aRs = ufloat(planet_yaml["aRs"][0], planet_yaml["aRs"][1])
                b = ufloat(planet_yaml["b"][0], planet_yaml["b"][1])
                inc = um.acos(b / aRs) * cst.rad2deg
            elif "b" in planet_yaml and "inc" in planet_yaml:
                b = ufloat(planet_yaml["b"][0], planet_yaml["b"][1])
                inc = ufloat(planet_yaml["inc"][0], planet_yaml["inc"][1])
                aRs = b / um.cos(inc * cst.deg2rad)
            else:
                read_file_status.append(
                    "ERROR: missing needed one of these pairs/combinations: (inc, aRs) or (b, aRs) or (b, inc) or (inc, aRs, b)"
                )
                inc, aRs, b = 90.0, 1.0, 0.0
                # sys.exit()
            planet_args["inc"] = inc
            planet_args["aRs"] = aRs
            planet_args["b"] = b

            if "T14" in planet_yaml:
                W = (
                    ufloat(planet_yaml["T14"][0], planet_yaml["T14"][1])
                    / planet_args["P"]
                )
            else:
                W = um.sqrt((1 + k) ** 2 - b ** 2) / np.pi / aRs
            planet_args["W"] = W

            ecc = ufloat(0.0, 0.0)
            if "ecc" in planet_yaml:
                # print('planet_yaml["ecc"]', planet_yaml['ecc'])
                if str(planet_yaml["ecc"]).lower() != "none":
                    try:
                        ecc = ufloat(planet_yaml["ecc"][0], planet_yaml["ecc"][1])
                    except:
                        read_file_status.append("wrong ecc format: setting to 0+/-0")
                        ecc = ufloat(0.0, 0.0)
            se = um.sqrt(ecc)
            w = ufloat(90.0, 0.0)
            if "w" in planet_yaml:
                # print('planet_yaml["w"]', planet_yaml['w'])
                if str(planet_yaml["w"]).lower() != "none":
                    try:
                        w = ufloat(planet_yaml["w"][0], planet_yaml["w"][1])
                    except:
                        read_file_status.append("wrong w format: setting to 90+/-0 deg")
                        w = ufloat(90.0, 0.0)
            w_r = w * cst.deg2rad
            f_c = se * um.cos(w_r)
            f_s = se * um.sin(w_r)
            planet_args["ecc"] = ecc
            planet_args["w"] = w
            planet_args["f_c"] = f_c
            planet_args["f_s"] = f_s

            key = "T_0"
            ck = self.check_yaml_keyword(key, planet_yaml)
            if "ERROR" in ck:
                planet_args[key] = None
            else:
                planet_args[key] = tuple(planet_yaml[key])
            read_file_status.append(ck)

            key = "Kms"
            ck = self.check_yaml_keyword(key, planet_yaml)
            if "ERROR" in ck:
                planet_args[key] = None
            else:
                planet_args[key] = ufloat(planet_yaml[key][0], planet_yaml[key][1])
            read_file_status.append(ck)

            # -- emcee_args
            emcee_yaml = yaml_input["emcee"]
            emcee_args["nwalkers"] = 128
            emcee_args["nprerun"] = 512
            emcee_args["nsteps"] = 1280
            emcee_args["nburn"] = 256
            emcee_args["nthin"] = 1
            emcee_args["nthreads"] = 1
            emcee_args["progress"] = False
            for key in [
                "nwalkers",
                "nprerun",
                "nsteps",
                "nburn",
                "nthin",
                "progress",
                "nthreads",
            ]:
                if key in emcee_yaml:
                    emcee_args[key] = emcee_yaml[key]

        else:
            read_file_status.append("NOT VALID INPUT FILE:\n{}".format(yaml_file_in))
            # sys.exit()

        return visit_args, star_args, planet_args, emcee_args, read_file_status

    def run_analysis(self):

        yaml_file_in = self.input_file

        start_time = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())

        # ======================================================================
        # CONFIGURATION
        # ======================================================================

        (
            visit_args,
            star_args,
            planet_args,
            emcee_args,
            read_file_status,
        ) = self.read_file()

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
        shape = visit_args["shape"]

        # visit_folder = Path('/home/borsato/Dropbox/Research/exoplanets/objects/KELT/KELT-6/data/CHEOPS_DATA/pycheops_analysis/visit_01/')
        visit_name = "visit_{:02d}_{:s}_{:s}_shape_ap{:s}_BF".format(
            visit_number, file_key, shape.lower(), aperture.upper()
        )

        logs_folder = os.path.join(main_folder, "logs")
        if not os.path.isdir(logs_folder):
            os.makedirs(logs_folder, exist_ok=True)
        log_file = os.path.join(logs_folder, "{}_{}.log".format(start_time, visit_name))
        olog = open(log_file, "w")

        printlog("", olog=olog)
        printlog(
            "###############################################################################",
            olog=olog,
        )
        printlog(
            " ANALYSIS OF A SINGLEVISIT OF CHEOPS OBSERVATION - DET. PARS. FROM BAYES FACTOR",
            olog=olog,
        )
        printlog(
            "###############################################################################",
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

        printlog("rho    = {} rho_sun".format(10 ** star.logrho), olog=olog)

        # ======================================================================
        # Load data
        printlog("Loading dataset", olog=olog)

        dataset = Dataset(
            file_key=file_key,
            target=star_name,
            download_all=True,
            view_report_on_download=False,
            n_threads=emcee_args["nthreads"],
        )

        # Get original light-curve
        printlog("Getting light curve for selected aperture", olog=olog)
        t, f, ef = dataset.get_lightcurve(
            aperture=aperture, reject_highpoints=False, decontaminate=False
        )

        printlog(
            "Selected aperture = {} px ({})".format(dataset.ap_rad, aperture), olog=olog
        )

        # Clip outliers
        printlog("Clip outliers", olog=olog)
        t, f, ef = dataset.clip_outliers(verbose=True)

        clipping = visit_args["clipping"]
        if clipping:
            printlog(
                "Further clipping on {} with {}-sigma wrt the median.".format(
                    clipping[0], clipping[1]
                ),
                olog=olog,
            )
            if clipping[0] in dataset.lc.keys():
                x = dataset.lc[clipping[0]]
                mask = pyca.mask_data_clipping(x, clipping[1], clip_type="median")
                t, f, ef = dataset.mask_data(mask)

        # =====================
        # planetary parameters

        # parameters type for pycheops fit:
        # p = value ==> fixed parameter
        # p = (min, max) ==> boundaries min and max, initial value at mid-point between min and max
        # p = (min, value, max) ==> boundaries min and max, value is the starting point
        # p = ufloat(value, std_err) ==> Gaussian prior
        # if you want to do calculation with ufloat use um.FUNC or normal operators like **
        # if you want to access value from ufloat use: p.n
        # if you want to access std_err from ufloat use: p.s

        P = planet_args["P"]

        D = planet_args["D"]
        k = planet_args["k"]

        inc = planet_args["inc"]
        aRs = planet_args["aRs"]
        b = planet_args["b"]

        W = planet_args["W"]

        ecc = planet_args["ecc"]
        w = planet_args["w"]
        f_c = planet_args["f_c"]
        f_s = planet_args["f_s"]

        T_0 = planet_args["T_0"]  # look at the light curve and change it accordingly

        # # RV SEMI-AMPLITUDE IN m/s NEEDED TO USE MASSRADIUS FUNCTION
        # Kms = planet_args['Kms']

        printlog(
            "\nPARAMETERS OF INPUT - CUSTOM - TO BE MODIFY FOR EACH TARGET & PLANET & VISIT",
            olog=olog,
        )
        printlog("P        = {} d".format(P), olog=olog)
        printlog("k        = {} ==> D = k^2 = {}".format(k, D), olog=olog)
        printlog("i        = {} deg".format(inc), olog=olog)
        printlog("a/Rs = {}".format(aRs), olog=olog)
        printlog("b        = {}".format(b), olog=olog)
        printlog(
            "W        = {}*P = {} d = {} h".format(W, W * P, W * P * 24.0), olog=olog
        )
        printlog("ecc    = {}".format(ecc), olog=olog)
        printlog("w        = {}".format(w), olog=olog)
        printlog("f_c    = {}".format(f_c), olog=olog)
        printlog("f_s    = {}".format(f_s), olog=olog)
        printlog("T_0    = {}\n".format(T_0), olog=olog)

        # determine the out-of-transit lc for initial guess of c based on T_0 min/max
        oot = np.logical_or(t < T_0[0], t > T_0[2])

        # DEFINE HERE HOW TO USE THE PARAMETERS,
        # WE HAVE TO DEFINE IF IT VARY (FIT) OR NOT (FIXED)
        in_par = Parameters()
        in_par["P"] = Parameter(
            "P", value=P.n, vary=False, min=-np.inf, max=np.inf, user_data=None
        )
        in_par["T_0"] = Parameter(
            "T_0", value=T_0[1], vary=True, min=T_0[0], max=T_0[2], user_data=None
        )

        # I will randomize only fitting parameters...
        in_par["D"] = Parameter(
            "D",
            value=np.abs(np.random.normal(loc=D.n, scale=D.s)),
            vary=True,
            min=0.5 * D.n,
            max=1.5 * D.n,
            user_data=D,
        )
        in_par["W"] = Parameter(
            "W",
            value=np.abs(np.random.normal(loc=W.n, scale=W.s)),
            vary=True,
            min=0.5 * W.n,
            max=1.5 * W.n,
            user_data=W,
        )
        in_par["b"] = Parameter(
            "b",
            value=np.abs(np.random.normal(loc=b.n, scale=b.s)),
            vary=True,
            min=0.0,
            max=1.5,
            user_data=b,
        )
        if shape == "fix":
            for n in ["D", "W", "b"]:
                in_par[n].vary = False
            in_par["D"].value = D.n
            in_par["W"].value = W.n
            in_par["b"].value = b.n

        in_par["h_1"] = Parameter(
            "h_1",
            value=star.h_1.n,
            vary=True,
            min=0.0,
            max=1.0,
            user_data=ufloat(star.h_1.n, 0.1),
        )
        in_par["h_2"] = Parameter(
            "h_2",
            value=star.h_2.n,
            vary=True,
            min=0.0,
            max=1.0,
            user_data=ufloat(star.h_2.n, 0.1),
        )

        in_par["f_s"] = Parameter(
            "f_s", value=f_s.n, vary=False, min=-np.inf, max=np.inf, user_data=None
        )
        in_par["f_c"] = Parameter(
            "f_c", value=f_c.n, vary=False, min=-np.inf, max=np.inf, user_data=None
        )

        in_par["logrho"] = Parameter(
            "logrho",
            value=star.logrho.n,
            vary=True,
            min=-9,
            max=6,
            user_data=star.logrho,
        )
        in_par["c"] = Parameter(
            "c", value=np.median(f[oot]), vary=True, min=0.5, max=1.5, user_data=None
        )

        ### *** 1) FIT TRANSIT MODEL ONLY WITH LMFIT
        det_par = {
            "dfdt": None,
            "d2fdt2": None,
            "dfdbg": None,
            "dfdcontam": None,
            "dfdsmear": None,
            "dfdx": None,
            "dfdy": None,
            "d2fdx2": None,
            "d2fdy2": None,
            "dfdsinphi": None,
            "dfdcosphi": None,
            "dfdsin2phi": None,
            "dfdcos2phi": None,
            "dfdsin3phi": None,
            "dfdcos3phi": None,
            "ramp": None,
            "glint_scale": None,
        }

        # LMFIT 0-------------------------------------------------------------
        printlog("\n- LMFIT - ONLY TRANSIT MODEL", olog=olog)
        # Fit with lmfit
        lmfit0 = dataset.lmfit_transit(
            P=in_par["P"],
            T_0=in_par["T_0"],
            f_c=in_par["f_c"],
            f_s=in_par["f_s"],
            D=in_par["D"],
            W=in_par["W"],
            b=in_par["b"],
            h_1=in_par["h_1"],
            h_2=in_par["h_2"],
            logrhoprior=in_par["logrho"],
            c=in_par["c"],
            dfdt=det_par["dfdt"],
            d2fdt2=det_par["d2fdt2"],
            dfdbg=det_par["dfdbg"],
            dfdcontam=det_par["dfdcontam"],
            dfdsmear=det_par["dfdsmear"],
            dfdx=det_par["dfdx"],
            dfdy=det_par["dfdy"],
            d2fdx2=det_par["d2fdx2"],
            d2fdy2=det_par["d2fdy2"],
            dfdsinphi=det_par["dfdsinphi"],
            dfdcosphi=det_par["dfdcosphi"],
            dfdsin2phi=det_par["dfdsin2phi"],
            dfdcos2phi=det_par["dfdcos2phi"],
            dfdsin3phi=det_par["dfdsin3phi"],
            dfdcos3phi=det_par["dfdcos3phi"],
            ramp=det_par["ramp"],
            glint_scale=det_par["glint_scale"],
        )

        lmfit0_rep = dataset.lmfit_report(min_correl=0.5)
        # for l in lmfit0_rep:
        #     printlog(l, olog=olog)
        printlog(lmfit0_rep, olog=olog)
        printlog("", olog=olog)

        # roll angle plot
        fig = dataset.rollangle_plot(figsize=plt.rcParams["figure.figsize"], fontsize=8)
        for ext in fig_ext:
            fig.savefig(
                os.path.join(
                    visit_folder.resolve(),
                    "00_lmfit0_roll_angle_vs_residual.{}".format(ext),
                ),
                bbox_inches="tight",
            )
        plt.close(fig)
        # best-fit plot
        params_lm0 = lmfit0.params.copy()
        fig, _ = pyca.model_plot_fit(
            dataset,
            params_lm0,
            par_type="lm",
            nsamples=0,
            flatchains=None,
            model_filename=os.path.join(visit_folder.resolve(), "00_lc_lmfit0.dat"),
        )
        for ext in fig_ext:
            fig.savefig(
                os.path.join(visit_folder.resolve(), "00_lc_lmfit0.{}".format(ext)),
                bbox_inches="tight",
            )
        plt.close(fig)
        pyca.quick_save_params(
            os.path.join(visit_folder.resolve(), "00_params_lmfit0.dat"),
            params_lm0,
            dataset.lc["bjd_ref"],
        )

        ### *** 2) determine the std of the residuals w.r.t. fitted parameters
        dataset.gp = None  # force gp = None in the dataset
        stats_lm0 = pyca.computes_rms(
            dataset, params_best=params_lm0, glint=False, olog=olog
        )
        sigma_0 = stats_lm0["flux-all (w/o GP)"]["RMS (unbinned)"][0] * 1.0e-6
        dprior = ufloat(0, sigma_0)
        printlog("sigma_0 = {:.6f}".format(sigma_0), olog=olog)

        printlog(
            "Assign prior ufloat(0,sigma_0) to all the detrending parameters", olog=olog
        )
        for k in det_par.keys():
            if k not in ["ramp", "glint_scale"]:
                det_par[k] = dprior
            # printlog("{:20s} = {:.6f}".format(k, det_par[k]), olog=olog)
        printlog(
            "dfdt and d2fdt2 will have a prior of kind N(0, sigma_0/dt),\nwhere dt=max(t)-min(t)",
            olog=olog,
        )
        dt = dataset.lc["time"][-1] - dataset.lc["time"][0]
        for k in ["dfdt", "d2fdt2"]:
            det_par[k] = dprior / dt

        ### *** 3) while loop to determine bayes factor and which parameters remove and keep
        while_cnt = 0
        while True:
            # LMFIT -------------------------------------------------------------
            printlog("\n- LMFIT - iter {}".format(while_cnt), olog=olog)
            # Fit with lmfit
            lmfit_loop = dataset.lmfit_transit(
                P=in_par["P"],
                T_0=in_par["T_0"],
                f_c=in_par["f_c"],
                f_s=in_par["f_s"],
                D=in_par["D"],
                W=in_par["W"],
                b=in_par["b"],
                h_1=in_par["h_1"],
                h_2=in_par["h_2"],
                logrhoprior=in_par["logrho"],
                c=in_par["c"],
                dfdt=det_par["dfdt"],
                d2fdt2=det_par["d2fdt2"],
                dfdbg=det_par["dfdbg"],
                dfdcontam=det_par["dfdcontam"],
                dfdsmear=det_par["dfdsmear"],
                dfdx=det_par["dfdx"],
                dfdy=det_par["dfdy"],
                d2fdx2=det_par["d2fdx2"],
                d2fdy2=det_par["d2fdy2"],
                dfdsinphi=det_par["dfdsinphi"],
                dfdcosphi=det_par["dfdcosphi"],
                dfdsin2phi=det_par["dfdsin2phi"],
                dfdcos2phi=det_par["dfdcos2phi"],
                dfdsin3phi=det_par["dfdsin3phi"],
                dfdcos3phi=det_par["dfdcos3phi"],
                ramp=det_par["ramp"],
                glint_scale=det_par["glint_scale"],
            )
            printlog(dataset.lmfit_report(min_correl=0.5), olog=olog)
            printlog("", olog=olog)

            params_lm_loop = lmfit_loop.params.copy()

            printlog(
                "Bayes Factors ( >~ 1 ==> discard parameter) sorted in descending order",
                olog=olog,
            )
            BF = pyca.computes_bayes_factor(params_lm_loop)
            # printlog("{}".format(BF), olog=olog)
            nBFgtone = 0
            to_rem = {}
            for k, v in BF.items():
                printlog("{:20s} = {:7.3f}".format(k, v), olog=olog)
                if v > 1:
                    if "sin" in k:
                        kc = k.replace("sin", "cos")
                        vc = BF[kc]
                        if vc > 1 and nBFgtone == 0:
                            to_rem[k] = v
                            to_rem[kc] = vc
                            nBFgtone += 1
                    elif "cos" in k:
                        ks = k.replace("cos", "sin")
                        vs = BF[ks]
                        if vs > 1 and nBFgtone == 0:
                            to_rem[k] = v
                            to_rem[ks] = vs
                            nBFgtone += 1
                    else:
                        if nBFgtone == 0:
                            to_rem[k] = v
                        nBFgtone += 1
            # if none detrending parameters has BF > 1 we can stop
            if nBFgtone == 0:
                break
            else:
                # remove detrending parameter with highest BF
                for k, v in to_rem.items():
                    printlog(
                        "Removing parameter: {:20s} with Bayes Factor = {:7.3f}".format(
                            k, v
                        ),
                        olog=olog,
                    )
                    det_par[k] = None

                while_cnt += 1

        printlog(
            "\n-DONE BAYES FACTOR SELECTION IN {} ITERATIONS".format(while_cnt),
            olog=olog,
        )
        printlog("with detrending parameters:", olog=olog)
        det_list = []
        for k, v in det_par.items():
            if v is not None:
                det_list.append(k)
                printlog("{:20s} = {:9.5f}".format(k, v), olog=olog)
        printlog("\n{}".format(", ".join(det_list)), olog=olog)

        # roll angle plot
        fig = dataset.rollangle_plot(figsize=plt.rcParams["figure.figsize"], fontsize=8)
        for ext in fig_ext:
            fig.savefig(
                os.path.join(
                    visit_folder.resolve(),
                    "01_lmfit_loop_roll_angle_vs_residual.{}".format(ext),
                ),
                bbox_inches="tight",
            )
        plt.close(fig)
        # best-fit plot
        fig, _ = pyca.model_plot_fit(
            dataset,
            params_lm_loop,
            par_type="lm",
            nsamples=0,
            flatchains=None,
            model_filename=os.path.join(visit_folder.resolve(), "01_lc_lmfit_loop.dat"),
        )
        for ext in fig_ext:
            fig.savefig(
                os.path.join(visit_folder.resolve(), "01_lc_lmfit_loop.{}".format(ext)),
                bbox_inches="tight",
            )
        plt.close(fig)
        stats_lm = pyca.computes_rms(
            dataset, params_best=params_lm_loop, glint=False, olog=olog
        )
        pyca.quick_save_params(
            os.path.join(visit_folder.resolve(), "01_params_lmfit_loop.dat"),
            params_lm_loop,
            dataset.lc["bjd_ref"],
        )

        ### *** ==============================================================
        ### *** ===== EMCEE ==================================================

        nwalkers = emcee_args["nwalkers"]
        nprerun = emcee_args["nprerun"]
        nsteps = emcee_args["nsteps"]
        nburn = emcee_args["nburn"]
        nthin = emcee_args["nthin"]
        progress = emcee_args["progress"]

        # Run emcee from last best fit
        printlog("\n-Run emcee from last best fit with:", olog=olog)
        printlog(" nwalkers = {}".format(nwalkers), olog=olog)
        printlog(" nprerun    = {}".format(nprerun), olog=olog)
        printlog(" nsteps     = {}".format(nsteps), olog=olog)
        printlog(" nburn        = {}".format(nburn), olog=olog)
        printlog(" nthin        = {}".format(nthin), olog=olog)
        printlog("", olog=olog)

        # EMCEE-------------------------------------------------------------
        result = dataset.emcee_sampler(
            params=params_lm_loop,
            nwalkers=nwalkers,
            burn=nprerun,
            steps=nsteps,
            thin=nthin,
            add_shoterm=False,
            progress=progress,
        )

        printlog(dataset.emcee_report(min_correl=0.5), olog=olog)

        printlog("\n-Plot trace of the chains", olog=olog)
        fig = dataset.trail_plot("all")  # add 'all' for all traces!
        for ext in fig_ext:
            fig.savefig(
                os.path.join(
                    visit_folder.resolve(), "02_trace_emcee_all.{}".format(ext)
                ),
                bbox_inches="tight",
            )
        plt.close(fig)

        printlog("\n-Plot corner full from pycheops (not removed nburn)", olog=olog)
        fig = dataset.corner_plot(plotkeys="all")
        for ext in fig_ext:
            fig.savefig(
                os.path.join(
                    visit_folder.resolve(), "02_corner_emcee_all.{}".format(ext)
                ),
                bbox_inches="tight",
            )
        plt.close(fig)

        printlog(
            "\n-Computing my parameters and plot models with random samples", olog=olog
        )
        params_med, _, params_mle, stats_mle = pyca.get_best_parameters(
            result, dataset, nburn=nburn, dataset_type="visit", update_dataset=True
        )
        # update emcee.params -> median and emcee.params_mle -> mle
        for p in dataset.emcee.params:
            dataset.emcee.params[p] = params_med[p]
            dataset.emcee.params_best[p] = params_mle[p]

        printlog("MEDIAN PARAMETERS", olog=olog)
        for p in params_med:
            printlog(
                "{:20s} = {:20.12f} +/- {:20.12f}".format(
                    p, params_med[p].value, params_med[p].stderr
                ),
                olog=olog,
            )
        pyca.quick_save_params(
            os.path.join(visit_folder.resolve(), "02_params_emcee_median.dat"),
            params_med,
            dataset.lc["bjd_ref"],
        )

        _ = pyca.computes_rms(dataset, params_best=params_med, glint=False, olog=olog)
        fig, _ = pyca.model_plot_fit(
            dataset,
            params_med,
            par_type="median",
            nsamples=nwalkers,
            flatchains=result.chain,
            model_filename=os.path.join(
                visit_folder.resolve(), "02_lc_emcee_median.dat"
            ),
        )
        for ext in fig_ext:
            fig.savefig(
                os.path.join(
                    visit_folder.resolve(), "02_lc_emcee_median.{}".format(ext)
                ),
                bbox_inches="tight",
            )
        plt.close(fig)
        fig = pyca.plot_fft(dataset, params_med, star=star)
        for ext in fig_ext:
            fig.savefig(
                os.path.join(
                    visit_folder.resolve(), "02_fft_emcee_median.{}".format(ext)
                ),
                bbox_inches="tight",
            )
        plt.close(fig)

        printlog("MLE PARAMETERS", olog=olog)
        for p in params_mle:
            printlog(
                "{:20s} = {:20.12f} +/- {:20.12f}".format(
                    p, params_mle[p].value, params_mle[p].stderr
                ),
                olog=olog,
            )
        pyca.quick_save_params(
            os.path.join(visit_folder.resolve(), "02_params_emcee_mle.dat"),
            params_mle,
            dataset.lc["bjd_ref"],
        )

        _ = pyca.computes_rms(dataset, params_best=params_mle, glint=False, olog=olog)
        fig, _ = pyca.model_plot_fit(
            dataset,
            params_mle,
            par_type="mle",
            nsamples=nwalkers,
            flatchains=result.chain,
            model_filename=os.path.join(visit_folder.resolve(), "02_lc_emcee_mle.dat"),
        )
        for ext in fig_ext:
            fig.savefig(
                os.path.join(visit_folder.resolve(), "02_lc_emcee_mle.{}".format(ext)),
                bbox_inches="tight",
            )
        plt.close(fig)
        fig = pyca.plot_fft(dataset, params_mle, star=star)
        for ext in fig_ext:
            fig.savefig(
                os.path.join(visit_folder.resolve(), "02_fft_emcee_mle.{}".format(ext)),
                bbox_inches="tight",
            )
        plt.close(fig)

        file_emcee = pyca.save_dataset(
            dataset, visit_folder.resolve(), star_name, file_key, gp=False
        )
        printlog("-Dumped dataset into file {}".format(file_emcee), olog=olog)

        ### *** ==============================================================
        ### *** ===== TRAIN GP ===============================================
        printlog("", olog=olog)
        printlog("TRAIN GP HYPERPARAMETERS FIXING PARAMETERS", olog=olog)

        params_fixed = pyca.copy_parameters(params_mle)
        # for p in ['T_0','D','W','b']: # only transit shape
        for p in params_mle:  # fixing all transit and detrending parameters
            params_fixed[p].set(vary=False)
        params_fixed["log_sigma"].set(vary=True)

        result_gp_train = dataset.emcee_sampler(
            params=params_fixed,
            nwalkers=nwalkers,
            burn=nprerun,
            steps=nsteps,
            thin=nthin,
            add_shoterm=True,
            progress=progress,
        )

        printlog(dataset.emcee_report(min_correl=0.5), olog=olog)

        printlog("\n-Plot trace of the chains of GP training", olog=olog)
        fig = dataset.trail_plot("all")  # add 'all' for all traces!
        for ext in fig_ext:
            fig.savefig(
                os.path.join(
                    visit_folder.resolve(), "03_trace_emcee_gp_train.{}".format(ext)
                ),
                bbox_inches="tight",
            )
        plt.close(fig)

        printlog(
            "\n-Computing my parameters and plot models with random samples", olog=olog
        )
        (
            params_med_gp_train,
            _,
            params_mle_gp_train,
            stats_mle,
        ) = pyca.get_best_parameters(
            result_gp_train,
            dataset,
            nburn=nburn,
            dataset_type="visit",
            update_dataset=False,
        )

        fig, _ = pyca.model_plot_fit(
            dataset,
            params_med_gp_train,
            par_type="median-GPtrain",
            nsamples=nwalkers,
            flatchains=result.chain,
            model_filename=os.path.join(
                visit_folder.resolve(), "03_lc_emcee_median_gp_train.dat"
            ),
        )
        for ext in fig_ext:
            fig.savefig(
                os.path.join(
                    visit_folder.resolve(), "03_lc_emcee_median_gp_train.{}".format(ext)
                ),
                bbox_inches="tight",
            )
        plt.close(fig)

        fig, _ = pyca.model_plot_fit(
            dataset,
            params_mle_gp_train,
            par_type="mle-GPtrain",
            nsamples=nwalkers,
            flatchains=result.chain,
            model_filename=os.path.join(
                visit_folder.resolve(), "03_lc_emcee_mle_gp_train.dat"
            ),
        )
        for ext in fig_ext:
            fig.savefig(
                os.path.join(
                    visit_folder.resolve(), "03_lc_emcee_mle_gp_train.{}".format(ext)
                ),
                bbox_inches="tight",
            )
        plt.close(fig)

        ### *** =======================================++=====================
        ### *** ===== FIT TRANSIT + DETRENDING + GP =++=======================
        printlog("\nRUN FULL FIT TRANSIT+DETRENDING+GP W/ EMCEE", olog=olog)

        params_fit_gp = pyca.copy_parameters(params_mle)
        for p in ["log_S0", "log_omega0", "log_sigma"]:
            # params_fit_gp[p] = params_mle_gp_train[p]
            # printlog("{} = {} user_data = {}".format(p, params_fit_gp[p], params_fit_gp[p].user_data), olog=olog)
            # params_fit_gp[p].user_data = ufloat(params_mle_gp_train[p].value, 2*params_mle_gp_train[p].stderr)
            # printlog("{} = {} user_data = {}".format(p, params_fit_gp[p], params_fit_gp[p].user_data), olog=olog)
            params_fit_gp.add(
                p,
                value=params_mle_gp_train[p].value,
                vary=True,
                min=params_mle_gp_train[p].min,
                max=params_mle_gp_train[p].max,
            )
            params_fit_gp[p].user_data = ufloat(
                params_mle_gp_train[p].value, 2 * params_mle_gp_train[p].stderr
            )

        # log_Q = 1/sqrt(2)
        params_fit_gp.add("log_Q", value=np.log(1 / np.sqrt(2)), vary=False)

        result_gp = dataset.emcee_sampler(
            params=params_fit_gp,
            nwalkers=nwalkers,
            burn=nprerun,
            steps=nsteps,
            thin=nthin,
            # add_shoterm = True, # not needed the second time
            progress=progress,
        )

        printlog(dataset.emcee_report(min_correl=0.5), olog=olog)

        printlog("\n-Plot trace of the chains", olog=olog)
        fig = dataset.trail_plot("all")  # add 'all' for all traces!
        for ext in fig_ext:
            fig.savefig(
                os.path.join(
                    visit_folder.resolve(), "04_trace_emcee_all.{}".format(ext)
                ),
                bbox_inches="tight",
            )
        plt.close(fig)

        printlog("\n-Plot corner full from pycheops (not removed nburn)", olog=olog)
        fig = dataset.corner_plot(plotkeys="all")
        for ext in fig_ext:
            fig.savefig(
                os.path.join(
                    visit_folder.resolve(), "04_corner_emcee_all.{}".format(ext)
                ),
                bbox_inches="tight",
            )
        plt.close(fig)

        printlog(
            "\n-Computing my parameters and plot models with random samples", olog=olog
        )
        params_med_gp, _, params_mle_gp, _ = pyca.get_best_parameters(
            result_gp, dataset, nburn=nburn, dataset_type="visit", update_dataset=True
        )
        # update emcee.params -> median and emcee.params_mle -> mle
        for p in dataset.emcee.params:
            dataset.emcee.params[p] = params_med_gp[p]
            dataset.emcee.params_best[p] = params_mle_gp[p]

        printlog("MEDIAN PARAMETERS w/ GP", olog=olog)
        for p in params_med_gp:
            printlog(
                "{:20s} = {:20.12f} +/- {:20.12f}".format(
                    p, params_med_gp[p].value, params_med_gp[p].stderr
                ),
                olog=olog,
            )
        pyca.quick_save_params(
            os.path.join(visit_folder.resolve(), "04_params_emcee_median_gp.dat"),
            params_med_gp,
            dataset.lc["bjd_ref"],
        )

        _ = pyca.computes_rms(
            dataset, params_best=params_med_gp, glint=False, olog=olog
        )
        fig, _ = pyca.model_plot_fit(
            dataset,
            params_med_gp,
            par_type="median w/ GP",
            nsamples=nwalkers,
            flatchains=result_gp.chain,
            model_filename=os.path.join(
                visit_folder.resolve(), "04_lc_emcee_median_gp.dat"
            ),
        )
        for ext in fig_ext:
            fig.savefig(
                os.path.join(
                    visit_folder.resolve(), "04_lc_emcee_median_gp.{}".format(ext)
                ),
                bbox_inches="tight",
            )
        plt.close(fig)
        fig = pyca.plot_fft(dataset, params_med_gp, star=star)
        for ext in fig_ext:
            fig.savefig(
                os.path.join(
                    visit_folder.resolve(), "04_fft_emcee_median_gp.{}".format(ext)
                ),
                bbox_inches="tight",
            )
        plt.close(fig)

        printlog("MLE PARAMETERS w/ GP", olog=olog)
        for p in params_mle_gp:
            printlog(
                "{:20s} = {:20.12f} +/- {:20.12f}".format(
                    p, params_mle_gp[p].value, params_mle_gp[p].stderr
                ),
                olog=olog,
            )
        pyca.quick_save_params(
            os.path.join(visit_folder.resolve(), "04_params_emcee_mle_gp.dat"),
            params_mle_gp,
            dataset.lc["bjd_ref"],
        )

        _ = pyca.computes_rms(
            dataset, params_best=params_mle_gp, glint=False, olog=olog
        )
        fig, _ = pyca.model_plot_fit(
            dataset,
            params_mle_gp,
            par_type="mle w/ GP",
            nsamples=nwalkers,
            flatchains=result_gp.chain,
            model_filename=os.path.join(
                visit_folder.resolve(), "04_lc_emcee_mle_gp.dat"
            ),
        )
        for ext in fig_ext:
            fig.savefig(
                os.path.join(
                    visit_folder.resolve(), "04_lc_emcee_mle_gp.{}".format(ext)
                ),
                bbox_inches="tight",
            )
        plt.close(fig)
        fig = pyca.plot_fft(dataset, params_mle_gp, star=star)
        for ext in fig_ext:
            fig.savefig(
                os.path.join(
                    visit_folder.resolve(), "04_fft_emcee_mle_gp.{}".format(ext)
                ),
                bbox_inches="tight",
            )
        plt.close(fig)

        file_emcee = pyca.save_dataset(
            dataset, visit_folder.resolve(), star_name, file_key, gp=True
        )
        printlog("-Dumped dataset into file {}".format(file_emcee), olog=olog)

        printlog("", olog=olog)
        printlog(" *********** ", olog=olog)
        printlog("--COMPLETED--", olog=olog)
        printlog(" *********** ", olog=olog)
        printlog("", olog=olog)

        olog.close()


if __name__ == "__main__":
    sb = SingleBayes()
    sb.run_analysis()
