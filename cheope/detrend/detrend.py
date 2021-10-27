#!/usr/bin/env python
# coding: utf-8

# WG-P3 EXPLORE/TTV


from math import floor
import emcee
import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt

# import pycheops
from pycheops import Dataset, StarProperties
from pycheops.dataset import _kw_to_Parameter, _log_prior
from pycheops.funcs import massradius, rhostar
from pycheops.instrument import CHEOPS_ORBIT_MINUTES
from pycheops.ld import ca_to_h1h2, h1h2_to_ca, h1h2_to_q1q2, q1q2_to_h1h2
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
from astropy.io import fits

import cheope.pyconstants as cst
import cheope.pycheops_analysis as pyca
import cheope.linear_ephemeris as lep
from cheope.parameters import ReadFile
from .optimizers import Optimizers


from cheope.pycheops_analysis import FITSDataset

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
        self.input_pars = [
            "P",
            "T_0",
            "D",
            "W",
            "b",
            "h_1",
            "h_2",
            "f_s",
            "f_c",
            "logrho",
        ]

    def run(self):

        inpars = ReadFile(self.input_file)

        start_time = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())

        # ======================================================================
        # CONFIGURATION
        # ======================================================================

        (
            visit_args,
            star_args,
            planet_args,
            emcee_args,
            ultranest_args,
            read_file_status,
        ) = (
            inpars.visit_args,
            inpars.star_args,
            inpars.planet_args,
            inpars.emcee_args,
            inpars.ultranest_args,
            inpars.read_file_status,
        )

        def category_args(par):
            if par in star_args.keys():
                return star_args
            elif par in planet_args.keys():
                return planet_args
            else:
                self.read_file_status.append(
                    f"ERROR: {par} is not defined in neither the star or planet arguments"
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
        # star_args["logrho"] = star.logrho.n
        # star_args["logrho_fit"] = True
        # star_args["logrho_bounds"] = [-9, 6]
        # star_args["logrho_user_data"] = star.logrho

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
        printlog("a/Rs     = {}".format(aRs), olog=olog)
        printlog("b        = {}".format(b), olog=olog)
        printlog(
            "W        = {}*P = {} d = {} h".format(W, W * P, W * P * 24.0), olog=olog
        )
        printlog("ecc    = {}".format(ecc), olog=olog)
        printlog("w      = {}".format(w), olog=olog)
        printlog("f_c    = {}".format(f_c), olog=olog)
        printlog("f_s    = {}".format(f_s), olog=olog)
        printlog("T_0    = {}\n".format(T_0), olog=olog)

        # determine the out-of-transit lc for initial guess of c based on T_0 min/max
        oot = np.logical_or(
            t < planet_args["T_0_bounds"][0], t > planet_args["T_0_bounds"][1]
        )

        # DEFINE HERE HOW TO USE THE PARAMETERS,
        # WE HAVE TO DEFINE IF IT VARY (FIT) OR NOT (FIXED)

        in_par = Parameters()

        for key in self.input_pars:
            print(key)
            cat = category_args(key)
            if key in ["D", "W", "b"]:
                # Randomize here
                val = np.abs(
                    np.random.normal(
                        loc=cat[key + "_user_data"].n, scale=cat[key + "_user_data"].s
                    )
                )
            else:
                val = cat[key]
            in_par[key] = Parameter(
                key,
                value=val,
                vary=cat[key + "_fit"],
                min=cat[key + "_bounds"][0],
                max=cat[key + "_bounds"][1],
                user_data=cat[key + "_user_data"],
                # TODO Exception has occurred: TypeError
                # loop of ufunc does not support argument 0 of type AffineScalarFunc which has no callable arcsin method
            )
            # print(in_par[key])
            # if key == 'b':
            #     break

        # Treat "c" separately
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
        #
        # set fixed LD parameters
        in_par["h_1"].vary = False
        in_par["h_2"].vary = False
        #
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

        # input params plot
        fig, _ = pyca.model_plot_fit(
            dataset,
            in_par,
            par_type="input",
            nsamples=0,
            flatchains=None,
            model_filename=os.path.join(visit_folder.resolve(), "00_lc_0_input.dat"),
        )
        for ext in fig_ext:
            fig.savefig(
                os.path.join(visit_folder.resolve(), "00_lc_0_input.{}".format(ext)),
                bbox_inches="tight",
            )
        plt.close(fig)
        pyca.quick_save_params(
            os.path.join(visit_folder.resolve(), "00_params_0_input.dat"),
            in_par,
            dataset.lc["bjd_ref"],
        )

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
                # T_0=in_par["T_0"],
                f_c=in_par["f_c"],
                f_s=in_par["f_s"],
                # D=in_par["D"],
                # W=in_par["W"],
                # b=in_par["b"],
                # h_1=in_par["h_1"],
                # h_2=in_par["h_2"],
                T_0=params_lm0["T_0"],
                D=params_lm0["D"],
                W=params_lm0["W"],
                b=params_lm0["b"],
                h_1=params_lm0["h_1"],
                h_2=params_lm0["h_2"],
                logrhoprior=in_par["logrho"],
                c=params_lm0["c"],
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
                    ##TODO if k == 'glint_scale': remove/set to none the variable of pycheops add_glint

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

        # set the LD to proper fit or fix
        keys = ["h_1", "h_2"]
        for key in keys:
            cat = category_args(key)
            params_lm_loop[key] = in_par[key]
            params_lm_loop[key].vary = cat[key + "_fit"]

        if visit_args["optimizer"].lower() == "ultranest":
            ### *** ===== Ultranest ==================================================

            live_points = ultranest_args["live_points"]
            tolerance = ultranest_args["tol"]
            cluster_num_live_points = ultranest_args["cluster_num_live_points"]
            logdir = os.path.join(visit_args["main_folder"], "ultranest")
            resume = ultranest_args["resume"]
            adaptive_nsteps = ultranest_args["adaptive_nsteps"]

            # Run emcee from last best fit
            printlog("\n-Run Ultranest from last best fit with:", olog=olog)
            printlog(" live_points              = {}".format(live_points), olog=olog)
            printlog(" tolerance                = {}".format(tolerance), olog=olog)
            printlog(
                " cluster_num_live_points  = {}".format(cluster_num_live_points),
                olog=olog,
            )
            printlog(
                " logdir                   = {}".format(logdir),
                olog=olog,
            )
            printlog(
                " resume                   = {}".format(resume),
                olog=olog,
            )
            printlog("", olog=olog)
            result = dataset.ultranest_sampler(
                params=params_lm_loop,
                live_points=live_points,
                tol=tolerance,
                cluster_num_live_points=cluster_num_live_points,
                logdir=logdir,
                resume=resume,
                adaptive_nsteps=adaptive_nsteps,
                add_shoterm=False,
            )
        else:
            ### *** ===== EMCEE ==================================================
            optimizer = Optimizers()
            optimizer.emcee(
                inpars=inpars,
                dataset=dataset,
                olog=olog,
                params_lm_loop=params_lm_loop,
                star=star,
            )

        printlog("", olog=olog)
        printlog(" *********** ", olog=olog)
        printlog("--COMPLETED--", olog=olog)
        printlog(" *********** ", olog=olog)
        printlog("", olog=olog)

        olog.close()


class MultivisitAnalysis:
    def __init__(self, input_file):
        self.input_file = input_file

    # ======================================================================
    # ======================================================================
    # ======================================================================
    # ======================================================================
    def run(self):

        start_time = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())

        # ======================================================================
        # CONFIGURATION
        # ======================================================================

        inpars = ReadFile(self.input_file, multivisit=True)

        (visit_args, star_args, planet_args, emcee_args, read_file_status,) = (
            inpars.visit_args,
            inpars.star_args,
            inpars.planet_args,
            inpars.emcee_args,
            inpars.read_file_status,
        )

        def category_args(par):
            if par in star_args.keys():
                return star_args
            elif par in planet_args.keys():
                return planet_args
            else:
                read_file_status.append(
                    f"ERROR: {par} is not defined in neither the star or planet arguments"
                )

        seed = visit_args["seed"]
        np.random.seed(seed)

        main_folder = os.path.abspath(visit_args["main_folder"])
        if not os.path.isdir(main_folder):
            os.makedirs(main_folder, exist_ok=True)

        visit_name = os.path.basename(main_folder)
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
        printlog(" ANALYSIS OF MULTIPLE VISITS OF CHEOPS OBSERVATION", olog=olog)
        printlog(
            "################################################################",
            olog=olog,
        )

        printlog("\nPreparing datasets_list:", olog=olog)
        printlog("from:", olog=olog)
        datasets_list = []
        for k, v in visit_args["datasets"].items():
            printlog("{} {}".format(k, v), olog=olog)
            datasets_list.append(os.path.abspath(v["file_name"]))

        printlog("to:", olog=olog)
        for il, dl in enumerate(datasets_list):
            printlog("{:02d}: {:s}".format(il, dl), olog=olog)

        printlog("\nLoad CustomMultiVisit", olog=olog)
        M = pyca.CustomMultiVisit(
            target=star_args["star_name"],
            datasets_list=datasets_list,
            id_kws={
                "dace": star_args["dace"],
                "teff": star_args["teff"],
                "logg": star_args["logg"],
                "metal": star_args["feh"],
            },
            verbose=True,
        )

        printlog("\nDefine new T_0 and P based on datasets and T_ref, P_ref", olog=olog)
        T_ref = planet_args["T_ref"]
        P_ref = planet_args["P_ref"]

        for k, v in planet_args.items():
            printlog("{} = {}".format(k, v), olog=olog)

        # default in pycheops MultiVisit
        # T_0   = ufloat(M.tzero(T_ref, P_ref), T_ref.s)  # Time of mid-transit closest to middle of datasets
        # LBo: using input ephemeris
        # T_0 = T_ref - cst.btjd
        # LBo: pycheops-like but propagating error on linear ephemeris
        t_mid = np.median([np.median(d.lc["time"]) for d in M.datasets])
        epo = np.rint((t_mid + cst.btjd - T_ref.n) / P_ref.n)
        T_0 = T_ref + epo * P_ref - cst.btjd
        printlog(
            "median times: {}".format([np.median(d.lc["time"]) for d in M.datasets]),
            olog=olog,
        )
        printlog("t_mid = {} => epo = {}".format(t_mid, epo), olog=olog)
        printlog("T_ref = {:.5f} ({:.5f})".format(T_ref, T_ref.n - cst.btjd), olog=olog)
        printlog("P_ref = {:.5f}".format(P_ref), olog=olog)
        printlog("T_0   = {:.5f}".format(T_0), olog=olog)
        printlog("", olog=olog)

        printlog("Updating common transit parameters", olog=olog)
        gnames = pyca.global_names.copy()  # .remove('Tref')

        # new_params = M.datasets[0].emcee.params_best.copy()
        new_params = Parameters()
        # for p in gnames:
        for p in ["D", "W", "b", "h_1", "h_2"]:
            par = []
            # wei = []
            for i, m in enumerate(M.datasets):
                px = m.emcee.params_best[p]
                par.append(px.value)
                if i == 0:
                    pxmin = px.min
                    pxmax = px.max
                    pxuser = px.user_data
            par = np.array(par)
            par_mean = np.mean(par)
            cat = category_args(p)
            new_params[p] = Parameter(
                p,
                value=par_mean,
                vary=cat[p + "_fit"],
                min=pxmin,
                max=pxmax,
                user_data=pxuser,
            )
            printlog(
                "{:15s} ==> {:.6f} bounds = ( {:.6f} , {:.6f}) priors = {:.6f} vary: {}".format(
                    p,
                    new_params[p].value,
                    new_params[p].min,
                    new_params[p].max,
                    new_params[p].user_data,
                    new_params[p].vary,
                ),
                olog=olog,
            )
        for p in ["f_c", "f_s"]:
            m = M.datasets[0]
            px = m.emcee.params_best[p]
            new_params[p] = Parameter(
                p,
                value=px.value,
                vary=False,
                min=px.min,
                max=px.max,
                user_data=px.user_data,
            )

        if visit_args["GP"]:
            log_S0_v, log_S0_e = [], []
            log_omega0_v, log_omega0_e = [], []
            for i, m in enumerate(M.datasets):
                printlog("dataset {}: gp status = {}".format(i + 1, m.gp), olog=olog)
                if m.gp is not None:
                    if m.gp is True:
                        log_S0_v.append(m.emcee.params["log_S0"].value)
                        log_S0_e.append(m.emcee.params["log_S0"].stderr)
                        log_omega0_v.append(m.emcee.params["log_omega0"].value)
                        log_omega0_e.append(m.emcee.params["log_omega0"].stderr)
            if len(log_S0_v) > 0:
                log_S0_mean, swei = np.average(
                    log_S0_v, weights=1.0 / (np.array(log_S0_e) ** 2), returned=True
                )
                log_S0_err = 1.0 / np.sqrt(swei)
                log_S0 = ufloat(log_S0_mean, log_S0_err)
                log_omega0_mean, swei = np.average(
                    log_omega0_v,
                    weights=1.0 / (np.array(log_omega0_e) ** 2),
                    returned=True,
                )
                log_omega0_err = 1.0 / np.sqrt(swei)
                log_omega0 = ufloat(log_omega0_mean, log_omega0_err)

                printlog("log_S0 = {}".format(log_S0), olog=olog)
                printlog("log_omega0 = {}".format(log_omega0), olog=olog)
            else:
                printlog(
                    "I did not find any GP hyperparameters! Set to default", olog=olog
                )
                log_S0 = Parameter("log_S0", value=-12, vary=True, min=-30, max=0)
                log_omega0 = Parameter(
                    "log_omega0", value=3, vary=True, min=-2.3, max=8
                )
                log_S0.vary = True
                log_omega0.vary = True
        else:
            log_S0, log_omega0 = None, None
            printlog("I did not find any GP hyperparameters! Not using GP!", olog=olog)

        # create extra priors for detrending: 0 value for dfdt_XX
        extra_priors = {}
        printlog("checking priors for detrending", olog=olog)
        for i, m in enumerate(M.datasets):
            # print('dataset: {:02d}'.format(i+1))
            for kd in pyca.detrend_default.keys():
                if kd in m.emcee.params_best:
                    pkd = m.emcee.params_best[kd]
                    if kd == "dfdt":
                        extra_priors["dfdt_{:02d}".format(i + 1)] = ufloat(0, 1e-9)
                    else:
                        extra_priors["{:s}_{:02d}".format(kd, i + 1)] = ufloat(
                            pkd.value, pkd.stderr
                        )
        printlog("extra priors:", olog=olog)
        for k, v in extra_priors.items():
            printlog("{:15s} = {:.6f}".format(k, v), olog=olog)

        # START FITTING THE LINEAR EPHEMERIS
        printlog("\nRUNNING EMCEE - FIT LINEAR EPHEM", olog=olog)
        sys.stdout.flush()

        result_lin = M.fit_transit(
            T_0=T_0,
            P=P_ref,
            # TTV yes or not, default: ttv=False
            ttv=False,
            # default for decorrelation:
            unroll=visit_args["unroll"],
            nroll=visit_args["nroll"],
            # if you want to check the effect of roll angle or how it does without
            # unroll=False,
            unwrap=visit_args["unwrap"],  # defaul False
            D=new_params["D"],
            W=new_params["W"],
            b=new_params["b"],
            f_c=new_params["f_c"],
            f_s=new_params["f_s"],
            h_1=new_params["h_1"],
            h_2=new_params["h_2"],
            log_omega0=log_omega0,
            log_S0=log_S0,
            extra_priors=extra_priors,
            burn=emcee_args["nprerun"],
            steps=emcee_args["nsteps"],
            nwalkers=emcee_args["nwalkers"],
            progress=emcee_args["progress"],
            n_threads=emcee_args["nthreads"],
        )
        # WARNING: better have priors on shape and gp hyperparameters,
        # otherwise gp will try to fit also the transit!

        printlog("REPORT", olog=olog)
        printlog("{}".format(M.fit_report(min_correl=0.8)), olog=olog)

        printlog("PARAMETERS MULTIVISIT - LIN", olog=olog)
        par_med, _, par_mle, _ = pyca.get_best_parameters(
            result_lin, M, nburn=0, dataset_type="multivisit"
        )
        # updates params/parbest in result and M.result
        result_lin.params = par_med.copy()
        M.result.params = par_med.copy()
        result_lin.parbest = par_mle.copy()
        M.result.parbest = par_mle.copy()
        pyca.quick_save_params(
            os.path.join(main_folder, "params_med_lin.dat"), par_med, bjd_ref=cst.btjd
        )
        pyca.quick_save_params(
            os.path.join(main_folder, "params_mle_lin.dat"), par_mle, bjd_ref=cst.btjd
        )

        # bin30m_ph = bin30m/result_lin.params['P'].value
        bin30m_ph = False

        printlog("LC no-detrend plot", olog=olog)
        fig = M.plot_fit(
            title="Not detrended",
            data_offset=0.01,
            binwidth=bin30m_ph,
            res_offset=0.005,
            detrend=False,
        )
        for ext in fig_ext:
            plt_file = os.path.join(
                main_folder, "lcs_nodetrend_plot_lin.{}".format(ext)
            )
            fig.savefig(plt_file, bbox_inches="tight")
        plt.close(fig)

        printlog("LC detrended plot", olog=olog)
        fig = M.plot_fit(
            title="Detrended",
            data_offset=0.01,
            binwidth=bin30m_ph,
            res_offset=0.005,
            detrend=True,
        )
        for ext in fig_ext:
            plt_file = os.path.join(main_folder, "lcs_detrend_plot_lin.{}".format(ext))
            fig.savefig(plt_file, bbox_inches="tight")
        plt.close(fig)

        fig, out_lin = pyca.custom_plot_phase(M, result_lin, title="Fit lin. ephem.")
        for ext in fig_ext:
            plt_file = os.path.join(main_folder, "lcs_phased_plot_lin.{}".format(ext))
            fig.savefig(plt_file, bbox_inches="tight")
        plt.close(fig)
        out_file = plt_file.replace(".png", ".dat")
        out = np.column_stack([v for v in out_lin.values()])
        head = "".join(["{:s} ".format(k) for k in out_lin.keys()])
        fmt = "%23.16e " * (len(out_lin) - 1) + "%03.0f"
        np.savetxt(out_file, out, header=head, fmt=fmt)

        printlog("Trace plot", olog=olog)
        fig = M.trail_plot(plotkeys="all", plot_kws={"alpha": 0.1})
        for ext in fig_ext:
            plt_file = os.path.join(main_folder, "trace_plot_all_lin.{}".format(ext))
            fig.savefig(plt_file, bbox_inches="tight")
        plt.close(fig)

        printlog("Corner plot", olog=olog)
        pk = []
        for n in gnames:
            if n in result_lin.params:
                if result_lin.params[n].vary:
                    pk.append(n)
        pk.append("T_0")
        printlog("{}".format(pk), olog=olog)
        fig = M.corner_plot(plotkeys=pk)
        for ext in fig_ext:
            plt_file = os.path.join(main_folder, "corner_plot_all_lin.{}".format(ext))
            fig.savefig(plt_file, bbox_inches="tight")
        plt.close(fig)

        try:
            printlog("MASSRADIUS MULTIVISIT", olog=olog)
            _, fig = M.massradius(
                m_star=star_args["Mstar"],
                r_star=star_args["Rstar"],
                K=planet_args["Kms"],
                jovian=True,
                verbose=True,
            )
            for ext in fig_ext:
                plt_file = os.path.join(main_folder, "massradius_lin.{}".format(ext))
                fig.savefig(plt_file, bbox_inches="tight")
            plt.close(fig)
        except:
            printlog("massradius error: to be investigated", olog=olog)

        sys.stdout.flush()

        params_fit = result_lin.parbest.copy()
        printlog("\nRUNNING EMCEE - FIT TTV", olog=olog)
        sys.stdout.flush()

        if "log_S0" in params_fit:
            log_S0, log_omega0 = params_fit["log_S0"], params_fit["log_omega0"]
        else:
            log_S0, log_omega0 = None, None

        result_fit = M.fit_transit(
            T_0=params_fit["T_0"].value,
            P=params_fit["P"].value,
            # TTV yes or not, default: ttv=False
            ttv=True,
            # default for decorrelation:
            unroll=visit_args["unroll"],
            nroll=visit_args["nroll"],
            # if you want to check the effect of roll angle or how it does without
            # unroll=False,
            unwrap=visit_args["unwrap"],  # defaul False
            D=params_fit["D"],
            W=params_fit["W"],
            b=params_fit["b"],
            f_c=params_fit["f_c"],
            f_s=params_fit["f_s"],
            h_1=params_fit["h_1"],
            h_2=params_fit["h_2"],
            log_omega0=log_omega0,
            log_S0=log_S0,
            extra_priors=extra_priors,
            burn=emcee_args["nprerun"],
            steps=emcee_args["nsteps"],
            nwalkers=emcee_args["nwalkers"],
            progress=emcee_args["progress"],
        )

        printlog("REPORT", olog=olog)
        printlog("{}".format(M.fit_report(min_correl=0.8)), olog=olog)

        printlog("PARAMETERS MULTIVISIT - FIT", olog=olog)
        par_med, stats_med, par_mle, stats_mle = pyca.get_best_parameters(
            result_fit, M, nburn=0, dataset_type="multivisit"
        )
        # updates params/parbest in result and M.result
        result_fit.params = par_med.copy()
        M.result.params = par_med.copy()
        result_fit.parbest = par_mle.copy()
        M.result.parbest = par_mle.copy()
        pyca.quick_save_params(
            os.path.join(main_folder, "params_med_fit.dat"), par_med, bjd_ref=cst.btjd
        )
        pyca.quick_save_params(
            os.path.join(main_folder, "params_mle_fit.dat"), par_mle, bjd_ref=cst.btjd
        )

        printlog("LC no-detrend plot", olog=olog)
        fig = M.plot_fit(
            title="Not detrended",
            data_offset=0.01,
            binwidth=bin30m_ph,
            res_offset=0.005,
            detrend=False,
        )
        for ext in fig_ext:
            plt_file = os.path.join(
                main_folder, "lcs_nodetrend_plot_fit.{}".format(ext)
            )
            fig.savefig(plt_file, bbox_inches="tight")
        plt.close(fig)

        printlog("LC detrended plot", olog=olog)
        fig = M.plot_fit(
            title="Detrended",
            data_offset=0.01,
            binwidth=bin30m_ph,
            res_offset=0.005,
            detrend=True,
        )
        for ext in fig_ext:
            plt_file = os.path.join(main_folder, "lcs_detrend_plot_fit.{}".format(ext))
            fig.savefig(plt_file, bbox_inches="tight")
        plt.close(fig)

        fig, out_fit = pyca.custom_plot_phase(M, result_fit, title="Fit TTV")
        for ext in fig_ext:
            plt_file = os.path.join(main_folder, "lcs_phased_plot_fit.{}".format(ext))
            fig.savefig(plt_file, bbox_inches="tight")
        plt.close(fig)
        printlog("{}".format(plt_file), olog=olog)
        printlog("{}".format(os.path.splitext(plt_file)), olog=olog)
        out_file = "{}.dat".format(os.path.splitext(plt_file)[0])
        out = np.column_stack([v for v in out_fit.values()])
        head = "".join(["{:s} ".format(k) for k in out_fit.keys()])
        fmt = "%23.16e " * (len(out_fit) - 1) + "%03.0f"
        np.savetxt(out_file, out, header=head, fmt=fmt)

        printlog("Trace plot", olog=olog)
        fig = M.trail_plot(plotkeys="all", plot_kws={"alpha": 0.1})
        for ext in fig_ext:
            plt_file = os.path.join(main_folder, "trace_plot_all_fit.{}".format(ext))
            fig.savefig(plt_file, bbox_inches="tight")
        plt.close(fig)

        printlog("Corner plot", olog=olog)
        pk = []
        for n in result_fit.params:
            if result_fit.params[n].vary:
                if "ttv" in n or n in gnames:
                    pk.append(n)
        printlog("{}".format(pk), olog=olog)
        fig = M.corner_plot(plotkeys=pk)
        for ext in fig_ext:
            plt_file = os.path.join(main_folder, "corner_plot_all_fit.{}".format(ext))
            fig.savefig(plt_file, bbox_inches="tight")
        plt.close(fig)

        printlog("MASSRADIUS MULTIVISIT", olog=olog)
        try:
            _, fig = M.massradius(
                m_star=star_args["Mstar"],
                r_star=star_args["Rstar"],
                K=planet_args["Kms"],
                jovian=True,
                verbose=True,
            )
            for ext in fig_ext:
                plt_file = os.path.join(main_folder, "massradius_fit.{}".format(ext))
                fig.savefig(plt_file, bbox_inches="tight")
            plt.close(fig)
        except:
            printlog("massradius error: to be investigated", olog=olog)
        sys.stdout.flush()

        bjdc = cst.btjd
        printlog("\nTTV SUMMARY", olog=olog)
        # extract single visit T_0
        printlog("Input linear ephem: {} + N x {}".format(T_ref, P_ref), olog=olog)
        Tr = T_ref - bjdc
        T0s, err_T0s = [], []
        epo_1, Tlin_1, oc_1 = [], [], []

        for m in M.datasets:
            tt = m.emcee.params["T_0"].value  # + m.bjd_ref - bjdc
            ett = m.emcee.params["T_0"].stderr
            T0s.append(tt)
            err_T0s.append(ett)

            epo = np.rint((tt - Tr.n) / P_ref.n)
            tl = Tr + epo * P_ref  # lep.linear_transit_time(Tr.n, P_ref.n, epo)
            epo_1.append(epo)
            Tlin_1.append(tl)
            # oc_1.append(ufloat(tt, ett) - tl)
            oc_1.append(tt - tl.n)

        T0s, err_T0s = np.array(T0s), np.array(err_T0s)
        epo_1, Tlin_1, oc_1 = (
            np.array(epo_1),
            np.array(Tlin_1),
            np.array(oc_1) * cst.day2sec,
        )

        # recompute new T_ref e P_ref from T0s
        epo_a, Tr_x, Pr_x, err_lin_x = lep.compute_lin_ephem(
            T0s, eT0=err_T0s, epoin=epo_1, modefit="wls"
        )

        Tr_a = ufloat(Tr_x, err_lin_x[0])
        Pr_a = ufloat(Pr_x, err_lin_x[1])
        printlog(
            "Updated with fitted T0s the input linear ephem: {} + N x {}".format(
                Tr_a, Pr_a
            ),
            olog=olog,
        )
        Tlin_a, oc_a = [], []

        Tr_b = ufloat(result_lin.params["T_0"].value, result_lin.params["T_0"].stderr)
        Pr_b = ufloat(result_lin.params["P"].value, result_lin.params["P"].stderr)
        printlog("Fitted linear ephem: {} + N x {}".format(Tr_b, Pr_b), olog=olog)
        epo_b, Tlin_b, oc_b = [], [], []

        oc_c = []

        for i, m in enumerate(M.datasets):
            tl_a = Tr_a + epo_a[i] * Pr_a
            Tlin_a.append(tl_a)
            # oc_a.append(ufloat(T0s[i], err_T0s[i]) - tl_a)
            oc_a.append(T0s[i] - tl_a.n)

            epox = lep.calculate_epoch(T0s[i], Tr_b.n, Pr_b.n)
            tl_b = Tr_b + epox * Pr_b
            epo_b.append(epox)
            Tlin_b.append(tl_b)
            # oc_b.append(ufloat(T0s[i], err_T0s[i]) - tl_b)
            oc_b.append(T0s[i] - tl_b.n)

            k = "ttv_{:02d}".format(i + 1)
            oc_c.append(ufloat(result_fit.params[k].value, result_fit.params[k].stderr))

        printlog("\nT0s {}".format(T0s), olog=olog)
        printlog("err_T0s {}".format(err_T0s), olog=olog)
        printlog("\nepo {}".format(epo_1), olog=olog)
        printlog("Tlin_1 {}".format(Tlin_1), olog=olog)
        printlog("oc_1 (s) {}".format(oc_1), olog=olog)
        epo_a, Tlin_a, oc_a = (
            np.array(epo_a),
            np.array(Tlin_a),
            np.array(oc_a) * cst.day2sec,
        )
        printlog("\nepo_a {}".format(epo_a), olog=olog)
        printlog("Tlin_a {}".format(Tlin_a), olog=olog)
        printlog("oc_a  (s) {}".format(oc_a), olog=olog)
        epo_b, Tlin_b, oc_b = (
            np.array(epo_b),
            np.array(Tlin_b),
            np.array(oc_b) * cst.day2sec,
        )
        printlog("\nepo_b {}".format(epo_b), olog=olog)
        printlog("Tlin_b {}".format(Tlin_b), olog=olog)
        printlog("oc_b  (s) {}".format(oc_b), olog=olog)
        oc_c = np.array(oc_c)
        printlog("\noc_c  (s) {}".format(oc_c), olog=olog)

        printlog("", olog=olog)
        sys.stdout.flush()

        t0_file = os.path.join(main_folder, "T0s_summary.dat")
        with open(t0_file, "w") as f:
            l = "# BJD_TDB - {}".format(bjdc)
            printlog(l, olog=olog)
            f.write(l + "\n")
            l = "# _1: input linear ephem = {:.6f} (+/- {:.6f}) + N x {:.6f} (+/- {:.6f})".format(
                Tr.n, Tr.s, P_ref.n, P_ref.s
            )
            printlog(l, olog=olog)
            f.write(l + "\n")
            l = "# _a linear ephem = {:.6f} (+/- {:.6f}) + N x {:.6f} (+/- {:.6f})".format(
                Tr_a.n, Tr_a.s, Pr_a.n, Pr_a.s
            )
            printlog(l, olog=olog)
            f.write(l + "\n")
            l = "# _b linear ephem = {:.6f} (+/- {:.6f}) + N x {:.6f} (+/- {:.6f})".format(
                Tr_b.n, Tr_b.s, Pr_b.n, Pr_b.s
            )
            printlog(l, olog=olog)
            f.write(l + "\n")
            l = "# _c fitted TTV (or O-C)"
            printlog(l, olog=olog)
            f.write(l + "\n")
            head = "# 0 epo_1 1 T_0_1 2 err_T_0_1 3 Tlin_1 4 unc_Tlin_1 5 oc_s_1"
            head += " 6 epo_a 7 Tlin_a 8 unc_Tlin_a 9 oc_s_a"
            head += " 10 epo_b 11 Tlin_b 12 unc_Tlin_b 13 oc_s_b"
            head += " 14 oc_s_c 15 err_oc_s_c"
            printlog(head, olog=olog)
            f.write(head + "\n")

            for i, e1 in enumerate(epo_1):

                l1 = "{:+05.0f} {:13.6f} {:+13.6f} {:13.6f} {:+13.6f} {:+13.6f}".format(
                    e1,
                    T0s[i],
                    err_T0s[i],
                    Tlin_1[i].n,
                    Tlin_1[i].s,
                    oc_1[i],
                )
                la = "{:+05.0f} {:13.6f} {:+13.6f} {:+13.6f}".format(
                    epo_a[i],
                    Tlin_a[i].n,
                    Tlin_a[i].s,
                    oc_a[i],
                )
                lb = "{:+05.0f} {:13.6f} {:+13.6f} {:+13.6f}".format(
                    epo_b[i],
                    Tlin_b[i].n,
                    Tlin_b[i].s,
                    oc_b[i],
                )
                lc = "{:+13.6f} {:+13.6f}".format(oc_c[i].n, oc_c[i].s)
                l = "{} {} {} {}".format(l1, la, lb, lc)
                printlog(l, olog=olog)
                f.write(l + "\n")

        print("Plotting O-C...")
        file = np.genfromtxt(t0_file)

        epochs = file[:, 1]

        ocs = file[:, 14]
        ocs_err = file[:, 15]

        plt.errorbar(
            epochs,
            ocs,
            yerr=ocs_err,
            fmt="ko",
        )
        plt.axhline(0, color="firebrick", linestyle="--", linewidth=2)
        plt.xlabel("BJD_TDB - 2457000")
        plt.ylabel("O - C (s)")
        plt.savefig(os.path.join(main_folder, "OC_plot.pdf"))

        return M, result_lin, result_fit


class SingleBayesKeplerTess:
    def __init__(self, input_file):
        self.input_file = input_file
        self.input_pars = [
            "P",
            "T_0",
            "D",
            "W",
            "b",
            "h_1",
            "h_2",
            "f_s",
            "f_c",
            "logrho",
        ]

    # =============================================================================

    def load_fits_file(self, file_fits, olog=None):

        info = {}

        printlog("Reading fits file: {}".format(file_fits), olog=olog)
        # load fits file and extract needed data and header keywords
        with fits.open(file_fits) as hdul:
            hdul.info()
            btjd = hdul[1].header["BJDREFI"]
            data_raw = hdul[1].data
            exp_time = hdul[1].header["TIMEDEL"]

        info["BTJD"] = btjd
        info["EXP_TIME"] = exp_time

        printlog(
            "Time ref. = {} and exposure time {} in days.".format(btjd, exp_time),
            olog=olog,
        )
        nraw = np.shape(data_raw)
        names = data_raw.names.copy()
        data_keys = []
        ok = np.ones(nraw).astype(bool)
        for k in names:
            # if ("PSF" not in k) or ("SAP_FLUX" not in k):
            if "PSF" in k:
                pass
            elif "SAP_FLUX" in k:
                data_keys.append(k)
            else:
                nan = np.isnan(data_raw[k])
                nnan = np.sum(nan)
                printlog("{} ==> n(NaN) = {}".format(k, nnan), olog=olog)
                ok = np.logical_and(ok, ~nan)
                data_keys.append(k)
        nok = np.sum(ok)
        info["n_data"] = nok
        printlog("Total number of good points: {}".format(nok), olog=olog)
        # data = data_raw[ok]
        data = {}
        for k in data_keys:
            data[k] = data_raw[k][ok]

        return data, info

    # =============================================================================

    def get_transit_epochs(self, data, info, visit_args, planet_args, olog=None):

        T_ref = planet_args["T_ref"]
        P = planet_args["P_user_data"]
        Wd = planet_args["W"] * P
        vdurh = visit_args["single_duration_hour"]
        if vdurh is None:
            vdurh = 1.5 * Wd.n * cst.day2hour + 3.0 * CHEOPS_ORBIT_MINUTES * cst.min2day
        vdur = vdurh / cst.day2hour
        hdur = 0.5 * vdur
        vdur_co = vdur * cst.day2min / CHEOPS_ORBIT_MINUTES

        btjd = info["BTJD"]

        printlog("Computing feasible epochs", olog=olog)
        t = data["TIME"] + btjd
        emin = np.rint((np.min(t) - T_ref.n) / P.n)
        x = T_ref.n + P.n * emin
        if x < np.min(t):
            emin += 1
        emax = np.rint((np.max(t) - T_ref.n) / P.n)
        x = T_ref.n + P.n * emax
        if x > np.max(t):
            emax -= 1
        printlog("epoch min = {} max = {}".format(emin, emax), olog=olog)
        epochs = np.arange(emin, emax + 1, 1)

        printlog(
            "Selecting lc portion centered on transit time with duration of {:.5f} d = {:.2} CHEOPS orbits".format(
                vdur, vdur_co
            ),
            olog=olog,
        )

        # printlog("t min: {:.5f}".format(np.min(t)))
        # printlog("t max: {:.5f}".format(np.max(t)))

        transits = []

        for i_epo, epo in enumerate(epochs):
            printlog("", olog=olog)

            bjd_lin = T_ref.n + P.n * epo
            printlog(
                "Epoch = {:.0f} ==> T_0 = {:.5f} BJD_TDB [{:.5f} , {:.5f}] [[{:.5f} , {:.5f}]]".format(
                    epo,
                    bjd_lin,
                    bjd_lin - hdur,
                    bjd_lin + hdur,
                    bjd_lin - (Wd.n * 0.5),
                    bjd_lin + (Wd.n * 0.5),
                ),
                olog=olog,
            )

            sel = np.logical_and(t >= bjd_lin - hdur, t < bjd_lin + hdur)
            nsel = np.sum(sel)
            wsel = np.logical_and(
                t >= bjd_lin - (Wd.n * 0.5), t < bjd_lin + (Wd.n * 0.5)
            )
            nwsel = np.sum(wsel)
            printlog("nsel = {:d} nwsel = {:d}".format(nsel, nwsel), olog=olog)
            if nsel > 0 and nwsel > 3:
                tra = {}
                tra["epoch"] = epo
                tra["data"] = {}
                for k, v in data.items():
                    tra["data"][k] = v[sel]
                bjdref = int(np.min(tra["data"]["TIME"]) + btjd)
                tra["bjdref"] = bjdref
                Tlin = bjd_lin - bjdref
                tra["T_0"] = Tlin
                tra["T_0_bounds"] = [Tlin - 0.5 * Wd.n, Tlin + 0.5 * Wd.n]
                tra["T_0_user_data"] = ufloat(Tlin, 0.5 * Wd.n)
                transits.append(tra)
            else:
                printlog("Not enough points for lc fit.", olog=olog)
                pass

        return transits

    def single_epoch_analysis(
        self,
        transit,
        info,
        star,
        visit_args,
        star_args,
        planet_args,
        emcee_args,
        epoch_folder,
        olog=None,
    ):
        read_file_status = []

        def category_args(par):
            if par in star_args.keys():
                return star_args
            elif par in planet_args.keys():
                return planet_args
            else:
                read_file_status.append(
                    f"ERROR: {par} is not defined in neither the star or planet arguments"
                )

        epoch_name = os.path.basename(epoch_folder)

        printlog("\nLoading dataset", olog=olog)
        # fake dataset
        dataset = FITSDataset(
            file_key="CH_PR100015_TG006701_V0200",  # fake file_key
            target=star.identifier,
            download_all=True,
            view_report_on_download=False,
        )

        printlog("Getting light curve for selected aperture", olog=olog)
        t, f, ef = dataset.get_FITS_lightcurve(
            visit_args, transit, info, reject_highpoints=False, verbose=True
        )

        if visit_args["clip_outliers"] > 0:
            # TO clip or not to clip
            printlog("Clip outliers", olog=olog)
            t, f, ef = dataset.clip_outliers(verbose=True)

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

        T_0 = transit["T_0"]

        planet_args["T_0"] = transit["T_0"]
        planet_args["T_0_fit"] = True
        planet_args["T_0_bounds"] = transit["T_0_bounds"]
        planet_args["T_0_user_data"] = transit["T_0_user_data"]

        # # RV SEMI-AMPLITUDE IN m/s NEEDED TO USE MASSRADIUS FUNCTION
        # Kms = planet_args['Kms']

        printlog(
            "\nPARAMETERS OF INPUT - CUSTOM - TO BE MODIFY FOR EACH TARGET & PLANET & VISIT",
            olog=olog,
        )
        printlog("P    = {} d".format(P), olog=olog)
        printlog("k    = {} ==> D = k^2 = {}".format(k, D), olog=olog)
        printlog("i    = {} deg".format(inc), olog=olog)
        printlog("a/Rs = {}".format(aRs), olog=olog)
        printlog("b    = {}".format(b), olog=olog)
        printlog("W    = {}*P = {} d = {} h".format(W, W * P, W * P * 24.0), olog=olog)
        printlog("ecc  = {}".format(ecc), olog=olog)
        printlog("w    = {}".format(w), olog=olog)
        printlog("f_c  = {}".format(f_c), olog=olog)
        printlog("f_s  = {}".format(f_s), olog=olog)
        printlog("T_0  = {}\n".format(transit["T_0_user_data"]), olog=olog)

        # determine the out-of-transit lc for initial guess of c based on T_0 min/max
        oot = np.logical_or(t < transit["T_0_bounds"][0], t > transit["T_0_bounds"][1])

        # DEFINE HERE HOW TO USE THE PARAMETERS,
        # WE HAVE TO DEFINE IF IT VARY (FIT) OR NOT (FIXED)

        # TODO logrho seems to be broken when passing from the old way to read data to the new one
        in_par = Parameters()

        for key in self.input_pars:
            print(key)
            cat = category_args(key)
            if key in ["D", "W", "b"]:
                # Randomize here
                val = np.abs(
                    np.random.normal(
                        loc=cat[key + "_user_data"].n, scale=cat[key + "_user_data"].s
                    )
                )
            else:
                val = cat[key]
            in_par[key] = Parameter(
                key,
                value=val,
                vary=cat[key + "_fit"],
                min=cat[key + "_bounds"][0],
                max=cat[key + "_bounds"][1],
                user_data=cat[key + "_user_data"],
            )

        if visit_args["shape"] == "fix":
            for n in ["D", "W", "b"]:
                in_par[n].vary = False
            in_par["D"].value = D.n
            in_par["W"].value = W.n
            in_par["b"].value = b.n

        # c calculated separately
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

        t_exp_s = info["EXP_TIME"] * cst.day2sec

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
            t_exp_s=t_exp_s,
        )

        lmfit0_rep = dataset.lmfit_report(min_correl=0.5)
        # for l in lmfit0_rep:
        #   printlog(l, olog=olog)
        printlog(lmfit0_rep, olog=olog)
        printlog("", olog=olog)

        # # roll angle plot
        # fig = dataset.rollangle_plot(figsize=plt.rcParams["figure.figsize"], fontsize=8)
        # for ext in fig_ext:
        #     fig.savefig(
        #         os.path.join(
        #             epoch_folder, "00_lmfit0_roll_angle_vs_residual.{}".format(ext)
        #         ),
        #         bbox_inches="tight",
        #     )
        # plt.close(fig)

        # input params plot

        k = "t_exp"
        in_par[k] = Parameter(k, value=lmfit0.params[k].value, vary=False)
        k = "n_over"
        in_par[k] = Parameter(k, value=lmfit0.params[k].value, vary=False)

        fig, _ = pyca.model_plot_fit(
            dataset,
            in_par,
            par_type="input",
            nsamples=0,
            flatchains=None,
            model_filename=os.path.join(epoch_folder, "00_lc_0_input.dat"),
        )
        for ext in fig_ext:
            fig.savefig(
                os.path.join(epoch_folder, "00_lc_0_input.{}".format(ext)),
                bbox_inches="tight",
            )
        plt.close(fig)
        pyca.quick_save_params(
            os.path.join(epoch_folder, "00_params_0_input.dat"),
            in_par,
            dataset.lc["bjd_ref"],
        )

        # best-fit plot
        params_lm0 = lmfit0.params.copy()
        fig, _ = pyca.model_plot_fit(
            dataset,
            params_lm0,
            par_type="lm",
            nsamples=0,
            flatchains=None,
            model_filename=os.path.join(epoch_folder, "00_lc_lmfit0.dat"),
        )
        for ext in fig_ext:
            fig.savefig(
                os.path.join(epoch_folder, "00_lc_lmfit0.{}".format(ext)),
                bbox_inches="tight",
            )
        plt.close(fig)
        pyca.quick_save_params(
            os.path.join(epoch_folder, "00_params_lmfit0.dat"),
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
            # if (
            #     (k not in ["dfdcontam", "dfdsmear", "ramp", "glint_scale"])
            #     or ("dfdcos" not in k)
            #     or ("dfdsin" not in k)
            # ):  # phi not in Kepler/TESS
            if (
                (k in ["dfdcontam", "dfdsmear", "ramp", "glint_scale"])
                or ("dfdcos" in k)
                or ("dfdsin" in k)
            ):
                pass
            else:
                det_par[k] = dprior
            printlog("{:20s} = {}".format(k, det_par[k]), olog=olog)
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
                # T_0=in_par["T_0"],
                f_c=in_par["f_c"],
                f_s=in_par["f_s"],
                # D=in_par["D"],
                # W=in_par["W"],
                # b=in_par["b"],
                # h_1=in_par["h_1"],
                # h_2=in_par["h_2"],
                T_0=params_lm0["T_0"],
                D=params_lm0["D"],
                W=params_lm0["W"],
                b=params_lm0["b"],
                h_1=params_lm0["h_1"],
                h_2=params_lm0["h_2"],
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
                t_exp_s=info["EXP_TIME"] * cst.day2sec,
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

        # # roll angle plot
        # fig = dataset.rollangle_plot(figsize=plt.rcParams["figure.figsize"], fontsize=8)
        # for ext in fig_ext:
        #     fig.savefig(
        #         os.path.join(
        #             epoch_folder, "01_lmfit_loop_roll_angle_vs_residual.{}".format(ext)
        #         ),
        #         bbox_inches="tight",
        #     )
        # plt.close(fig)
        # best-fit plot
        fig, _ = pyca.model_plot_fit(
            dataset,
            params_lm_loop,
            par_type="lm",
            nsamples=0,
            flatchains=None,
            model_filename=os.path.join(epoch_folder, "01_lc_lmfit_loop.dat"),
        )
        for ext in fig_ext:
            fig.savefig(
                os.path.join(epoch_folder, "01_lc_lmfit_loop.{}".format(ext)),
                bbox_inches="tight",
            )
        plt.close(fig)
        stats_lm = pyca.computes_rms(
            dataset, params_best=params_lm_loop, glint=False, olog=olog
        )
        pyca.quick_save_params(
            os.path.join(epoch_folder, "01_params_lmfit_loop.dat"),
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
        nthreads = emcee_args["nthreads"]
        progress = emcee_args["progress"]

        # Run emcee from last best fit
        printlog("\n-Run emcee from last best fit with:", olog=olog)
        printlog(" nwalkers = {}".format(nwalkers), olog=olog)
        printlog(" nprerun  = {}".format(nprerun), olog=olog)
        printlog(" nsteps   = {}".format(nsteps), olog=olog)
        printlog(" nburn    = {}".format(nburn), olog=olog)
        printlog(" nthin    = {}".format(nthin), olog=olog)
        printlog(" nthreads = {}".format(nthreads), olog=olog)
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
                os.path.join(epoch_folder, "02_trace_emcee_all.{}".format(ext)),
                bbox_inches="tight",
            )
        plt.close(fig)

        printlog("\n-Plot corner full from pycheops (not removed nburn)", olog=olog)
        fig = dataset.corner_plot(plotkeys="all")
        for ext in fig_ext:
            fig.savefig(
                os.path.join(epoch_folder, "02_corner_emcee_all.{}".format(ext)),
                bbox_inches="tight",
            )
        plt.close(fig)

        printlog(
            "\n-Computing my parameters and plot models with random samples", olog=olog
        )
        params_med, stats_med, params_mle, stats_mle = pyca.get_best_parameters(
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
            os.path.join(epoch_folder, "02_params_emcee_median.dat"),
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
            model_filename=os.path.join(epoch_folder, "02_lc_emcee_median.dat"),
        )
        for ext in fig_ext:
            fig.savefig(
                os.path.join(epoch_folder, "02_lc_emcee_median.{}".format(ext)),
                bbox_inches="tight",
            )
        plt.close(fig)
        fig = pyca.plot_fft(dataset, params_med, star=star)
        for ext in fig_ext:
            fig.savefig(
                os.path.join(epoch_folder, "02_fft_emcee_median.{}".format(ext)),
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
            os.path.join(epoch_folder, "02_params_emcee_mle.dat"),
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
            model_filename=os.path.join(epoch_folder, "02_lc_emcee_mle.dat"),
        )
        for ext in fig_ext:
            fig.savefig(
                os.path.join(epoch_folder, "02_lc_emcee_mle.{}".format(ext)),
                bbox_inches="tight",
            )
        plt.close(fig)
        fig = pyca.plot_fft(dataset, params_mle, star=star)
        for ext in fig_ext:
            fig.savefig(
                os.path.join(epoch_folder, "02_fft_emcee_mle.{}".format(ext)),
                bbox_inches="tight",
            )
        plt.close(fig)

        params = {"med": params_med, "mle": params_mle}

        file_emcee = pyca.save_dataset(
            dataset, epoch_folder, star.identifier, epoch_name, gp=False
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
                os.path.join(epoch_folder, "03_trace_emcee_gp_train.{}".format(ext)),
                bbox_inches="tight",
            )
        plt.close(fig)

        printlog(
            "\n-Computing my parameters and plot models with random samples", olog=olog
        )
        (
            params_med_gp_train,
            stats_med,
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
                epoch_folder, "03_lc_emcee_median_gp_train.dat"
            ),
        )
        for ext in fig_ext:
            fig.savefig(
                os.path.join(
                    epoch_folder, "03_lc_emcee_median_gp_train.{}".format(ext)
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
            model_filename=os.path.join(epoch_folder, "03_lc_emcee_mle_gp_train.dat"),
        )
        for ext in fig_ext:
            fig.savefig(
                os.path.join(epoch_folder, "03_lc_emcee_mle_gp_train.{}".format(ext)),
                bbox_inches="tight",
            )
        plt.close(fig)

        ### *** =======================================++=====================
        ### *** ===== FIT TRANSIT + DETRENDING + GP =++=======================
        printlog("\nRUN FULL FIT TRANSIT+DETRENDING+GP W/ EMCEE", olog=olog)

        params_fit_gp = pyca.copy_parameters(params_mle)
        for p in ["log_S0", "log_omega0", "log_sigma"]:
            # params_fit_gp[p] = params_mle_gp_train[p]
            # printlog("{} = {} user_data = {} (vary = {})".format(p,
            #   params_fit_gp[p], params_fit_gp[p].user_data, params_fit_gp[p].vary),
            #   olog=olog
            # )
            # params_fit_gp[p].user_data = ufloat(params_mle_gp_train[p].value, 2*params_mle_gp_train[p].stderr)
            # printlog("{} = {} user_data = {} (vary = {})".format(p,
            #   params_fit_gp[p], params_fit_gp[p].user_data, params_fit_gp[p].vary),
            #   olog=olog
            # )
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
                os.path.join(epoch_folder, "04_trace_emcee_all.{}".format(ext)),
                bbox_inches="tight",
            )
        plt.close(fig)

        printlog("\n-Plot corner full from pycheops (not removed nburn)", olog=olog)
        fig = dataset.corner_plot(plotkeys="all")
        for ext in fig_ext:
            fig.savefig(
                os.path.join(epoch_folder, "04_corner_emcee_all.{}".format(ext)),
                bbox_inches="tight",
            )
        plt.close(fig)

        printlog(
            "\n-Computing my parameters and plot models with random samples", olog=olog
        )
        (
            params_med_gp,
            stats_med_gp,
            params_mle_gp,
            stats_mle_gp,
        ) = pyca.get_best_parameters(
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
            os.path.join(epoch_folder, "04_params_emcee_median_gp.dat"),
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
            model_filename=os.path.join(epoch_folder, "04_lc_emcee_median_gp.dat"),
        )
        for ext in fig_ext:
            fig.savefig(
                os.path.join(epoch_folder, "04_lc_emcee_median_gp.{}".format(ext)),
                bbox_inches="tight",
            )
        plt.close(fig)
        fig = pyca.plot_fft(dataset, params_med_gp, star=star)
        for ext in fig_ext:
            fig.savefig(
                os.path.join(epoch_folder, "04_fft_emcee_median_gp.{}".format(ext)),
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
            os.path.join(epoch_folder, "04_params_emcee_mle_gp.dat"),
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
            model_filename=os.path.join(epoch_folder, "04_lc_emcee_mle_gp.dat"),
        )
        for ext in fig_ext:
            fig.savefig(
                os.path.join(epoch_folder, "04_lc_emcee_mle_gp.{}".format(ext)),
                bbox_inches="tight",
            )
        plt.close(fig)
        fig = pyca.plot_fft(dataset, params_mle_gp, star=star)
        for ext in fig_ext:
            fig.savefig(
                os.path.join(epoch_folder, "04_fft_emcee_mle_gp.{}".format(ext)),
                bbox_inches="tight",
            )
        plt.close(fig)

        params_gp = {"med": params_med_gp, "mle": params_mle_gp}

        file_emcee = pyca.save_dataset(
            dataset, epoch_folder, star.identifier, epoch_name, gp=True
        )
        printlog("-Dumped dataset into file {}".format(file_emcee), olog=olog)

        return (
            stats_lm,
            stats_med,
            stats_mle,
            params,
            stats_med_gp,
            stats_mle_gp,
            params_gp,
        )

    # =============================================================================

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

        # (
        #     visit_args,
        #     star_args,
        #     planet_args,
        #     emcee_args,
        #     read_file_status,
        # ) = self.read_file()

        # seed = 42
        seed = visit_args["seed"]
        np.random.seed(seed)

        passband = visit_args["passband"]
        aperture = visit_args["aperture"]
        shape = visit_args["shape"]
        file_fits = visit_args["file_fits"]
        file_name = os.path.basename(file_fits).replace(
            ".fits", "_{}_{}".format(aperture, shape)
        )

        logs_folder = os.path.join(visit_args["main_folder"], "logs")
        if not os.path.isdir(logs_folder):
            os.makedirs(logs_folder, exist_ok=True)
        log_file = os.path.join(logs_folder, "{}_{}.log".format(start_time, file_name))
        olog = open(log_file, "w")

        printlog("", olog=olog)
        printlog(
            "###################################################################",
            olog=olog,
        )
        printlog(
            " ANALYSIS OF TESS/KEPLER OBSERVATION - DET. PARS. FROM BAYES FACTOR",
            olog=olog,
        )
        printlog(
            "###################################################################",
            olog=olog,
        )

        # =====================
        # TARGET STAR
        # =====================
        # target name, without planet or something else. It has to be a name in simbad
        star_name = star_args["star_name"]

        printlog("TARGET: {}".format(star_name), olog=olog)
        printlog("FILE: {}".format(file_fits), olog=olog)
        printlog("PASSBAND: {}".format(passband), olog=olog)
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

        main_folder = os.path.join(visit_args["main_folder"], file_name)
        if not os.path.isdir(main_folder):
            os.makedirs(main_folder, exist_ok=True)

        printlog("SAVING OUTPUT INTO FOLDER {}".format(main_folder), olog=olog)

        # stellar parameters
        # from WG TS3
        Rstar = star_args["Rstar"]
        Mstar = star_args["Mstar"]
        teff = star_args["teff"]
        logg = star_args["logg"]
        feh = star_args["feh"]

        star = pyca.CustomStarProperties(
            star_name,
            match_arcsec=None,
            teff=teff,
            logg=logg,
            metal=feh,
            dace=False,
            passband=passband,
        )

        printlog("STAR INFORMATION", olog=olog)
        printlog(star, olog=olog)

        try:
            star_args["logrho"] = star.logrho.n
        except AttributeError:
            star_args["logrho"] = star.logrho
        star_args["logrho_fit"] = True
        star_args["logrho_bounds"] = [-9, 6]
        star_args["logrho_user_data"] = star.logrho

        if star.logrho is None:
            printlog("logrho not available from sweetcat...computed:", olog=olog)
            rho_star = Mstar / (Rstar ** 3)
            logrho = um.log10(rho_star)
            star.logrho = logrho
            printlog("logrho = {}".format(logrho), olog=olog)

        printlog("rho  = {} rho_sun".format(10 ** star.logrho), olog=olog)

        # ======================================================================
        # Load data
        fits_data, fits_info = self.load_fits_file(file_fits)
        transits = self.get_transit_epochs(
            fits_data, fits_info, visit_args, planet_args, olog=olog
        )

        printlog("Loop on transits ...", olog=olog)
        # loop on transits found in the TESS/Kepler data
        head = "# EPOCH"
        head = "{0:s} {1:s}_RChiSq {1:s}_lnL {1:s}_lnP {1:s}_BIC {1:s}_AIC {1:s}_RMS".format(
            head, "LM"
        )
        head = "{0:s} {1:s}_RChiSq {1:s}_lnL {1:s}_lnP {1:s}_BIC {1:s}_AIC {1:s}_RMS".format(
            head, "MED"
        )
        head = "{0:s} {1:s}_RChiSq {1:s}_lnL {1:s}_lnP {1:s}_BIC {1:s}_AIC {1:s}_RMS".format(
            head, "MLE"
        )
        head = "{0:s} err_T0_s".format(head)
        head = "{0:s} {1:s}_RChiSq_{2:s} {1:s}_lnL_{2:s} {1:s}_lnP_{2:s} {1:s}_BIC_{2:s} {1:s}_AIC_{2:s} {1:s}_RMS_{2:s}".format(
            head, "MED", "GP"
        )
        head = "{0:s} {1:s}_RChiSq_{2:s} {1:s}_lnL_{2:s} {1:s}_lnP_{2:s} {1:s}_BIC_{2:s} {1:s}_AIC_{2:s} {1:s}_RMS_{2:s}".format(
            head, "MLE", "GP"
        )
        head = "{0:s} err_T0_s_{1:s}".format(head, "GP")
        lines = []

        for transit in transits:
            epo = transit["epoch"]
            printlog("\nTransit with epoch = {:.0f}".format(epo), olog=olog)
            # creates sub folder
            epoch_name = "epoch_{:04.0f}".format(epo)
            epoch_folder = os.path.join(main_folder, epoch_name)
            printlog("Output will be stored in {}".format(epoch_folder), olog=olog)
            if not os.path.isdir(epoch_folder):
                os.makedirs(epoch_folder, exist_ok=True)

            (
                s_lm,
                s_med,
                s_mle,
                params,
                s_med_gp,
                s_mle_gp,
                params_gp,
            ) = self.single_epoch_analysis(
                transit,
                fits_info,
                star,
                visit_args,
                star_args,
                planet_args,
                emcee_args,
                epoch_folder,
                olog=olog,
            )

            line = "{:04.0f}".format(epo)

            s = s_lm["flux-all (w/o GP)"]
            line = "{:s} {:12.3f} {:12.3f} {:12.3f} {:12.3f} {:12.3f} {:12.3f} {:12.3f}".format(
                line,
                s["Red. ChiSqr"],
                s["lnL"],
                s["lnP"],
                s["BIC"],
                s["AIC"],
                s["lnL"],
                s["RMS (unbinned)"][0],
            )

            s = s_med["flux-all (w/o GP)"]
            line = "{:s} {:12.3f} {:12.3f} {:12.3f} {:12.3f} {:12.3f} {:12.3f} {:12.3f}".format(
                line,
                s["Red. ChiSqr"],
                s["lnL"],
                s["lnP"],
                s["BIC"],
                s["AIC"],
                s["lnL"],
                s["RMS (unbinned)"][0],
            )
            s = s_mle["flux-all (w/o GP)"]
            line = "{:s} {:12.3f} {:12.3f} {:12.3f} {:12.3f} {:12.3f} {:12.3f} {:12.3f}".format(
                line,
                s["Red. ChiSqr"],
                s["lnL"],
                s["lnP"],
                s["BIC"],
                s["AIC"],
                s["lnL"],
                s["RMS (unbinned)"][0],
            )
            line = "{:s} {:7.1f}".format(
                line, params["med"]["T_0"].stderr * cst.day2sec
            )

            s = s_med_gp["flux-all (w/ GP)"]
            line = "{:s} {:12.3f} {:12.3f} {:12.3f} {:12.3f} {:12.3f} {:12.3f} {:12.3f}".format(
                line,
                s["Red. ChiSqr"],
                s["lnL"],
                s["lnP"],
                s["BIC"],
                s["AIC"],
                s["lnL"],
                s["RMS (unbinned)"][0],
            )
            s = s_mle_gp["flux-all (w/ GP)"]
            line = "{:s} {:12.3f} {:12.3f} {:12.3f} {:12.3f} {:12.3f} {:12.3f} {:12.3f}".format(
                line,
                s["Red. ChiSqr"],
                s["lnL"],
                s["lnP"],
                s["BIC"],
                s["AIC"],
                s["lnL"],
                s["RMS (unbinned)"][0],
            )
            line = "{:s} {:7.1f}".format(
                line, params_gp["med"]["T_0"].stderr * cst.day2sec
            )
            printlog("", olog=olog)
            printlog(head, olog=olog)
            printlog(line, olog=olog)

            lines.append(line)
        # END for transit in transits

        printlog("", olog=olog)
        printlog("# === SUMMARY === #", olog=olog)
        printlog("", olog=olog)
        summary_file = os.path.join(main_folder, "quick_summary.dat")
        ofs = open(summary_file, "w")
        printlog(head, olog=olog)
        ofs.write("{:s}\n".format(head))
        for line in lines:
            printlog(line, olog=olog)
            ofs.write("{:s}\n".format(line))
        ofs.close()

        printlog("", olog=olog)
        printlog(" *********** ", olog=olog)
        printlog("--COMPLETED--", olog=olog)
        printlog(" *********** ", olog=olog)
        printlog("", olog=olog)

        olog.close()

        return


class SingleBayesASCII:
    def __init__(self, input_file):
        self.input_file = input_file
        self.input_pars = [
            "P",
            "T_0",
            "D",
            "W",
            "b",
            "h_1",
            "h_2",
            "f_s",
            "f_c",
            "logrho",
        ]

    def get_ld_h1h2(self, visit_args):

        ld = visit_args["input_LD"]
        ld_type = ld["type"]
        coeff = ld["coeff"]
        if ld_type == "quad":
            u1, u2 = coeff[0], coeff[1]
            q1, q2 = pyca.u1u2_to_q1q2(u1, u2)
            h1, h2 = q1q2_to_h1h2(q1, q2)
        elif ld_type == "power2":
            c, alpha = coeff[0], coeff[1]
            h1, h2 = ca_to_h1h2(c, alpha)
        else:  # assumed power2_h1h2: h1, h2
            h1, h2 = coeff[0], coeff[1]

        return h1, h2

    # =============================================================================

    def load_ascii_file(self, visit_args, olog=None):

        file_ascii = os.path.abspath(visit_args["file_ascii"])
        file_columns = visit_args["file_columns"]

        printlog("Reading ascii file: {}".format(file_ascii), olog=olog)

        # data = np.genfromtxt(file_ascii, names = file_columns)
        # printlog("{}".format(data.dtype.names), olog=olog)
        # print(data)
        d = np.genfromtxt(file_ascii)
        data = {}
        for i, k in enumerate(file_columns):
            data[k] = d[:, i]
        printlog("with {}".format([k for k in data.keys()]), olog=olog)
        printlog("ndata = {}".format(len(data[file_columns[0]])), olog=olog)

        return data

    # # =============================================================================

    # ======================================================================
    def single_Bayes_ASCII(
        self,
        ascii_data,
        star_args,
        visit_args,
        planet_args,
        emcee_args,
        epoch_folder,
        olog=None,
        n_threads=1,
    ):
        def category_args(par):
            if par in star_args.keys():
                return star_args
            elif par in planet_args.keys():
                return planet_args
            else:
                self.read_file_status.append(
                    f"ERROR: {par} is not defined in neither the star or planet arguments"
                )

        epoch_name = os.path.basename(epoch_folder)

        printlog("\nLoading dataset", olog=olog)
        # fake dataset
        dataset = pyca.AsciiDataset(
            file_key="CH_PR100015_TG006701_V0200",  # fake file_key
            target=star_args["star_name"],
            download_all=True,
            view_report_on_download=False,
            n_threads=n_threads,
        )

        printlog("Getting light curve for selected aperture", olog=olog)
        t, f, ef = dataset.get_ascii_lightcurve(
            ascii_data,
            normalise=visit_args["normalise_flux"],
            reject_highpoints=False,
            verbose=True,
        )

        fig = plt.figure()
        plt.title("Original light curve")
        plt.errorbar(
            t,
            f,
            yerr=ef,
            color="C0",
            marker="o",
            ms=3,
            mec=pyca.out_color,
            mew=0.5,
            ls="",
            ecolor=pyca.out_color,
            elinewidth=0.5,
            capsize=0,
        )
        plt.ylabel("flux")
        plt.xlabel("$\mathrm{{BJD}}_\mathrm{{TDB}} - {}$".format(dataset.lc["bjd_ref"]))
        for ext in fig_ext:
            fig.savefig(
                os.path.join(epoch_folder, "00_lc_original.{}".format(ext)),
                bbox_inches="tight",
            )
        plt.close(fig)

        # from WG TS3
        Rstar = star_args["Rstar"]
        Mstar = star_args["Mstar"]
        teff = star_args["teff"]
        logg = star_args["logg"]
        feh = star_args["feh"]

        star = StarProperties(
            star_args["star_name"],
            match_arcsec=None,
            teff=teff,
            logg=logg,
            metal=feh,
            dace=False,
        )

        # TO clip or not to clip
        printlog("Clip outliers", olog=olog)
        t, f, ef = dataset.clip_outliers(verbose=True)

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

        Tref = planet_args["T_ref"]
        epo = np.rint((dataset.lc["bjd_ref"] - Tref.n) / P)
        Tlin = Tref.n + epo * P - dataset.lc["bjd_ref"]

        planet_args["T_0"] = Tlin
        planet_args["T_0_bounds"] = [Tlin - W * P * 0.5, Tlin + W * P * 0.5]
        planet_args["T_0_user_data"] = ufloat(Tlin, max(planet_args["T_0_bounds"]))

        # T_0 = (Tlin - W * 0.5, Tlin, Tlin + W * 0.5)

        # # RV SEMI-AMPLITUDE IN m/s NEEDED TO USE MASSRADIUS FUNCTION
        # Kms = planet_args['Kms']

        printlog(
            "\nPARAMETERS OF INPUT - CUSTOM - TO BE MODIFY FOR EACH TARGET & PLANET & VISIT",
            olog=olog,
        )
        printlog("P    = {} d".format(P), olog=olog)
        printlog("k    = {} ==> D = k^2 = {}".format(k, D), olog=olog)
        printlog("i    = {} deg".format(inc), olog=olog)
        printlog("a/Rs = {}".format(aRs), olog=olog)
        printlog("b    = {}".format(b), olog=olog)
        printlog("W    = {}*P = {} d = {} h".format(W, W * P, W * P * 24.0), olog=olog)
        printlog("ecc  = {}".format(ecc), olog=olog)
        printlog("w    = {}".format(w), olog=olog)
        printlog("f_c  = {}".format(f_c), olog=olog)
        printlog("f_s  = {}".format(f_s), olog=olog)
        printlog("T_0  = {}\n".format(planet_args["T_0_user_data"]), olog=olog)

        # determine the out-of-transit lc for initial guess of c based on T_0 min/max
        oot = np.logical_or(
            t < planet_args["T_0_bounds"][0], t > planet_args["T_0_bounds"][1]
        )

        # DEFINE HERE HOW TO USE THE PARAMETERS,
        # WE HAVE TO DEFINE IF IT VARY (FIT) OR NOT (FIXED)
        in_par = Parameters()

        for key in self.input_pars:
            print(key)
            cat = category_args(key)
            if key in ["D", "W", "b"]:
                # Randomize here
                val = np.abs(
                    np.random.normal(
                        loc=cat[key + "_user_data"].n, scale=cat[key + "_user_data"].s
                    )
                )
            else:
                val = cat[key]
            in_par[key] = Parameter(
                key,
                value=val,
                vary=cat[key + "_fit"],
                min=cat[key + "_bounds"][0],
                max=cat[key + "_bounds"][1],
                user_data=cat[key + "_user_data"],
            )

        # c parameter treated separately
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
        #   printlog(l, olog=olog)
        printlog(lmfit0_rep, olog=olog)
        printlog("", olog=olog)

        # roll angle plot
        fig = dataset.rollangle_plot(figsize=plt.rcParams["figure.figsize"], fontsize=8)
        for ext in fig_ext:
            fig.savefig(
                os.path.join(
                    epoch_folder, "00_lmfit0_roll_angle_vs_residual.{}".format(ext)
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
            model_filename=os.path.join(epoch_folder, "00_lc_lmfit0.dat"),
        )
        for ext in fig_ext:
            fig.savefig(
                os.path.join(epoch_folder, "00_lc_lmfit0.{}".format(ext)),
                bbox_inches="tight",
            )
        plt.close(fig)
        pyca.quick_save_params(
            os.path.join(epoch_folder, "00_params_lmfit0.dat"),
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
        # for k in det_par.keys():
        #   if(k not in ['dfdcontam', 'dfdsmear', 'ramp', 'glint_scale']):
        #     det_par[k] = dprior
        for k in dataset.lc["header"]:
            if k not in ["time", "flux", "flux_err"]:
                det_par[k] = dprior

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
                    epoch_folder, "01_lmfit_loop_roll_angle_vs_residual.{}".format(ext)
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
            model_filename=os.path.join(epoch_folder, "01_lc_lmfit_loop.dat"),
        )
        for ext in fig_ext:
            fig.savefig(
                os.path.join(epoch_folder, "01_lc_lmfit_loop.{}".format(ext)),
                bbox_inches="tight",
            )
        plt.close(fig)
        stats_lm = pyca.computes_rms(
            dataset, params_best=params_lm_loop, glint=False, olog=olog
        )
        pyca.quick_save_params(
            os.path.join(epoch_folder, "01_params_lmfit_loop.dat"),
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
        printlog(" nprerun  = {}".format(nprerun), olog=olog)
        printlog(" nsteps   = {}".format(nsteps), olog=olog)
        printlog(" nburn    = {}".format(nburn), olog=olog)
        printlog(" nthin    = {}".format(nthin), olog=olog)
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
                os.path.join(epoch_folder, "02_trace_emcee_all.{}".format(ext)),
                bbox_inches="tight",
            )
        plt.close(fig)

        printlog("\n-Plot corner full from pycheops (not removed nburn)", olog=olog)
        fig = dataset.corner_plot(plotkeys="all")
        for ext in fig_ext:
            fig.savefig(
                os.path.join(epoch_folder, "02_corner_emcee_all.{}".format(ext)),
                bbox_inches="tight",
            )
        plt.close(fig)

        printlog(
            "\n-Computing my parameters and plot models with random samples", olog=olog
        )
        params_med, stats_med, params_mle, stats_mle = pyca.get_best_parameters(
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
            os.path.join(epoch_folder, "02_params_emcee_median.dat"),
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
            model_filename=os.path.join(epoch_folder, "02_lc_emcee_median.dat"),
        )
        for ext in fig_ext:
            fig.savefig(
                os.path.join(epoch_folder, "02_lc_emcee_median.{}".format(ext)),
                bbox_inches="tight",
            )
        plt.close(fig)
        fig = pyca.plot_fft(dataset, params_med, star=star)
        for ext in fig_ext:
            fig.savefig(
                os.path.join(epoch_folder, "02_fft_emcee_median.{}".format(ext)),
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
            os.path.join(epoch_folder, "02_params_emcee_mle.dat"),
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
            model_filename=os.path.join(epoch_folder, "02_lc_emcee_mle.dat"),
        )
        for ext in fig_ext:
            fig.savefig(
                os.path.join(epoch_folder, "02_lc_emcee_mle.{}".format(ext)),
                bbox_inches="tight",
            )
        plt.close(fig)
        fig = pyca.plot_fft(dataset, params_mle, star=star)
        for ext in fig_ext:
            fig.savefig(
                os.path.join(epoch_folder, "02_fft_emcee_mle.{}".format(ext)),
                bbox_inches="tight",
            )
        plt.close(fig)

        params = {"med": params_med, "mle": params_mle}

        file_emcee = pyca.save_dataset(
            dataset, epoch_folder, star.identifier, epoch_name, gp=False
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
                os.path.join(epoch_folder, "03_trace_emcee_gp_train.{}".format(ext)),
                bbox_inches="tight",
            )
        plt.close(fig)

        printlog(
            "\n-Computing my parameters and plot models with random samples", olog=olog
        )
        (
            params_med_gp_train,
            stats_med,
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
                epoch_folder, "03_lc_emcee_median_gp_train.dat"
            ),
        )
        for ext in fig_ext:
            fig.savefig(
                os.path.join(
                    epoch_folder, "03_lc_emcee_median_gp_train.{}".format(ext)
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
            model_filename=os.path.join(epoch_folder, "03_lc_emcee_mle_gp_train.dat"),
        )
        for ext in fig_ext:
            fig.savefig(
                os.path.join(epoch_folder, "03_lc_emcee_mle_gp_train.{}".format(ext)),
                bbox_inches="tight",
            )
        plt.close(fig)

        ### *** =======================================++=====================
        ### *** ===== FIT TRANSIT + DETRENDING + GP =++=======================
        printlog("\nRUN FULL FIT TRANSIT+DETRENDING+GP W/ EMCEE", olog=olog)

        params_fit_gp = pyca.copy_parameters(params_mle)
        for p in ["log_S0", "log_omega0", "log_sigma"]:
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
                os.path.join(epoch_folder, "04_trace_emcee_all.{}".format(ext)),
                bbox_inches="tight",
            )
        plt.close(fig)

        printlog("\n-Plot corner full from pycheops (not removed nburn)", olog=olog)
        fig = dataset.corner_plot(plotkeys="all")
        for ext in fig_ext:
            fig.savefig(
                os.path.join(epoch_folder, "04_corner_emcee_all.{}".format(ext)),
                bbox_inches="tight",
            )
        plt.close(fig)

        printlog(
            "\n-Computing my parameters and plot models with random samples", olog=olog
        )
        (
            params_med_gp,
            stats_med_gp,
            params_mle_gp,
            stats_mle_gp,
        ) = pyca.get_best_parameters(
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
            os.path.join(epoch_folder, "04_params_emcee_median_gp.dat"),
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
            model_filename=os.path.join(epoch_folder, "04_lc_emcee_median_gp.dat"),
        )
        for ext in fig_ext:
            fig.savefig(
                os.path.join(epoch_folder, "04_lc_emcee_median_gp.{}".format(ext)),
                bbox_inches="tight",
            )
        plt.close(fig)
        fig = pyca.plot_fft(dataset, params_med_gp, star=star)
        for ext in fig_ext:
            fig.savefig(
                os.path.join(epoch_folder, "04_fft_emcee_median_gp.{}".format(ext)),
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
            os.path.join(epoch_folder, "04_params_emcee_mle_gp.dat"),
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
            model_filename=os.path.join(epoch_folder, "04_lc_emcee_mle_gp.dat"),
        )
        for ext in fig_ext:
            fig.savefig(
                os.path.join(epoch_folder, "04_lc_emcee_mle_gp.{}".format(ext)),
                bbox_inches="tight",
            )
        plt.close(fig)
        fig = pyca.plot_fft(dataset, params_mle_gp, star=star)
        for ext in fig_ext:
            fig.savefig(
                os.path.join(epoch_folder, "04_fft_emcee_mle_gp.{}".format(ext)),
                bbox_inches="tight",
            )
        plt.close(fig)

        params_gp = {"med": params_med_gp, "mle": params_mle_gp}

        file_emcee = pyca.save_dataset(
            dataset, epoch_folder, star.identifier, epoch_name, gp=True
        )
        printlog("-Dumped dataset into file {}".format(file_emcee), olog=olog)

        return (
            stats_lm,
            stats_med,
            stats_mle,
            params,
            stats_med_gp,
            stats_mle_gp,
            params_gp,
        )

    # =============================================================================

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

        file_ascii = visit_args["file_ascii"]
        file_name = "{:s}_pycheops".format(
            os.path.splitext(os.path.basename(file_ascii))[0]
        )

        logs_folder = os.path.join(visit_args["main_folder"], "logs")
        if not os.path.isdir(logs_folder):
            os.makedirs(logs_folder, exist_ok=True)
        log_file = os.path.join(logs_folder, "{}_{}.log".format(start_time, file_name))
        olog = open(log_file, "w")

        printlog("", olog=olog)
        printlog(
            "###################################################################",
            olog=olog,
        )
        printlog(
            " ANALYSIS OF LITERATURE OBSERVATION - DET. PARS. FROM BAYES FACTOR",
            olog=olog,
        )
        printlog(
            "###################################################################",
            olog=olog,
        )

        # =====================
        # TARGET STAR
        # =====================
        # target name, without planet or something else. It has to be a name in simbad
        star_name = star_args["star_name"]

        printlog("TARGET: {}".format(star_name), olog=olog)
        printlog("FILE: {}".format(file_ascii), olog=olog)

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

        main_folder = os.path.join(visit_args["main_folder"], file_name)
        if not os.path.isdir(main_folder):
            os.makedirs(main_folder, exist_ok=True)

        printlog("SAVING OUTPUT INTO FOLDER {}".format(main_folder), olog=olog)

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
        printlog(
            "Updating LD coeff. based on input: {} ==> {}".format(
                visit_args["input_LD"]["type"], visit_args["input_LD"]["coeff"]
            )
        )
        h1, h2 = self.get_ld_h1h2(visit_args)
        star.h_1 = ufloat(h1, star.h_1.s)
        star.h_2 = ufloat(h2, star.h_2.s)

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
        # # Load data
        ascii_data = self.load_ascii_file(visit_args, olog=olog)

        (
            s_lm,
            s_med,
            s_mle,
            params,
            s_med_gp,
            s_mle_gp,
            params_gp,
        ) = self.single_Bayes_ASCII(
            ascii_data,
            star_args,
            visit_args,
            planet_args,
            emcee_args,
            main_folder,
            olog=olog,
        )

        head = "# EPOCH"
        head = "{0:s} {1:s}_RChiSq {1:s}_lnL {1:s}_lnP {1:s}_BIC {1:s}_AIC {1:s}_RMS".format(
            head, "LM"
        )
        head = "{0:s} {1:s}_RChiSq {1:s}_lnL {1:s}_lnP {1:s}_BIC {1:s}_AIC {1:s}_RMS".format(
            head, "MED"
        )
        head = "{0:s} {1:s}_RChiSq {1:s}_lnL {1:s}_lnP {1:s}_BIC {1:s}_AIC {1:s}_RMS".format(
            head, "MLE"
        )
        head = "{0:s} err_T0_s".format(head)
        head = "{0:s} {1:s}_RChiSq_{2:s} {1:s}_lnL_{2:s} {1:s}_lnP_{2:s} {1:s}_BIC_{2:s} {1:s}_AIC_{2:s} {1:s}_RMS_{2:s}".format(
            head, "MED", "GP"
        )
        head = "{0:s} {1:s}_RChiSq_{2:s} {1:s}_lnL_{2:s} {1:s}_lnP_{2:s} {1:s}_BIC_{2:s} {1:s}_AIC_{2:s} {1:s}_RMS_{2:s}".format(
            head, "MLE", "GP"
        )
        head = "{0:s} err_T0_s_{1:s}".format(head, "GP")

        Tref = planet_args["T_ref"]
        P = planet_args["P_user_data"]
        bjd_ref = int(ascii_data["time"][0])
        epo = np.rint((bjd_ref - Tref.n) / P.n)
        line = "{:04.0f}".format(epo)

        s = s_lm["flux-all (w/o GP)"]
        line = "{:s} {:12.3f} {:12.3f} {:12.3f} {:12.3f} {:12.3f} {:12.3f} {:12.3f}".format(
            line,
            s["Red. ChiSqr"],
            s["lnL"],
            s["lnP"],
            s["BIC"],
            s["AIC"],
            s["lnL"],
            s["RMS (unbinned)"][0],
        )

        s = s_med["flux-all (w/o GP)"]
        line = "{:s} {:12.3f} {:12.3f} {:12.3f} {:12.3f} {:12.3f} {:12.3f} {:12.3f}".format(
            line,
            s["Red. ChiSqr"],
            s["lnL"],
            s["lnP"],
            s["BIC"],
            s["AIC"],
            s["lnL"],
            s["RMS (unbinned)"][0],
        )
        s = s_mle["flux-all (w/o GP)"]
        line = "{:s} {:12.3f} {:12.3f} {:12.3f} {:12.3f} {:12.3f} {:12.3f} {:12.3f}".format(
            line,
            s["Red. ChiSqr"],
            s["lnL"],
            s["lnP"],
            s["BIC"],
            s["AIC"],
            s["lnL"],
            s["RMS (unbinned)"][0],
        )
        line = "{:s} {:7.1f}".format(line, params["med"]["T_0"].stderr * cst.day2sec)

        s = s_med_gp["flux-all (w/ GP)"]
        line = "{:s} {:12.3f} {:12.3f} {:12.3f} {:12.3f} {:12.3f} {:12.3f} {:12.3f}".format(
            line,
            s["Red. ChiSqr"],
            s["lnL"],
            s["lnP"],
            s["BIC"],
            s["AIC"],
            s["lnL"],
            s["RMS (unbinned)"][0],
        )
        s = s_mle_gp["flux-all (w/ GP)"]
        line = "{:s} {:12.3f} {:12.3f} {:12.3f} {:12.3f} {:12.3f} {:12.3f} {:12.3f}".format(
            line,
            s["Red. ChiSqr"],
            s["lnL"],
            s["lnP"],
            s["BIC"],
            s["AIC"],
            s["lnL"],
            s["RMS (unbinned)"][0],
        )
        line = "{:s} {:7.1f}".format(line, params_gp["med"]["T_0"].stderr * cst.day2sec)
        printlog("", olog=olog)
        printlog("# === SUMMARY === #", olog=olog)
        printlog(head, olog=olog)
        printlog(line, olog=olog)
        printlog("", olog=olog)

        summary_file = os.path.join(main_folder, "quick_summary.dat")
        ofs = open(summary_file, "w")
        ofs.write("{:s}\n".format(head))
        ofs.write("{:s}\n".format(line))
        ofs.close()

        printlog("", olog=olog)
        printlog(" *********** ", olog=olog)
        printlog("--COMPLETED--", olog=olog)
        printlog(" *********** ", olog=olog)
        printlog("", olog=olog)

        olog.close()

        return


if __name__ == "__main__":
    sb = SingleBayes()
    sb.run_analysis()
