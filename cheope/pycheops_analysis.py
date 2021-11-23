#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
import matplotlib.pyplot as plt

# pycheops stuff
import pycheops
from pycheops import Dataset, StarProperties, MultiVisit
from pycheops.core import load_config
from pycheops.dataset import _kw_to_Parameter, _log_prior
from pycheops.funcs import massradius, rhostar
from pycheops.models import TransitModel, FactorModel
from pycheops.utils import lcbin
from pycheops.ld import h1h2_to_ca, h1h2_to_q1q2
from pycheops.ld import stagger_power2_interpolator
from pycheops.ld import atlas_h1h2_interpolator
from pycheops.ld import phoenix_h1h2_interpolator
from pycheops.instrument import CHEOPS_ORBIT_MINUTES

from lmfit import Parameter, Parameters
from lmfit.printfuncs import gformat
from uncertainties import ufloat, UFloat
import uncertainties.umath as um

import warnings
from scipy.interpolate import interp1d

import requests
from pathlib import Path
from os.path import getmtime
from time import localtime, mktime

from cryptography.fernet import Fernet

# import corner

try:
    import pickle5 as pickle
except:
    import pickle
import tarfile
import re

from astropy.table import Table
from astropy.timeseries import LombScargle
from astropy.convolution import convolve, Gaussian1DKernel
from astropy.time import Time
from astropy.coordinates import SkyCoord, get_body

# Temporary(?) fix for problem with coordinate look-up for HD stars in astropy
from astropy.coordinates.name_resolve import sesame_database

sesame_database.set("simbad")
print("pycheops version {}".format(pycheops.__version__))

# rcParams:
# matplotlib rc params
my_dpi = 192
plt.rcParams["text.usetex"] = False
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.serif"] = ["Computer Modern Roman", "Palatino", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["figure.figsize"] = [5, 5]
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"
plt.rcParams["figure.dpi"] = my_dpi
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 12
plt.rcParams["xtick.labelsize"] = plt.rcParams["font.size"] - 2
plt.rcParams["ytick.labelsize"] = plt.rcParams["xtick.labelsize"]
# plt.rcParams['animation.html']    = 'jshtml'

# other common packages
import numpy as np
import os
import sys
import time
from pathlib import Path
from astropy.time import Time

# from emcee import EnsembleSampler
import emcee
import corner

try:
    from celerite2 import terms
    from celerite2 import GaussianProcess as GP

    module_celerite2 = True
except:
    module_celerite2 = False
    from celerite import terms, GP

print("module_celerite2 = {}".format(module_celerite2))
if not module_celerite2:
    sys.exit()

# custom packages
import cheope.pyconstants as cst

# =====================================================================
# GLOBAL PARAMETERS
out_color = "lightgray"
# RMS
# 10min, 1hr, 3hr, 6hr
rms_time = [10.0 / 60.0, 1.0, 3.0, 6.0]

# common transit parameters
global_names = ["P", "D", "b", "W", "f_c", "f_s", "h_1", "h_2", "Tref"]
# per-transit parameters
transit_names = [
    "dT",
    "T_0",
    "c",
    "dfdt",
    "d2fdt2",
    "dfdbg",
    "dfdcontam",
    "dfdx",
    "d2fdx2",
    "dfdy",
    "d2fdy2",
    "dfdsinphi",
    "dfdcosphi",
    "dfdsin2phi",
    "dfdcos2phi",
    "dfdsin3phi",
    "dfdcos3phi",
    "glint_scale",
    "log_sigma",
    "log_S0",
    "log_omega0",  # Gaussian Process
]


# =======================
# global detrending setup
#
detrend_bounds = (-1, 1)
detrend_default = {
    "dfdbg": None,
    "dfdcontam": None,
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
    "dfdt": None,
    "d2fdt2": None,
    "glint_scale": None,
}
detrend = {}
# DEFAULT = 00
detrend["00"] = detrend_default.copy()

detrend["01a"] = detrend_default.copy()
detrend["01a"]["dfdt"] = detrend_bounds

detrend["01b"] = detrend["01a"].copy()
detrend["01b"]["d2fdt2"] = detrend_bounds

detrend["02"] = detrend_default.copy()
detrend["02"]["dfdbg"] = detrend_bounds

detrend["03"] = detrend_default.copy()
detrend["03"]["dfdcontam"] = detrend_bounds

detrend["04a"] = detrend_default.copy()
detrend["04a"]["dfdx"] = detrend_bounds
detrend["04a"]["dfdy"] = detrend_bounds

detrend["04b"] = detrend["04a"].copy()
detrend["04b"]["d2fdx2"] = detrend_bounds
detrend["04b"]["d2fdy2"] = detrend_bounds

detrend["05a"] = detrend_default.copy()
detrend["05a"]["dfdsinphi"] = detrend_bounds
detrend["05a"]["dfdcosphi"] = detrend_bounds

detrend["05b"] = detrend["05a"].copy()
detrend["05b"]["dfdsin2phi"] = detrend_bounds
detrend["05b"]["dfdcos2phi"] = detrend_bounds

detrend["05c"] = detrend["05b"].copy()
detrend["05c"]["dfdsin3phi"] = detrend_bounds
detrend["05c"]["dfdcos3phi"] = detrend_bounds

detrend["06"] = detrend_default.copy()
detrend["06"]["glint_scale"] = (0, 2)
# to use the glint must add the glint function with
# glint_func = dataset.add_glint()
# or ad-hoc
# dataset.planet_check()
# glint_func = dataset.add_glint(moon=True, nspline=21, binwidth=7)

# all the parameters, but the glint
# special care for d2fdt2 and d2fdx2, d2fdy2
detrend["07a"] = detrend_default.copy()
for k in detrend["07a"].keys():
    detrend["07a"][k] = detrend_bounds
detrend["07a"]["glint_scale"] = None

# all the parameters,
# special care for d2fdt2 and d2fdx2, d2fdy2
detrend["07b"] = detrend["07a"].copy()
detrend["07b"]["glint_scale"] = (0, 2)
# ======================================================================

params_units = {}
params_units["T_0"] = "BJD_TDB - "
params_units["P"] = "d"
params_units["D"] = "flux"
params_units["W"] = "day/P"
params_units["b"] = "-"
params_units["f_c"] = "-"
params_units["f_s"] = "-"
params_units["h_1"] = "-"
params_units["h_2"] = "-"
params_units["c"] = "flux"
params_units["k"] = "-"
params_units["aR"] = "-"
params_units["sini"] = "-"
params_units["logrho"] = "log(rho_sun)"
params_units["e"] = "-"
params_units["q_1"] = "-"
params_units["q_2"] = "-"
params_units["log_sigma"] = "-"
params_units["log_sigma_w"] = "-"
params_units["sigma_w"] = "flux"
params_units["inc"] = "deg"
params_units["LD_c"] = "-"
params_units["LD_alpha"] = "-"
params_units["log_S0"] = "-"
params_units["log_omega0"] = "-"
params_units["log_Q"] = "-"
params_units["det"] = "-"
params_units["t_exp"] = "d"
params_units["n_over"] = "-"


# ======================================================================
# ======================================================================
def printlog(l, olog=None):
    print(l)
    if olog is not None:
        olog.write("{}\n".format(l))
    return


# =====================================================================
# EXTRACT AND PLOT ALL THE LCs AT DIFFERENT APERTURE
def plot_all_apertures(dataset, out_folder):

    output_folder = Path(out_folder).resolve()

    # extract all the apertures and plot them all
    figraw = plt.figure()
    figraw.suptitle("{:s} raw".format(dataset.target), fontsize=8)
    axde_r = plt.subplot2grid((2, 2), (0, 0), fig=figraw)
    axde_r.set_ylabel("flux", fontsize=8)
    axop_r = plt.subplot2grid((2, 2), (0, 1), fig=figraw)
    axri_r = plt.subplot2grid((2, 2), (1, 0), fig=figraw)
    axri_r.set_ylabel("flux", fontsize=8)
    axrs_r = plt.subplot2grid((2, 2), (1, 1), fig=figraw)
    ax_r = [axde_r, axop_r, axri_r, axrs_r]

    figcli = plt.figure()
    figcli.suptitle("{:s} clipped".format(dataset.target), fontsize=8)
    axde_c = plt.subplot2grid((2, 2), (0, 0), fig=figcli)
    axde_c.set_ylabel("flux", fontsize=8)
    axop_c = plt.subplot2grid((2, 2), (0, 1), fig=figcli)
    axri_c = plt.subplot2grid((2, 2), (1, 0), fig=figcli)
    axri_c.set_ylabel("flux", fontsize=8)
    axrs_c = plt.subplot2grid((2, 2), (1, 1), fig=figcli)
    ax_c = [axde_c, axop_c, axri_c, axrs_c]

    all_ap = ["DEFAULT", "OPTIMAL", "RINF", "RSUP"]
    ix = [2, 3]

    for i, ap in enumerate(all_ap):
        print("APERTURE: {}".format(ap))
        # raw lc
        t, f, ef = dataset.get_lightcurve(
            aperture=ap, reject_highpoints=False, decontaminate=False
        )
        ax_r[i].set_title(
            "aperture {:s} ({:.0f}px)".format(ap, dataset.ap_rad), fontsize=6
        )
        ax_r[i].ticklabel_format(useOffset=False)
        ax_r[i].tick_params(labelsize=6)

        ax_r[i].errorbar(
            t,
            f,
            yerr=ef,
            color="C0",
            marker="o",
            ms=2,
            mec=out_color,
            mew=0.2,
            ls="",
            ecolor=out_color,
            elinewidth=0.4,
            capsize=0,
        )
        # clipped lc
        t, f, ef = dataset.clip_outliers(verbose=True)
        ax_c[i].set_title(
            "aperture {:s} ({:.0f}px)".format(ap, dataset.ap_rad), fontsize=6
        )
        ax_c[i].ticklabel_format(useOffset=False)
        ax_c[i].tick_params(labelsize=6)

        ax_c[i].errorbar(
            t,
            f,
            yerr=ef,
            color="C0",
            marker="o",
            ms=2,
            mec=out_color,
            mew=0.2,
            ls="",
            ecolor=out_color,
            elinewidth=0.4,
            capsize=0,
        )

        if i in ix:
            ax_r[i].set_xlabel(
                r"BJD$_\mathrm{{TDB}}-{:.0f}$".format(dataset.lc["bjd_ref"]), fontsize=8
            )
            ax_c[i].set_xlabel(
                r"BJD$_\mathrm{{TDB}}-{:.0f}$".format(dataset.lc["bjd_ref"]), fontsize=8
            )
        print()

    figraw.tight_layout()
    figcli.tight_layout()
    # plt.draw()

    figraw.savefig(
        output_folder.joinpath("00_lc_raw_all_apertures.png"), bbox_inches="tight"
    )
    figcli.savefig(
        output_folder.joinpath("00_lc_clipped_all_apertures.png"), bbox_inches="tight"
    )
    plt.close("all")

    return


# =====================================================================
# PLOT SINGLE EXTRACTED LC
def plot_single_lc(t, f, ef, bjd_ref):

    fig = plt.figure()
    plt.errorbar(
        t,
        f,
        yerr=ef,
        color="C0",
        marker="o",
        ms=2,
        mec=out_color,
        mew=0.5,
        ls="",
        ecolor=out_color,
        elinewidth=0.5,
        capsize=0,
    )
    plt.xlabel(r"BJD$_\mathrm{{TDB}} - {}$".format(bjd_ref))
    plt.ylabel("flux")
    plt.tight_layout()
    # plt.draw()

    return fig


# =====================================================================
# PLOT CUSTOM DIAGNOSTICS
def plot_custom_diagnostics(lc):

    nrows, ncols = 3, 2
    ms = 2
    mec = "None"
    lsize = 8
    ax = []
    fig = plt.figure()

    # contam vs roll angle
    ax.append(plt.subplot2grid((nrows, ncols), (0, 0)))
    ax[-1].tick_params(axis="both", labelsize=8)
    ax[-1].plot(
        lc["roll_angle"],
        lc["contam"] / np.median(lc["contam"]),
        color="C1",
        marker="o",
        ms=ms,
        mec=mec,
        ls="",
    )
    ax[-1].set_xlabel("roll angle", fontsize=lsize)
    ax[-1].set_ylabel("contam (/median)", fontsize=lsize)

    # roll angle vs time
    ax.append(plt.subplot2grid((nrows, ncols), (0, 1)))
    ax[-1].tick_params(axis="both", labelsize=8)
    ax[-1].plot(
        lc["time"], lc["roll_angle"], color="C1", marker="o", ms=ms, mec=mec, ls=""
    )
    ax[-1].set_xlabel("time", fontsize=lsize)
    ax[-1].set_ylabel("roll angle", fontsize=lsize)

    # bg vs roll angle
    ax.append(plt.subplot2grid((nrows, ncols), (1, 0)))
    ax[-1].tick_params(axis="both", labelsize=8)
    ax[-1].plot(
        lc["roll_angle"],
        lc["bg"] / np.median(lc["bg"]),
        color="C1",
        marker="o",
        ms=ms,
        mec=mec,
        ls="",
    )
    ax[-1].set_xlabel("roll angle", fontsize=lsize)
    ax[-1].set_ylabel("bg (/median)", fontsize=lsize)

    # bg vs time
    ax.append(plt.subplot2grid((nrows, ncols), (1, 1)))
    ax[-1].tick_params(axis="both", labelsize=8)
    ax[-1].plot(
        lc["time"],
        lc["bg"] / np.median(lc["bg"]),
        color="C1",
        marker="o",
        ms=ms,
        mec=mec,
        ls="",
    )
    ax[-1].set_xlabel("time", fontsize=lsize)
    ax[-1].set_ylabel("bg (/median)", fontsize=lsize)

    # contam vs bg
    ax.append(plt.subplot2grid((nrows, ncols), (2, 0)))
    ax[-1].tick_params(axis="both", labelsize=8)
    ax[-1].plot(
        lc["bg"] / np.median(lc["bg"]),
        lc["contam"] / np.median(lc["contam"]),
        color="C1",
        marker="o",
        ms=ms,
        mec=mec,
        ls="",
    )
    ax[-1].set_xlabel("bg (/median)", fontsize=lsize)
    ax[-1].set_ylabel("contam (/median)", fontsize=lsize)

    # contam vs time
    ax.append(plt.subplot2grid((nrows, ncols), (2, 1)))
    ax[-1].tick_params(axis="both", labelsize=8)
    ax[-1].plot(
        lc["time"],
        lc["contam"] / np.median(lc["contam"]),
        color="C1",
        marker="o",
        ms=ms,
        mec=mec,
        ls="",
    )
    ax[-1].set_xlabel("time", fontsize=lsize)
    ax[-1].set_ylabel("contam (/median)", fontsize=lsize)

    plt.tight_layout()
    # plt.draw()

    return fig


# =====================================================================
# PLOT DIAGNOSTICS AS CORNER PLOT
def plot_corner_diagnostics(dataset):

    lc = dataset.lc

    d = []
    l = []
    for k, v in lc.items():
        if isinstance(v, np.ndarray):
            l.append(k)
            d.append(v)

    # fig = corner.corner(d, bins=33,
    #   color='gray',
    #   labels=l
    # )

    color = "gray"
    hw = 1.0
    nv = len(l)
    fig = plt.figure(figsize=(nv * hw, nv * hw))
    ax = []
    for i in range(nv):
        iv = d[i]

        for j in range(nv - 1, -1, -1):
            jv = d[j]

            if j > i:
                # print('j ({}) > i ({})'.format(j, i))
                ax.append(plt.subplot2grid((nv, nv), (j, i)))

                ax[-1].plot(iv, jv, color=color, marker="o", ms=1, mec=None, ls="")
                ax[-1].xaxis.set_ticks([])
                ax[-1].set_xlabel(l[i], fontsize=5)
                ax[-1].yaxis.set_ticks([])
                ax[-1].set_ylabel(l[j], fontsize=5)

    plt.tight_layout()
    # plt.draw()

    return fig


# =====================================================================
# COMPUTE FULL MODEL FOR GIVEN DATASET AND PARAMETER SET
class DatasetModel:
    def __init__(self, n_time):
        self.n = n_time
        self.time = np.zeros((n_time))
        self.tra = np.zeros((n_time))
        self.trend = np.zeros((n_time))
        self.glint = np.zeros((n_time))
        self.gp = np.zeros((n_time))
        self.all_nogp = np.zeros((n_time))
        self.all = np.zeros((n_time))


def get_full_model(dataset, params_in=None, time=None):

    t_data = dataset.lc["time"]
    f_data = dataset.lc["flux"]
    ef_data = dataset.lc["flux_err"]

    # time vector
    if time is not None:
        t = time
    else:
        t = t_data
    # init model object
    n = len(t)
    m = DatasetModel(n)
    m.time = t.copy()

    # parameter set
    if params_in is None:
        try:
            params = dataset.emcee.params_best.copy()
        except:
            print("emcee params_best not found...using lmfit")
            try:
                params = dataset.lmfit.params_best.copy()
            except:
                print("lmfit params_best not found...exiting")
                return
    else:
        params = params_in.copy()

    # complete model
    m.all_nogp = dataset.model.eval(params, t=t)

    # transit, detrending, and glint model
    if "glint_scale" in params:
        m.tra = dataset.model.left.left.eval(params, t=t)
        m.trend = dataset.model.left.right.eval(params, t=t)
        m.glint = dataset.model.right.eval(params, t=t)
    else:
        m.tra = dataset.model.left.eval(params, t=t)
        m.trend = dataset.model.right.eval(params, t=t)

    # gaussian process model
    if "log_S0" in params:
        if module_celerite2:
            err2 = ef_data ** 2 + np.exp(params["log_sigma"].value) ** 2
            kernel = terms.SHOTerm(
                S0=np.exp(params["log_S0"].value),
                Q=1 / np.sqrt(2),
                w0=np.exp(params["log_omega0"].value),
            )
            gp = GP(kernel, mean=0)
            gp.compute(t_data, diag=err2, quiet=True)
        else:
            gp = dataset.gp
            gp.set_parameter("kernel:terms[0]:log_S0", params["log_S0"].value)
            gp.set_parameter("kernel:terms[0]:log_omega0", params["log_omega0"].value)
            gp.set_parameter("kernel:terms[1]:log_sigma", params["log_sigma"].value)

        # computes residuals with complete model on data, needed for gp res.
        res_data = f_data - dataset.model.eval(params, t=t_data)
        m.gp = gp.predict(res_data, t, return_cov=False, return_var=False)

    m.all = m.all_nogp + m.gp

    return m


# =====================================================================


def binned_rms(
    stats_dict, t_lc, residuals, rms_time_in, keyword="flux-all (w/o GP)", olog=None
):
    rms_unbin = np.std(residuals, ddof=1) * 1.0e6
    for binw in rms_time_in:
        if binw >= 1.0:
            binw_str = "{:1.0f}hr".format(binw)
        else:
            binw_str = "{:2.0f}min".format(binw * 60.0)
        nrms = 1
        try:
            _, _, e_bin, n_bin = lcbin(
                t_lc, residuals, binwidth=binw * cst.hour2day
            )  # returns: t_bin, f_bin, e_bin, n_bin
            rms_bin = e_bin * np.sqrt(n_bin - 1)
            nrms = len(rms_bin)
            if nrms > 1:
                rms = np.mean(rms_bin) * 1.0e6
                std = np.std(rms_bin, ddof=1) * 1.0e6 / np.sqrt(nrms - 1)
                printlog(
                    "RMS ({:8s}) = {:8.2f} +/- {:6.2f} (n_bin = {})".format(
                        binw_str, rms, std, nrms
                    ),
                    olog=olog,
                )
            else:
                rms, std = rms_unbin, 0.0
                printlog(
                    "RMS ({:8s}) = {:8.2f} +/- {:6.2f} (n_bin = {}): N_BIN < 2!".format(
                        binw_str, rms, std, nrms
                    ),
                    olog=olog,
                )
        except:
            rms, std, nrms = rms_unbin, 0.0, 1
            printlog(
                "RMS ({:8s}) = {:8.2f} +/- {:6.2f} (n_bin = {}): CANNOT BIN THE LC!".format(
                    binw_str, rms, std, nrms
                ),
                olog=olog,
            )
        stats_dict[keyword]["RMS ({:8s})".format(binw_str)] = [
            rms,
            std,
            nrms,
        ]
    return


# COMPUTE RMS FOR A SINGLE DATASET FOR GIVEN PARAMS
def computes_rms(dataset, params_best=None, glint=False, do_rms_bin=True, olog=None):

    lc = dataset.lc
    t, f, ef = lc["time"], lc["flux"], lc["flux_err"]

    if params_best is None:
        try:
            parbest = dataset.emcee.params_best
        except:
            print("emcee params_best not found...using lmfit")
            try:
                parbest = dataset.lmfit.params_best
            except:
                print("lmfit params_best not found...exiting")
                return
    else:
        parbest = params_best

    nfit = np.sum([parbest[p].vary for p in parbest])
    if "log_S0" in parbest and "log_omega0" in parbest:
        ngpfit = 3  # log_sigma in the kernel
    else:
        ngpfit = 0
    ndata = len(t)

    lnpr = 0.0
    for k in parbest:
        v = parbest[k].value
        u = parbest[k].user_data
        if isinstance(u, UFloat):
            lnpr += -0.5 * ((u.n - v) / u.s) ** 2
    lnpr += _log_prior(parbest["D"], parbest["W"], parbest["b"])

    try:
        j2 = np.exp(parbest["log_sigma"]) ** 2
    except:
        j2 = 0.0
    ef2 = ef ** 2
    err2 = ef2 + j2

    # fbest = dataset.model.eval(parbest, t=t)
    mfit = get_full_model(dataset, parbest)
    fbest = mfit.all_nogp

    statistics = {}

    printlog("", olog=olog)
    printlog("=================", olog=olog)
    printlog("flux-all (w/o GP)", olog=olog)

    res = f - fbest
    # stats
    chisquare = np.sum(res * res / ef2)
    chisquare_red = chisquare / (ndata - nfit)
    lnL = -0.5 * np.sum((res * res / err2) + np.log(2.0 * np.pi * err2))
    lnP = lnL + lnpr
    bic = -2.0 * lnL + (nfit - ngpfit) * np.log(ndata)
    aic = -2.0 * lnL + 2.0 * nfit
    printlog(
        "dof = ndata ({}) - nfit ({}) = {} (ngpfit = {})".format(
            ndata, nfit - ngpfit, ndata - nfit - ngpfit, ngpfit
        ),
        olog=olog,
    )

    statistics["flux-all (w/o GP)"] = {}
    statistics["flux-all (w/o GP)"]["ChiSqr"] = chisquare
    statistics["flux-all (w/o GP)"]["Red. ChiSqr"] = chisquare_red
    statistics["flux-all (w/o GP)"]["lnL"] = lnL
    statistics["flux-all (w/o GP)"]["lnP"] = lnP
    statistics["flux-all (w/o GP)"]["BIC"] = bic
    statistics["flux-all (w/o GP)"]["AIC"] = aic
    statistics["flux-all (w/o GP)"]["ndata"] = ndata
    statistics["flux-all (w/o GP)"]["nfit"] = nfit
    statistics["flux-all (w/o GP)"]["ngpfit"] = ngpfit
    statistics["flux-all (w/o GP)"]["dof"] = ndata - nfit - ngpfit

    printlog("ChiSqr         = {}".format(chisquare), olog=olog)
    printlog("Red. ChiSqr    = {}".format(chisquare_red), olog=olog)
    printlog("lnL            = {}".format(lnL), olog=olog)
    printlog("lnP            = {}".format(lnP), olog=olog)
    printlog("BIC            = {}".format(bic), olog=olog)
    printlog("AIC            = {}".format(aic), olog=olog)

    rms_unbin = np.std(res, ddof=1) * 1.0e6
    printlog("RMS ({:8s}) = {:8.2f}".format("unbinned", rms_unbin), olog=olog)
    statistics["flux-all (w/o GP)"]["RMS (unbinned)"] = [rms_unbin, 0.0, 1]

    if do_rms_bin:
        binned_rms(statistics, t, res, rms_time, keyword="flux-all (w/o GP)")
    printlog("", olog=olog)

    printlog("GP status: {}".format(dataset.gp), olog=olog)
    if dataset.gp is not None and dataset.gp is not False:
        printlog("=================", olog=olog)
        printlog("flux-all (w/ GP)", olog=olog)
        mu0 = mfit.gp
        # if(module_celerite2):
        #   kernel = terms.SHOTerm(
        #                 S0=np.exp(parbest['log_S0'].value),
        #                 Q=1/np.sqrt(2),
        #                 w0=np.exp(parbest['log_omega0'].value)
        #                 )
        #   gp = GP(kernel, mean=0)
        #   gp.compute(t, diag=err2, quiet=True)
        #   mu0 = gp.predict(res, t, return_cov=False, return_var=False)
        # else:
        #   dataset.gp.set_parameter('kernel:terms[0]:log_S0',     parbest['log_S0'])
        #   dataset.gp.set_parameter('kernel:terms[0]:log_omega0', parbest['log_omega0'])
        #   dataset.gp.set_parameter('kernel:terms[1]:log_sigma',  parbest['log_sigma'])
        #   # lnL = dataset.gp.log_likelihood(res)
        #   mu0 = dataset.gp.predict(res, t, return_cov=False, return_var=False)

        res = res - mu0
        # stats
        chisquare = np.sum(res * res / ef2)
        chisquare_red = chisquare / (ndata - nfit)
        # lnL          += lnp
        lnL = -0.5 * np.sum((res * res / err2) + np.log(2.0 * np.pi * err2))
        lnP = lnL + lnpr
        bic = -2.0 * lnL + nfit * np.log(ndata)
        aic = -2.0 * lnL + 2.0 * nfit
        printlog(
            "dof = ndata ({}) - nfit ({}) = {} (ngpfit = {})".format(
                ndata, nfit, ndata - nfit, ngpfit
            ),
            olog=olog,
        )

        statistics["flux-all (w/ GP)"] = {}
        statistics["flux-all (w/ GP)"]["ChiSqr"] = chisquare
        statistics["flux-all (w/ GP)"]["Red. ChiSqr"] = chisquare_red
        statistics["flux-all (w/ GP)"]["lnL"] = lnL
        statistics["flux-all (w/ GP)"]["lnP"] = lnP
        statistics["flux-all (w/ GP)"]["BIC"] = bic
        statistics["flux-all (w/ GP)"]["AIC"] = aic
        statistics["flux-all (w/ GP)"]["ndata"] = ndata
        statistics["flux-all (w/ GP)"]["nfit"] = nfit
        statistics["flux-all (w/ GP)"]["ngpfit"] = ngpfit
        statistics["flux-all (w/ GP)"]["dof"] = ndata - nfit - ngpfit

        printlog("ChiSqr         = {}".format(chisquare), olog=olog)
        printlog("Red. ChiSqr    = {}".format(chisquare_red), olog=olog)
        printlog("lnL            = {}".format(lnL), olog=olog)
        printlog("BIC            = {}".format(bic), olog=olog)
        printlog("AIC            = {}".format(aic), olog=olog)

        rms_unbin = np.std(res, ddof=1) * 1.0e6
        printlog("RMS ({:8s}) = {:8.2f}".format("unbinned", rms_unbin), olog=olog)
        statistics["flux-all (w/ GP)"]["RMS (unbinned)"] = [rms_unbin, 0.0, 1]
        if do_rms_bin:
            binned_rms(statistics, t, res, rms_time, keyword="flux-all (w/ GP)")
            # for binw in rms_time:
            #     _, _, e_bin, n_bin = lcbin(t, res, binwidth=binw * cst.hour2day)
            #     rms_bin = e_bin * np.sqrt(n_bin - 1)
            #     nrms = len(rms_bin)
            #     rms = np.mean(rms_bin) * 1.0e6
            #     std = np.std(rms_bin, ddof=1) * 1.0e6 / np.sqrt(nrms - 1)
            #     if binw >= 1.0:
            #         binw_str = "{:1.0f}hr".format(binw)
            #     else:
            #         binw_str = "{:2.0f}min".format(binw * 60.0)
            #     printlog(
            #         "RMS ({:8s}) = {:8.2f} +/- {:6.2f} (n_bin = {})".format(
            #             binw_str, rms, std, nrms
            #         ),
            #         olog=olog,
            #     )
            #     statistics["flux-all (w/ GP)"]["RMS ({:8s})".format(binw_str)] = [
            #         rms,
            #         std,
            #         nrms,
            #     ]
        printlog("", olog=olog)

    printlog("=================", olog=olog)
    printlog("flux-transit", olog=olog)
    # without trend/glint/etc
    # if(glint):
    #   ftra = dataset.model.left.left.eval(parbest, t=t)
    # else:
    #   ftra  = dataset.model.left.eval(parbest, t=t)
    ftra = mfit.tra
    res = f - ftra

    statistics["flux-transit"] = {}
    rms_unbin = np.std(res, ddof=1) * 1.0e6
    printlog("RMS ({:8s}) = {:8.2f}".format("unbinned", rms_unbin), olog=olog)
    statistics["flux-transit"]["RMS (unbinned)"] = [rms_unbin, 0.0, 1]
    if do_rms_bin:
        binned_rms(statistics, t, res, rms_time, keyword="flux-transit")
        # for binw in rms_time:
        #     _, _, e_bin, n_bin = lcbin(t, res, binwidth=binw * cst.hour2day)
        #     rms_bin = e_bin * np.sqrt(n_bin - 1)
        #     nrms = len(rms_bin)
        #     rms = np.mean(rms_bin) * 1.0e6
        #     std = np.std(rms_bin, ddof=1) * 1.0e6 / np.sqrt(nrms - 1)
        #     if binw >= 1.0:
        #         binw_str = "{:1.0f}hr".format(binw)
        #     else:
        #         binw_str = "{:2.0f}min".format(binw * 60.0)
        #     printlog(
        #         "RMS ({:8s}) = {:8.2f} +/- {:6.2f} (n_bin = {})".format(
        #             binw_str, rms, std, nrms
        #         ),
        #         olog=olog,
        #     )
        #     statistics["flux-transit"]["RMS ({:8s})".format(binw_str)] = [
        #         rms,
        #         std,
        #         nrms,
        #     ]
    printlog("", olog=olog)

    return statistics


def computes_rms_ultra(
    dataset, params_best=None, glint=False, do_rms_bin=True, olog=None
):

    lc = dataset.lc
    t, f, ef = lc["time"], lc["flux"], lc["flux_err"]

    if params_best is None:
        try:
            parbest = dataset.emcee.params_best
        except:
            print("emcee params_best not found...using lmfit")
            try:
                parbest = dataset.lmfit.params_best
            except:
                print("lmfit params_best not found...exiting")
                return
    else:
        parbest = params_best

    nfit = np.sum([parbest[p].vary for p in parbest])
    if "log_S0" in parbest and "log_omega0" in parbest:
        ngpfit = 3  # log_sigma in the kernel
    else:
        ngpfit = 0
    ndata = len(t)

    lnpr = 0.0
    for k in parbest:
        v = parbest[k].value
        u = parbest[k].user_data
        if isinstance(u, UFloat):
            lnpr += -0.5 * ((u.n - v) / u.s) ** 2
    lnpr += _log_prior(parbest["D"], parbest["W"], parbest["b"])

    try:
        j2 = np.exp(parbest["log_sigma"]) ** 2
    except:
        j2 = 0.0
    ef2 = ef ** 2
    err2 = ef2 + j2

    # fbest = dataset.model.eval(parbest, t=t)
    mfit = get_full_model(dataset, parbest)
    fbest = mfit.all_nogp

    statistics = {}

    printlog("", olog=olog)
    printlog("=================", olog=olog)
    printlog("flux-all (w/o GP)", olog=olog)

    res = f - fbest
    # stats
    chisquare = np.sum(res * res / ef2)
    chisquare_red = chisquare / (ndata - nfit)
    lnL = -0.5 * np.sum((res * res / err2) + np.log(2.0 * np.pi * err2))
    lnP = lnL + lnpr
    bic = -2.0 * lnL + (nfit - ngpfit) * np.log(ndata)
    aic = -2.0 * lnL + 2.0 * nfit
    printlog(
        "dof = ndata ({}) - nfit ({}) = {} (ngpfit = {})".format(
            ndata, nfit - ngpfit, ndata - nfit - ngpfit, ngpfit
        ),
        olog=olog,
    )

    statistics["flux-all (w/o GP)"] = {}
    statistics["flux-all (w/o GP)"]["ChiSqr"] = chisquare
    statistics["flux-all (w/o GP)"]["Red. ChiSqr"] = chisquare_red
    statistics["flux-all (w/o GP)"]["lnL"] = lnL
    statistics["flux-all (w/o GP)"]["lnP"] = lnP
    statistics["flux-all (w/o GP)"]["BIC"] = bic
    statistics["flux-all (w/o GP)"]["AIC"] = aic
    statistics["flux-all (w/o GP)"]["ndata"] = ndata
    statistics["flux-all (w/o GP)"]["nfit"] = nfit
    statistics["flux-all (w/o GP)"]["ngpfit"] = ngpfit
    statistics["flux-all (w/o GP)"]["dof"] = ndata - nfit - ngpfit

    printlog("ChiSqr         = {}".format(chisquare), olog=olog)
    printlog("Red. ChiSqr    = {}".format(chisquare_red), olog=olog)
    printlog("lnL            = {}".format(lnL), olog=olog)
    printlog("lnP            = {}".format(lnP), olog=olog)
    printlog("BIC            = {}".format(bic), olog=olog)
    printlog("AIC            = {}".format(aic), olog=olog)

    rms_unbin = np.std(res, ddof=1) * 1.0e6
    printlog("RMS ({:8s}) = {:8.2f}".format("unbinned", rms_unbin), olog=olog)
    statistics["flux-all (w/o GP)"]["RMS (unbinned)"] = [rms_unbin, 0.0, 1]

    if do_rms_bin:
        binned_rms(statistics, t, res, rms_time, keyword="flux-all (w/o GP)")
    printlog("", olog=olog)

    printlog("GP status: {}".format(dataset.gp), olog=olog)
    if dataset.gp is not None and dataset.gp is not False:
        printlog("=================", olog=olog)
        printlog("flux-all (w/ GP)", olog=olog)
        mu0 = mfit.gp
        # if(module_celerite2):
        #   kernel = terms.SHOTerm(
        #                 S0=np.exp(parbest['log_S0'].value),
        #                 Q=1/np.sqrt(2),
        #                 w0=np.exp(parbest['log_omega0'].value)
        #                 )
        #   gp = GP(kernel, mean=0)
        #   gp.compute(t, diag=err2, quiet=True)
        #   mu0 = gp.predict(res, t, return_cov=False, return_var=False)
        # else:
        #   dataset.gp.set_parameter('kernel:terms[0]:log_S0',     parbest['log_S0'])
        #   dataset.gp.set_parameter('kernel:terms[0]:log_omega0', parbest['log_omega0'])
        #   dataset.gp.set_parameter('kernel:terms[1]:log_sigma',  parbest['log_sigma'])
        #   # lnL = dataset.gp.log_likelihood(res)
        #   mu0 = dataset.gp.predict(res, t, return_cov=False, return_var=False)

        res = res - mu0
        # stats
        chisquare = np.sum(res * res / ef2)
        chisquare_red = chisquare / (ndata - nfit)
        # lnL          += lnp
        lnL = -0.5 * np.sum((res * res / err2) + np.log(2.0 * np.pi * err2))
        lnP = lnL + lnpr
        bic = -2.0 * lnL + nfit * np.log(ndata)
        aic = -2.0 * lnL + 2.0 * nfit
        printlog(
            "dof = ndata ({}) - nfit ({}) = {} (ngpfit = {})".format(
                ndata, nfit, ndata - nfit, ngpfit
            ),
            olog=olog,
        )

        statistics["flux-all (w/ GP)"] = {}
        statistics["flux-all (w/ GP)"]["ChiSqr"] = chisquare
        statistics["flux-all (w/ GP)"]["Red. ChiSqr"] = chisquare_red
        statistics["flux-all (w/ GP)"]["lnL"] = lnL
        statistics["flux-all (w/ GP)"]["lnP"] = lnP
        statistics["flux-all (w/ GP)"]["BIC"] = bic
        statistics["flux-all (w/ GP)"]["AIC"] = aic
        statistics["flux-all (w/ GP)"]["ndata"] = ndata
        statistics["flux-all (w/ GP)"]["nfit"] = nfit
        statistics["flux-all (w/ GP)"]["ngpfit"] = ngpfit
        statistics["flux-all (w/ GP)"]["dof"] = ndata - nfit - ngpfit

        printlog("ChiSqr         = {}".format(chisquare), olog=olog)
        printlog("Red. ChiSqr    = {}".format(chisquare_red), olog=olog)
        printlog("lnL            = {}".format(lnL), olog=olog)
        printlog("BIC            = {}".format(bic), olog=olog)
        printlog("AIC            = {}".format(aic), olog=olog)

        rms_unbin = np.std(res, ddof=1) * 1.0e6
        printlog("RMS ({:8s}) = {:8.2f}".format("unbinned", rms_unbin), olog=olog)
        statistics["flux-all (w/ GP)"]["RMS (unbinned)"] = [rms_unbin, 0.0, 1]
        if do_rms_bin:
            binned_rms(statistics, t, res, rms_time, keyword="flux-all (w/ GP)")
            # for binw in rms_time:
            #     _, _, e_bin, n_bin = lcbin(t, res, binwidth=binw * cst.hour2day)
            #     rms_bin = e_bin * np.sqrt(n_bin - 1)
            #     nrms = len(rms_bin)
            #     rms = np.mean(rms_bin) * 1.0e6
            #     std = np.std(rms_bin, ddof=1) * 1.0e6 / np.sqrt(nrms - 1)
            #     if binw >= 1.0:
            #         binw_str = "{:1.0f}hr".format(binw)
            #     else:
            #         binw_str = "{:2.0f}min".format(binw * 60.0)
            #     printlog(
            #         "RMS ({:8s}) = {:8.2f} +/- {:6.2f} (n_bin = {})".format(
            #             binw_str, rms, std, nrms
            #         ),
            #         olog=olog,
            #     )
            #     statistics["flux-all (w/ GP)"]["RMS ({:8s})".format(binw_str)] = [
            #         rms,
            #         std,
            #         nrms,
            #     ]
        printlog("", olog=olog)

    printlog("=================", olog=olog)
    printlog("flux-transit", olog=olog)
    # without trend/glint/etc
    # if(glint):
    #   ftra = dataset.model.left.left.eval(parbest, t=t)
    # else:
    #   ftra  = dataset.model.left.eval(parbest, t=t)
    ftra = mfit.tra
    res = f - ftra

    statistics["flux-transit"] = {}
    rms_unbin = np.std(res, ddof=1) * 1.0e6
    printlog("RMS ({:8s}) = {:8.2f}".format("unbinned", rms_unbin), olog=olog)
    statistics["flux-transit"]["RMS (unbinned)"] = [rms_unbin, 0.0, 1]
    if do_rms_bin:
        binned_rms(statistics, t, res, rms_time, keyword="flux-transit")
        # for binw in rms_time:
        #     _, _, e_bin, n_bin = lcbin(t, res, binwidth=binw * cst.hour2day)
        #     rms_bin = e_bin * np.sqrt(n_bin - 1)
        #     nrms = len(rms_bin)
        #     rms = np.mean(rms_bin) * 1.0e6
        #     std = np.std(rms_bin, ddof=1) * 1.0e6 / np.sqrt(nrms - 1)
        #     if binw >= 1.0:
        #         binw_str = "{:1.0f}hr".format(binw)
        #     else:
        #         binw_str = "{:2.0f}min".format(binw * 60.0)
        #     printlog(
        #         "RMS ({:8s}) = {:8.2f} +/- {:6.2f} (n_bin = {})".format(
        #             binw_str, rms, std, nrms
        #         ),
        #         olog=olog,
        #     )
        #     statistics["flux-transit"]["RMS ({:8s})".format(binw_str)] = [
        #         rms,
        #         std,
        #         nrms,
        #     ]
    printlog("", olog=olog)

    return statistics


# =====================================================================
# PLOT EACH RAW LC AND CLIP OUTLIERS
def plot_and_clip_lcs(datasets, apertures, out_folder, index_to_clip="all"):

    output_folder = Path(out_folder).resolve()

    if index_to_clip == "all":
        i_clip = [True] * len(datasets)
    else:
        i_clip = [True if i in index_to_clip else False for i in range(len(datasets))]

    for i, dataset in enumerate(datasets):

        lc = dataset.lc
        t, f, ef = lc["time"], lc["flux"], lc["flux_err"]

        title = r"{:s} - visit #{:d} - aperture {:s} ({:.0f}px)".format(
            dataset.target, i + 1, apertures[i], dataset.ap_rad
        )

        fig = plt.figure()

        plt.title(title, fontsize=8)
        plt.errorbar(
            t,
            f,
            yerr=ef,
            color="C0",
            marker="o",
            ms=2,
            mec=out_color,
            mew=0.5,
            ls="",
            ecolor=out_color,
            elinewidth=0.5,
            capsize=0,
        )
        plt.xlabel(r"BJD$_\mathrm{{TDB}} - {}$".format(lc["bjd_ref"]))
        plt.ylabel("flux")
        plt.tight_layout()
        # plt.draw()
        fig.savefig(
            output_folder.joinpath("00_lc_v{:02d}_raw.png".format(i + 1)),
            bbox_inches="tight",
        )
        fig.savefig(
            output_folder.joinpath("00_lc_v{:02d}_raw.pdf".format(i + 1)),
            bbox_inches="tight",
        )
        plt.close(fig)

        if i_clip[i]:
            # clip outliers
            print("\nClipping dataset index {} => {}".format(i, i_clip[i]))
            _ = dataset.clip_outliers(verbose=True)

        lc = dataset.lc
        t, f, ef = lc["time"], lc["flux"], lc["flux_err"]

        title = r"{:s} - visit #{:d} - aperture {:s} ({:.0f}px)".format(
            dataset.target, i + 1, apertures[i], dataset.ap_rad
        )

        fig = plt.figure()
        plt.title(title, fontsize=8)
        plt.errorbar(
            t,
            f,
            yerr=ef,
            color="C0",
            marker="o",
            ms=2,
            mec=out_color,
            mew=0.5,
            ls="",
            ecolor=out_color,
            elinewidth=0.5,
            capsize=0,
        )
        plt.xlabel(r"BJD$_\mathrm{{TDB}} - {}$".format(lc["bjd_ref"]))
        plt.ylabel("flux")
        plt.tight_layout()
        # plt.draw()
        fig.savefig(
            output_folder.joinpath("01_lc_v{:02d}_clipped_outliers.png".format(i + 1)),
            bbox_inches="tight",
        )
        fig.savefig(
            output_folder.joinpath("01_lc_v{:02d}_clipped_outliers.pdf".format(i + 1)),
            bbox_inches="tight",
        )
        plt.close(fig)

    return


# =====================================================================
# FULL COPY OF PARAMETERS SET FOR MULTIPLE-FIT
def copy_parameters(params_in):

    params_out = Parameters()
    for p in params_in:
        params_out[p] = Parameter(
            p,
            value=params_in[p].value,
            vary=params_in[p].vary,
            min=params_in[p].min,
            max=params_in[p].max,
            expr=params_in[p].expr,
            user_data=params_in[p].user_data,
            brute_step=None,  # 0
        )
        params_out[p].stderr = params_in[p].stderr

    return params_out


# =====================================================================
# GET FITTING PARAMETERS FROM PARAMETERS TYPE
def get_fitting_parameters(params, to_print=False):

    # fit parameters and names (as in pycheops)
    pfit = []  # fitting values
    pscale = []  # fitting scale
    pnames = []  # fitting names
    for p in params:
        if params[p].vary:
            pnames.append(p)
            pfit.append(params[p].value)
            if params[p].stderr is None:
                if params[p].user_data is None:
                    pscale.append(0.1 * (params[p].max - params[p].min))
                else:
                    pscale.append(params[p].user_data.s)
            else:
                if np.isfinite(params[p].stderr):
                    pscale.append(params[p].stderr)
                else:
                    pscale.append(0.1 * (params[p].max - params[p].min))

    if to_print:
        fmt_l = 12
        print("\nFitting parameters")
        print(
            "{:15s} {:>12s} {:>12s} {:>12s} {:>12s}".format(
                "name", "value", "scale", "min", "max"
            )
        )
        for i, n in enumerate(pnames):
            print(
                "{:15s} {:>12s} {:>12s} {:>12s} {:>12s}".format(
                    n,
                    gformat(pfit[i], length=fmt_l),
                    gformat(pscale[i], length=fmt_l),
                    gformat(params[n].min, length=fmt_l),
                    gformat(params[n].max, length=fmt_l),
                )
            )

    return pfit, pnames, pscale


# =====================================================================
# GET THE BEST PARAMETERS FROM EMCEE ANALYSIS WITH PYCHEOPS FOR ONE LC
def get_best_parameters(
    result, dataset, nburn=0, dataset_type="visit", update_dataset=False
):

    _, pnames, _ = get_fitting_parameters(result.params)
    #   print(pnames)

    if "multi" in dataset_type.lower():

        if nburn > 0:
            flatchain = dataset.sampler.get_chain(flat=True, discard=nburn)
            lnprob = dataset.sampler.get_log_prob(flat=True, discard=nburn)
        else:
            flatchain = dataset.sampler.get_chain(flat=True, discard=0)
            lnprob = dataset.sampler.get_log_prob(flat=True, discard=0)

        if update_dataset:
            print("Updating result.flatchain and result.lnprob")
            result.chain = flatchain.copy()
            result.lnprob = lnprob.copy()

        par_mle = result.parbest.copy()

    else:  # single visit dataset

        if nburn > 0:
            flatchain = dataset.sampler.get_chain(flat=True, discard=nburn)
            lnprob = dataset.sampler.get_log_prob(flat=True, discard=nburn)
            print("Updating result.flatchain and result.lnprob")
            result.chain = flatchain
            result.lnprob = lnprob
        else:
            flatchain = result.chain
            lnprob = result.lnprob

        par_mle = result.params_best.copy()

    pmed = np.percentile(flatchain, 50, interpolation="midpoint", axis=0)
    par_med = result.params.copy()

    idx_mle = np.argmax(lnprob)
    pmle = flatchain[idx_mle, :]

    # assign stderr = 0 if not fitted
    for p in par_med:
        if not par_med[p].vary:
            par_med[p].stderr = 0.0
            par_mle[p].stderr = 0.0

    for n in pnames:

        i_n = pnames.index(n)
        schain = flatchain[:, i_n]
        low, upp = high_posterior_density(schain)
        err = 0.5 * (upp - low)

        par_med[n].value = pmed[i_n]
        par_med[n].stderr = err

        par_mle[n].value = pmle[i_n]
        par_mle[n].stderr = err

        if "multi" in dataset_type.lower():
            # update also the dataset.result.params values!!
            dataset.result.params[n].value = pmed[i_n]
            dataset.result.params[n].stderr = err

            # update also the dataset.result.parbest values!!
            dataset.result.parbest[n].value = pmle[i_n]
            dataset.result.parbest[n].stderr = err

        else:
            # update also the dataset.emcee.params values!!
            dataset.emcee.params[n].value = pmed[i_n]
            dataset.emcee.params[n].stderr = err

            # update also the dataset.emcee.params_best values!!
            dataset.emcee.params_best[n].value = pmle[i_n]
            dataset.emcee.params_best[n].stderr = err

    # if(np.any([l1 in pnames for l1 in ['D', 'W', 'b']])):
    if "D" in pnames:
        D = flatchain[:, pnames.index("D")]
        k = np.sqrt(D)
        low, upp = high_posterior_density(k)
        err = 0.5 * (upp - low)
        par_med["k"].value = np.percentile(D, 50, interpolation="midpoint", axis=0)
        par_med["k"].stderr = err
        par_mle["k"].value = k[idx_mle]
        par_mle["k"].stderr = err
    else:
        k = par_med["k"].value

    if "W" in pnames:
        W = flatchain[:, pnames.index("W")]
    else:
        W = par_med["W"].value

    if "b" in pnames:
        b = flatchain[:, pnames.index("b")]
    else:
        b = par_med["b"].value

    aRs = np.sqrt((1 + k) ** 2 - b ** 2) / W / np.pi
    if np.any([l1 in pnames for l1 in ["D", "W", "b"]]):
        low, upp = high_posterior_density(aRs)
        err = 0.5 * (upp - low)
        aRs_med = np.percentile(aRs, 50, interpolation="midpoint", axis=0)
        aRs_mle = aRs[idx_mle]
    else:
        aRs_med = aRs
        aRs_mle = aRs
        err = 0.0
    par_med["aR"].value = aRs_med
    par_med["aR"].stderr = err
    par_mle["aR"].value = aRs_mle
    par_mle["aR"].stderr = err

    sini = np.sqrt(1 - (b / aRs) ** 2)
    if np.any([l1 in pnames for l1 in ["D", "W", "b"]]):
        low, upp = high_posterior_density(sini)
        err = 0.5 * (upp - low)
        sini_med = np.percentile(sini, 50, interpolation="midpoint", axis=0)
        sini_mle = sini[idx_mle]
    else:
        sini_med = sini
        sini_mle = sini
        err = 0.0
    par_med["sini"].value = sini_med
    par_med["sini"].stderr = err
    par_mle["sini"].value = sini_mle
    par_mle["sini"].stderr = err

    inc = np.arcsin(sini) * cst.rad2deg
    if np.any([l1 in pnames for l1 in ["D", "W", "b"]]):
        low, upp = high_posterior_density(inc)
        err = 0.5 * (upp - low)
        inc_med = np.percentile(inc, 50, interpolation="midpoint", axis=0)
        inc_mle = inc[idx_mle]
    else:
        inc_med = inc
        inc_mle = inc
        err = 0.0
    par_med["inc"] = Parameter("inc", value=inc_med, vary=False, min=0.0, max=180.0)
    par_med["inc"].stderr = err
    par_mle["inc"] = Parameter("inc", value=inc_mle, vary=False, min=0.0, max=180.0)
    par_mle["inc"].stderr = err

    if "log_sigma" in pnames:
        lsigma = flatchain[:, pnames.index("log_sigma")]
        sigmaw = np.exp(lsigma) * 1.0e6
        low, upp = high_posterior_density(sigmaw)
        err = 0.5 * (upp - low)
        par_med["sigma_w"].value = np.percentile(
            sigmaw, 50, interpolation="midpoint", axis=0
        )
        par_med["sigma_w"].stderr = err
        par_mle["sigma_w"].value = sigmaw[idx_mle]
        par_mle["sigma_w"].stderr = err

    if "f_c" in pnames or "f_s" in pnames:
        if "f_c" in pnames:
            f_c = flatchain[:, pnames.index("f_c")]
        else:
            f_c = par_med["f_c"]
        if "f_s" in pnames:
            f_s = flatchain[:, pnames.index("f_s")]
        else:
            f_s = par_med["f_s"]
        ecc = f_c ** 2 + f_s ** 2
        low, upp = high_posterior_density(ecc)
        err = 0.5 * (upp - low)
        par_med["e"].value = np.percentile(ecc, 50, interpolation="midpoint", axis=0)
        par_med["e"].stderr = err
        par_mle["e"].value = ecc[idx_mle]
        par_mle["e"].stderr = err

        w_1 = np.arctan2(f_s, f_c) * cst.rad2deg
        w_2 = w_1 % 360.0
        std1 = np.std(w_1, ddof=1)
        std2 = np.std(w_2, ddof=1)
        if std1 >= std2:
            w = w_2
        else:
            w = w_1
        low, upp = high_posterior_density(w)
        err = 0.5 * (upp - low)
        par_med["argp"] = Parameter(
            "argp",
            value=np.percentile(w, 50, interpolation="midpoint", axis=0),
            vary=False,
            min=0.0,
            max=360.0,
        )
        par_med["argp"].stderr = err
        par_mle["argp"] = Parameter(
            "argp", value=w[idx_mle], vary=False, min=0.0, max=180.0
        )
        par_mle["argp"].stderr = err

    if "h_1" in pnames or "h_2" in pnames:
        if "h_1" in pnames:
            h_1 = flatchain[:, pnames.index("h_1")]
        else:
            h_1 = par_med["h_1"].value
        if "h_2" in pnames:
            h_2 = flatchain[:, pnames.index("h_2")]
        else:
            h_2 = par_med["h_2"].value

        # c, alpha power-2 law
        c, alpha = h1h2_to_ca(h_1, h_2)

        low, upp = high_posterior_density(c)
        err = 0.5 * (upp - low)
        par_med["LD_c"] = Parameter(
            "LD_c",
            value=np.percentile(c, 50, interpolation="midpoint", axis=0),
            vary=False,
        )
        par_med["LD_c"].stderr = err
        par_mle["LD_c"] = Parameter("LD_c", value=c[idx_mle], vary=False)
        par_mle["LD_c"].stderr = err

        low, upp = high_posterior_density(alpha)
        err = 0.5 * (upp - low)
        par_med["LD_alpha"] = Parameter(
            "LD_alpha",
            value=np.percentile(alpha, 50, interpolation="midpoint", axis=0),
            vary=False,
        )
        par_med["LD_alpha"].stderr = err
        par_mle["LD_alpha"] = Parameter("LD_alpha", value=alpha[idx_mle], vary=False)
        par_mle["LD_alpha"].stderr = err

        # q1, q2
        q1, q2 = h1h2_to_q1q2(h_1, h_2)

        low, upp = high_posterior_density(q1)
        err = 0.5 * (upp - low)
        par_med["q_1"] = Parameter(
            "q_1",
            value=np.percentile(q1, 50, interpolation="midpoint", axis=0),
            vary=False,
        )
        par_med["q_1"].stderr = err
        par_mle["q_1"] = Parameter("q_1", value=q1[idx_mle], vary=False)
        par_mle["q_1"].stderr = err

        low, upp = high_posterior_density(q2)
        err = 0.5 * (upp - low)
        par_med["q_2"] = Parameter(
            "q_2",
            value=np.percentile(q2, 50, interpolation="midpoint", axis=0),
            vary=False,
        )
        par_med["q_2"].stderr = err
        par_mle["q_2"] = Parameter("q_2", value=q2[idx_mle], vary=False)
        par_mle["q_2"].stderr = err

    if "glint_scale" in par_med:
        glint = True
    else:
        glint = False

    print("PARAMS: MEDIAN")
    par_med.pretty_print(
        colwidth=20, precision=8, columns=["value", "stderr", "vary", "expr"]
    )
    if "multi" in dataset_type.lower():
        stats_med = None
    else:
        stats_med = computes_rms(dataset, params_best=par_med, glint=glint)
    print()
    print("PARAMS: BEST <-> MAX lnL")
    par_mle.pretty_print(
        colwidth=20, precision=8, columns=["value", "stderr", "vary", "expr"]
    )
    if "multi" in dataset_type.lower():
        stats_mle = None
    else:
        stats_mle = computes_rms(dataset, params_best=par_mle, glint=glint)

    return par_med, stats_med, par_mle, stats_mle


# =====================================================================
# GET THE BEST PARAMETERS FROM ULTRANEST ANALYSIS WITH PYCHEOPS FOR ONE LC
def get_best_parameters_ultranest(result, params, sampler, dataset_type="visit"):
    fitted_paramnames = sampler.paramnames

    par_med = params.copy()
    par_mle = params.copy()

    # GET THE MEDIAN PARAMETERS
    for name in fitted_paramnames:
        idx = result["paramnames"].index(name)
        new_value_med = result["posterior"]["median"][idx]
        new_value_mle = result["maximum_likelihood"]["point"][idx]
        errup = result["posterior"]["errup"][idx]
        errlo = result["posterior"]["errlo"][idx]
        stdev = result["posterior"]["stdev"][idx]
        par_med[name].set(value=new_value_med)
        par_mle[name].set(value=new_value_mle)
        par_med[name].stderr = stdev
        par_mle[name].stderr = stdev

        # print(par_med[name].stderr, par_med[name].value)
        # exit(0)

    if "D" in fitted_paramnames:
        idx = result["paramnames"].index("D")
        new_value_med = result["posterior"]["median"][idx]
        new_value_mle = result["maximum_likelihood"]["point"][idx]
        par_med["k"].value = Parameter(
            "k", value=new_value_med, vary=False, min=0.0, max=1.0, expr="sqrt(D)"
        )
        par_mle["k"].value = Parameter(
            "k", value=new_value_mle, vary=False, min=0.0, max=1.0, expr="sqrt(D)"
        )

    par_med["h_1"].set(value=params["h_1"].value)
    par_mle["h_2"].set(value=params["h_2"].value)

    inc = np.arcsin(params["sini"].value) * cst.rad2deg
    try:
        inc_med = inc
        inc_mle = inc
        err = np.arcsin(params["sini"].stderr) * cst.rad2deg
    except (AttributeError, TypeError):
        inc_med = inc
        inc_mle = inc
        err = 0.0

    par_med["inc"] = Parameter("inc", value=inc_med, vary=False, min=0.0, max=180.0)
    par_med["inc"].stderr = err
    par_mle["inc"] = Parameter("inc", value=inc_mle, vary=False, min=0.0, max=180.0)
    par_mle["inc"].stderr = err

    # assign stderr = 0 if not fitted
    for p in list(par_med.keys()):
        if not par_med[p].vary or par_med[p].stderr == None:
            par_med[p].stderr = 0.0
            par_mle[p].stderr = 0.0

    return par_med, par_mle


# def get_best_parameters_ultra(
#     result, dataset, dataset_type="visit", update_dataset=False
# ):

#     _, pnames, _ = get_fitting_parameters(result.params)
#   print(pnames)

# if "multi" in dataset_type.lower():

#     flatchain = dataset.sampler.get_chain(flat=True, discard=nburn)
#     lnprob = dataset.sampler.get_log_prob(flat=True, discard=0)

#     if update_dataset:
#         print("Updating result.flatchain and result.lnprob")
#         result.chain = flatchain.copy()
#         result.lnprob = lnprob.copy()

#     par_mle = result.parbest.copy()

# else:  # single visit dataset

#     if nburn > 0:
#         flatchain = dataset.sampler.get_chain(flat=True, discard=nburn)
#         lnprob = dataset.sampler.get_log_prob(flat=True, discard=nburn)
#         print("Updating result.flatchain and result.lnprob")
#         result.chain = flatchain
#         result.lnprob = lnprob
#     else:
#         flatchain = result.chain
#         lnprob = result.lnprob

#     par_mle = result.params_best.copy()

# pmed = np.percentile(flatchain, 50, interpolation="midpoint", axis=0)
# par_med = result.params.copy()

# idx_mle = np.argmax(lnprob)
# pmle = flatchain[idx_mle, :]

# # assign stderr = 0 if not fitted
# for p in par_med:
#     if not par_med[p].vary:
#         par_med[p].stderr = 0.0
#         par_mle[p].stderr = 0.0

# for n in pnames:

#     i_n = pnames.index(n)
#     schain = flatchain[:, i_n]
#     low, upp = high_posterior_density(schain)
#     err = 0.5 * (upp - low)

#     par_med[n].value = pmed[i_n]
#     par_med[n].stderr = err

#     par_mle[n].value = pmle[i_n]
#     par_mle[n].stderr = err

#     if "multi" in dataset_type.lower():
#         # update also the dataset.result.params values!!
#         dataset.result.params[n].value = pmed[i_n]
#         dataset.result.params[n].stderr = err

#         # update also the dataset.result.parbest values!!
#         dataset.result.parbest[n].value = pmle[i_n]
#         dataset.result.parbest[n].stderr = err

#     else:
#         # update also the dataset.emcee.params values!!
#         dataset.emcee.params[n].value = pmed[i_n]
#         dataset.emcee.params[n].stderr = err

#         # update also the dataset.emcee.params_best values!!
#         dataset.emcee.params_best[n].value = pmle[i_n]
#         dataset.emcee.params_best[n].stderr = err

# # if(np.any([l1 in pnames for l1 in ['D', 'W', 'b']])):
# if "D" in pnames:
#     D = flatchain[:, pnames.index("D")]
#     k = np.sqrt(D)
#     low, upp = high_posterior_density(k)
#     err = 0.5 * (upp - low)
#     par_med["k"].value = np.percentile(D, 50, interpolation="midpoint", axis=0)
#     par_med["k"].stderr = err
#     par_mle["k"].value = k[idx_mle]
#     par_mle["k"].stderr = err
# else:
#     k = par_med["k"].value

# if "W" in pnames:
#     W = flatchain[:, pnames.index("W")]
# else:
#     W = par_med["W"].value

# if "b" in pnames:
#     b = flatchain[:, pnames.index("b")]
# else:
#     b = par_med["b"].value

# aRs = np.sqrt((1 + k) ** 2 - b ** 2) / W / np.pi
# if np.any([l1 in pnames for l1 in ["D", "W", "b"]]):
#     low, upp = high_posterior_density(aRs)
#     err = 0.5 * (upp - low)
#     aRs_med = np.percentile(aRs, 50, interpolation="midpoint", axis=0)
#     aRs_mle = aRs[idx_mle]
# else:
#     aRs_med = aRs
#     aRs_mle = aRs
#     err = 0.0
# par_med["aR"].value = aRs_med
# par_med["aR"].stderr = err
# par_mle["aR"].value = aRs_mle
# par_mle["aR"].stderr = err

# sini = np.sqrt(1 - (b / aRs) ** 2)
# if np.any([l1 in pnames for l1 in ["D", "W", "b"]]):
#     low, upp = high_posterior_density(sini)
#     err = 0.5 * (upp - low)
#     sini_med = np.percentile(sini, 50, interpolation="midpoint", axis=0)
#     sini_mle = sini[idx_mle]
# else:
#     sini_med = sini
#     sini_mle = sini
#     err = 0.0
# par_med["sini"].value = sini_med
# par_med["sini"].stderr = err
# par_mle["sini"].value = sini_mle
# par_mle["sini"].stderr = err

# inc = np.arcsin(sini) * cst.rad2deg
# if np.any([l1 in pnames for l1 in ["D", "W", "b"]]):
#     low, upp = high_posterior_density(inc)
#     err = 0.5 * (upp - low)
#     inc_med = np.percentile(inc, 50, interpolation="midpoint", axis=0)
#     inc_mle = inc[idx_mle]
# else:
#     inc_med = inc
#     inc_mle = inc
#     err = 0.0
# par_med["inc"] = Parameter("inc", value=inc_med, vary=False, min=0.0, max=180.0)
# par_med["inc"].stderr = err
# par_mle["inc"] = Parameter("inc", value=inc_mle, vary=False, min=0.0, max=180.0)
# par_mle["inc"].stderr = err

# if "log_sigma" in pnames:
#     lsigma = flatchain[:, pnames.index("log_sigma")]
#     sigmaw = np.exp(lsigma) * 1.0e6
#     low, upp = high_posterior_density(sigmaw)
#     err = 0.5 * (upp - low)
#     par_med["sigma_w"].value = np.percentile(
#         sigmaw, 50, interpolation="midpoint", axis=0
#     )
#     par_med["sigma_w"].stderr = err
#     par_mle["sigma_w"].value = sigmaw[idx_mle]
#     par_mle["sigma_w"].stderr = err

# if "f_c" in pnames or "f_s" in pnames:
#     if "f_c" in pnames:
#         f_c = flatchain[:, pnames.index("f_c")]
#     else:
#         f_c = par_med["f_c"]
#     if "f_s" in pnames:
#         f_s = flatchain[:, pnames.index("f_s")]
#     else:
#         f_s = par_med["f_s"]
#     ecc = f_c ** 2 + f_s ** 2
#     low, upp = high_posterior_density(ecc)
#     err = 0.5 * (upp - low)
#     par_med["e"].value = np.percentile(ecc, 50, interpolation="midpoint", axis=0)
#     par_med["e"].stderr = err
#     par_mle["e"].value = ecc[idx_mle]
#     par_mle["e"].stderr = err

#     w_1 = np.arctan2(f_s, f_c) * cst.rad2deg
#     w_2 = w_1 % 360.0
#     std1 = np.std(w_1, ddof=1)
#     std2 = np.std(w_2, ddof=1)
#     if std1 >= std2:
#         w = w_2
#     else:
#         w = w_1
#     low, upp = high_posterior_density(w)
#     err = 0.5 * (upp - low)
#     par_med["argp"] = Parameter(
#         "argp",
#         value=np.percentile(w, 50, interpolation="midpoint", axis=0),
#         vary=False,
#         min=0.0,
#         max=360.0,
#     )
#     par_med["argp"].stderr = err
#     par_mle["argp"] = Parameter(
#         "argp", value=w[idx_mle], vary=False, min=0.0, max=180.0
#     )
#     par_mle["argp"].stderr = err

# if "h_1" in pnames or "h_2" in pnames:
#     if "h_1" in pnames:
#         h_1 = flatchain[:, pnames.index("h_1")]
#     else:
#         h_1 = par_med["h_1"].value
#     if "h_2" in pnames:
#         h_2 = flatchain[:, pnames.index("h_2")]
#     else:
#         h_2 = par_med["h_2"].value

#     # c, alpha power-2 law
#     c, alpha = h1h2_to_ca(h_1, h_2)

#     low, upp = high_posterior_density(c)
#     err = 0.5 * (upp - low)
#     par_med["LD_c"] = Parameter(
#         "LD_c",
#         value=np.percentile(c, 50, interpolation="midpoint", axis=0),
#         vary=False,
#     )
#     par_med["LD_c"].stderr = err
#     par_mle["LD_c"] = Parameter("LD_c", value=c[idx_mle], vary=False)
#     par_mle["LD_c"].stderr = err

#     low, upp = high_posterior_density(alpha)
#     err = 0.5 * (upp - low)
#     par_med["LD_alpha"] = Parameter(
#         "LD_alpha",
#         value=np.percentile(alpha, 50, interpolation="midpoint", axis=0),
#         vary=False,
#     )
#     par_med["LD_alpha"].stderr = err
#     par_mle["LD_alpha"] = Parameter("LD_alpha", value=alpha[idx_mle], vary=False)
#     par_mle["LD_alpha"].stderr = err

#     # q1, q2
#     q1, q2 = h1h2_to_q1q2(h_1, h_2)

#     low, upp = high_posterior_density(q1)
#     err = 0.5 * (upp - low)
#     par_med["q_1"] = Parameter(
#         "q_1",
#         value=np.percentile(q1, 50, interpolation="midpoint", axis=0),
#         vary=False,
#     )
#     par_med["q_1"].stderr = err
#     par_mle["q_1"] = Parameter("q_1", value=q1[idx_mle], vary=False)
#     par_mle["q_1"].stderr = err

#     low, upp = high_posterior_density(q2)
#     err = 0.5 * (upp - low)
#     par_med["q_2"] = Parameter(
#         "q_2",
#         value=np.percentile(q2, 50, interpolation="midpoint", axis=0),
#         vary=False,
#     )
#     par_med["q_2"].stderr = err
#     par_mle["q_2"] = Parameter("q_2", value=q2[idx_mle], vary=False)
#     par_mle["q_2"].stderr = err

# if "glint_scale" in par_med:
#     glint = True
# else:
#     glint = False

# print("PARAMS: MEDIAN")
# par_med.pretty_print(
#     colwidth=20, precision=8, columns=["value", "stderr", "vary", "expr"]
# )
# if "multi" in dataset_type.lower():
#     stats_med = None
# else:
#     stats_med = computes_rms(dataset, params_best=par_med, glint=glint)
# print()
# print("PARAMS: BEST <-> MAX lnL")
# par_mle.pretty_print(
#     colwidth=20, precision=8, columns=["value", "stderr", "vary", "expr"]
# )
# if "multi" in dataset_type.lower():
#     stats_mle = None
# else:
#     stats_mle = computes_rms(dataset, params_best=par_mle, glint=glint)

# return par_med, stats_med, par_mle, stats_mle


# =====================================================================
# FROM PARAMETER SET COMPUTES MODEL AND PLOT EVERITHING IN ONE FIG
def model_plot_fit(
    dataset,
    par_fit,
    par_type="median",
    nsamples=0,
    flatchains=None,
    model_filename=None,
):

    # figure
    xt_font = plt.rcParams["xtick.labelsize"]
    # xt_fond = 6
    # lt_font = 8

    # zorder
    zo_d = 6  # data
    zo_a = 10  # all
    zo_m = 9  # model transit
    zo_s = 8  # samples
    # zo_t = 7 # trend
    # lweight
    lw_m = 1.0  # model transit
    lw_a = 1.0  # all
    # lw_t = 0.9 # trend
    # marker size
    ms_d = 3  # data

    nrows = 5
    ncols = 1

    lc = dataset.lc
    t_data, f_data, ef_data = lc["time"], lc["flux"], lc["flux_err"]

    # oversampled
    n_data = len(t_data)
    texp = dataset.exptime * cst.sec2day
    t_fill = np.arange(t_data[0], t_data[-1] + 0.5 * texp, texp)
    n_fill = len(t_fill)

    # lc sampling
    # f_all_fit  = dataset.model.eval(par_fit, t=t_data) # without gp
    # res = f_data - f_fit
    mfit = get_full_model(dataset, params_in=par_fit, time=t_data)
    res_nogp = f_data - mfit.all_nogp
    res = f_data - mfit.all

    # print()
    # print('res_nogp min = {} max = {}'.format(np.min(res_nogp), np.max(res_nogp)))
    # print('fit gp   min = {} max = {}'.format(np.min(mfit.gp), np.max(mfit.gp)))
    # print('res      min = {} max = {}'.format(np.min(res), np.max(res)))
    # print()

    # oversampled
    # f_all_fill = dataset.model.eval(par_fit, t=t_fill) # without gp
    mfill = get_full_model(dataset, params_in=par_fit, time=t_fill)

    # yl_det = 'flux/trend'
    det_den = r"\mathrm{trend}"
    det_num = r"\mathrm{flux}"

    if "glint_scale" in par_fit:
        det_num += r" - \mathrm{glint}"

    if "log_S0" in par_fit:
        nrows += 1
        det_num += r" - \mathrm{gp}"

    yl_det = r"$\frac{{{:s}}}{{{:s}}}$".format(det_num, det_den)
    # ff_det = f/ff_trend - ff_glint - ff_gp
    f_det_fit = (f_data - mfit.gp - mfit.glint) / mfit.trend

    model = {}
    model["time"] = t_data
    model["flux"] = f_data
    model["flux_err"] = ef_data
    model["flux_all"] = mfit.all
    model["flux_transit"] = mfit.tra
    model["flux_trend"] = mfit.trend
    model["flux_glint"] = mfit.glint
    model["flux_gp"] = mfit.gp
    model["residuals"] = res

    if model_filename is not None:
        if os.path.isfile(model_filename):
            print("WARNING overwriting file model {}".format(model_filename))
        out = np.column_stack(
            (
                t_data,
                f_data,
                ef_data,
                mfit.all,
                mfit.tra,
                mfit.trend,
                mfit.glint,
                mfit.gp,
                res,
            )
        )

        heado = " all flux are normalized flux\n"
        heado = "{} time = BJD_TDB - {}\n".format(heado, lc["bjd_ref"])
        heado = (
            "{} flux_all = (flux_transit + flux_glint + flux_gp)*flux_trend\n".format(
                heado
            )
        )
        heado = "{} flux_detrended = (flux - flux_glint - flux_gp)/flux_trend\n".format(
            heado
        )
        heado += " ".join(k for k in model.keys())
        fmto = "%23.16e"
        np.savetxt(model_filename, out, header=heado, fmt=fmto)

    title = r"{:s} - aperture {:s} ({:.0f}px) - {:s}".format(
        dataset.target, lc["aperture"], dataset.ap_rad, par_type.upper()
    )

    fig = plt.figure()
    axs = []

    # -------------------
    # FULL MODEL W/ TREND
    i_row = 0
    axf = plt.subplot2grid((nrows, ncols), (i_row, 0), rowspan=2, colspan=1)
    axs.append(axf)
    axf.ticklabel_format(useOffset=False)
    axf.tick_params(labelsize=xt_font, labelbottom=False)
    axf.set_title(title, fontsize=8)
    axf.set_ylabel("flux")
    # data
    axf.errorbar(
        t_data,
        f_data,
        yerr=ef_data,
        color="C0",
        marker="o",
        ms=ms_d,
        mec=out_color,
        mew=0.5,
        ls="",
        ecolor=out_color,
        elinewidth=0.5,
        capsize=0,
        zorder=zo_d,
    )

    # transit model
    # axf.plot(t_fill, (f_all_fill - f_gp_fill - f_glint_fill)/f_trend_fill,
    axf.plot(
        t_fill, mfill.tra, color="black", marker="None", ls="-", lw=lw_m, zorder=zo_m
    )

    # transit + trend + glint + gp model
    # axf.plot(t_fill, f_all_fill,
    axf.plot(
        t_fill,
        mfill.all,  # mfill.all_nogp,
        color="C1",
        marker="None",
        ls="-",
        lw=lw_a,
        zorder=zo_a,
    )

    # -------------------
    # DETRENDED MODEL W/O TREND - GLINT - gp
    i_row += 2
    axd = plt.subplot2grid((nrows, ncols), (i_row, 0), rowspan=2, colspan=1)
    axs.append(axd)
    axd.ticklabel_format(useOffset=False)
    axd.tick_params(labelsize=xt_font, labelbottom=False)

    # data
    axd.errorbar(
        t_data,
        f_det_fit,
        yerr=ef_data,
        color="C0",
        marker="o",
        ms=ms_d,
        mec=out_color,
        mew=0.5,
        ls="",
        ecolor=out_color,
        elinewidth=0.5,
        capsize=0,
        zorder=zo_d,
    )
    # transit model
    # axd.plot(t_fill, f_tra_fill,
    axd.plot(
        t_fill, mfill.tra, color="black", marker="None", ls="-", lw=lw_m, zorder=zo_m
    )
    axd.set_ylabel(yl_det, fontsize=plt.rcParams["font.size"] + 4)

    i_row += 2
    if nrows == 6:
        # -------------------
        # RESIDUALS WITH gp
        axgp = plt.subplot2grid((nrows, ncols), (i_row, 0), rowspan=1, colspan=1)
        axs.append(axgp)
        axgp.ticklabel_format(useOffset=False)
        axgp.tick_params(labelsize=xt_font, labelbottom=False)

        i_row += 1
        # RES_GP = DATA - GLINT - TRAN*TREND
        axgp.errorbar(
            t_data,
            res_nogp,
            yerr=ef_data,
            color="C0",
            marker="o",
            ms=ms_d,
            mec=out_color,
            mew=0.5,
            ls="",
            ecolor=out_color,
            elinewidth=0.5,
            capsize=0,
            zorder=zo_d,
        )
        axgp.axhline(0.0, color="black", ls="-", lw=0.5, zorder=5)

        # GP MODEL
        axgp.plot(
            t_fill, mfill.gp, color="C3", marker="None", ls="-", lw=lw_m, zorder=zo_m
        )
        axgp.set_ylabel("res. (no gp)")

    # -------------------
    # RESIDUALS
    axr = plt.subplot2grid((nrows, ncols), (i_row, 0), rowspan=1, colspan=1)
    axs.append(axr)
    axr.ticklabel_format(useOffset=False)
    axr.tick_params(labelsize=xt_font, labelbottom=True)

    if "log_sigma" in par_fit:
        ef_sc = np.sqrt(ef_data ** 2 + np.exp(par_fit["log_sigma"]) ** 2)
    else:
        ef_sc = ef_data
    axr.errorbar(
        t_data,
        res,
        yerr=ef_sc,
        color="C0",
        marker="o",
        ms=ms_d,
        mec=out_color,
        mew=0.5,
        ls="",
        ecolor=out_color,
        elinewidth=0.5,
        capsize=0,
        zorder=zo_d,
    )
    axr.axhline(0.0, color="black", ls="-", lw=0.5, zorder=5)
    axr.set_ylabel("res.")
    axr.set_xlabel(r"BJD$_\mathrm{{TDB}} - {}$".format(lc["bjd_ref"]))

    if nsamples > 0 and flatchains is not None:
        _, pnames, _ = get_fitting_parameters(par_fit)
        npost, nfit = np.shape(flatchains)
        selchains = np.ones((npost))
        for n in pnames:
            i_fit = pnames.index(n)
            # cc = 0.999999426696856
            # h = (0.5*cc)*100
            # pl, pu = 50 - h, 50 + h
            # low, upp = np.percentile(flatchains[:,i_fit], [pl, pu], interpolation='midpoint')
            cc = 0.6827
            cmed = np.percentile(flatchains[:, i_fit], 50, interpolation="midpoint")
            rms = np.percentile(
                np.abs(flatchains[:, i_fit] - cmed), cc * 100, interpolation="midpoint"
            )
            k = 5
            low, upp = cmed - k * rms, cmed + k * rms
            withinhpd = np.logical_and(
                flatchains[:, i_fit] >= low, flatchains[:, i_fit] <= upp
            )
            selchains = np.logical_and(selchains, withinhpd)
        goodchains = flatchains[selchains, :]
        ngood, _ = np.shape(goodchains)

        smp_idx = np.random.choice(ngood, size=nsamples, replace=False)
        al_s = max(1.0 / nsamples, 0.05)
        for idx in smp_idx:
            # psmp = flatchains[idx,:]
            psmp = goodchains[idx, :]
            par_smp = par_fit.copy()
            for n in pnames:
                par_smp[n].value = psmp[pnames.index(n)]

            # sfit  = get_full_model(dataset, params_in=par_smp, time=t_data) # lc sampling
            sfill = get_full_model(
                dataset, params_in=par_smp, time=t_fill
            )  # oversampled

            # transit*trend + glint + gp model
            axf.plot(
                t_fill,
                sfill.all,
                color="C2",
                marker="None",
                ls="-",
                lw=lw_a,
                alpha=al_s,
                zorder=zo_s,
            )
            # transit (w/o trend, glint, gp)
            # s_tra = (s_all - s_gp - s_glint) / s_trend
            #       axd.plot(tt, s_tra,
            # axd.plot(tt, (s_all - s_gp - s_glint) / s_trend,
            axd.plot(
                t_fill,
                sfill.tra,
                color="C2",
                marker="None",
                ls="-",
                lw=lw_a,
                alpha=al_s,
                zorder=zo_s,
            )

    fig.align_ylabels(axs)
    plt.tight_layout()
    # plt.draw()

    return fig, model


# =====================================================================
# INIT A MODEL FOR EACH LC -> FIT WITH LM SINGLE LC
def init_multi_model(datasets, params, pnames, gnames, tnames, do_fit=True):

    print("\nInit pycheops multi model")
    tpars = {}
    gps = []
    params_fit = copy_parameters(params)
    single_pars = []
    for i, dataset in enumerate(datasets):
        v = "v{:d}".format(i + 1)
        print("\n{}".format(v))
        for n in tnames:
            k = "{:s}_{:s}".format(n, v)
            tpars[k] = params[k] if k in pnames else None
        lmfit = dataset.lmfit_transit(
            # global
            P=params["P"],
            D=params["D"],
            W=params["W"],
            b=params["b"],
            f_c=params["f_c"],
            f_s=params["f_s"],
            h_1=params["h_1"],
            h_2=params["h_2"],
            # per-transit
            T_0=tpars["T_0_{:s}".format(v)],
            c=tpars["c_{:s}".format(v)],
            dfdt=tpars["dfdt_{:s}".format(v)],
            d2fdt2=tpars["d2fdt2_{:s}".format(v)],
            dfdbg=tpars["dfdbg_{:s}".format(v)],
            dfdcontam=tpars["dfdcontam_{:s}".format(v)],
            dfdx=tpars["dfdx_{:s}".format(v)],
            dfdy=tpars["dfdy_{:s}".format(v)],
            d2fdx2=tpars["d2fdx2_{:s}".format(v)],
            d2fdy2=tpars["d2fdy2_{:s}".format(v)],
            dfdsinphi=tpars["dfdsinphi_{:s}".format(v)],
            dfdcosphi=tpars["dfdcosphi_{:s}".format(v)],
            dfdsin2phi=tpars["dfdsin2phi_{:s}".format(v)],
            dfdcos2phi=tpars["dfdcos2phi_{:s}".format(v)],
            dfdsin3phi=tpars["dfdsin3phi_{:s}".format(v)],
            dfdcos3phi=tpars["dfdcos3phi_{:s}".format(v)],
            logrhoprior=params["logrho"],
        )  # it set the proper dataset.model to use for the evaluations
        if tpars["glint_scale_{:s}".format(v)] is not None:
            dataset.add_glint(nspline=42, binwidth=7)
            # plt.draw()
            plt.close()
            lmfit = dataset.lmfit_transit(
                # global
                P=params["P"],
                D=params["D"],
                W=params["W"],
                b=params["b"],
                f_c=params["f_c"],
                f_s=params["f_s"],
                h_1=params["h_1"],
                h_2=params["h_2"],
                # per-transit
                T_0=tpars["T_0_{:s}".format(v)],
                c=tpars["c_{:s}".format(v)],
                dfdt=tpars["dfdt_{:s}".format(v)],
                d2fdt2=tpars["d2fdt2_{:s}".format(v)],
                dfdbg=tpars["dfdbg_{:s}".format(v)],
                dfdcontam=tpars["dfdcontam_{:s}".format(v)],
                dfdx=tpars["dfdx_{:s}".format(v)],
                dfdy=tpars["dfdy_{:s}".format(v)],
                d2fdx2=tpars["d2fdx2_{:s}".format(v)],
                d2fdy2=tpars["d2fdy2_{:s}".format(v)],
                dfdsinphi=tpars["dfdsinphi_{:s}".format(v)],
                dfdcosphi=tpars["dfdcosphi_{:s}".format(v)],
                dfdsin2phi=tpars["dfdsin2phi_{:s}".format(v)],
                dfdcos2phi=tpars["dfdcos2phi_{:s}".format(v)],
                dfdsin3phi=tpars["dfdsin3phi_{:s}".format(v)],
                dfdcos3phi=tpars["dfdcos3phi_{:s}".format(v)],
                glint_scale=tpars["glint_scale_{:s}".format(v)],
                logrhoprior=params["logrho"],
            )
        # Gaussian process
        ks = "log_S0_{:s}".format(v)
        ko = "log_omega0_{:s}".format(v)
        kw = "log_sigma_{:s}".format(v)
        if (
            (tpars[ks] is not None)
            and (tpars[ko] is not None)
            and (tpars[kw] is not None)
        ):
            if module_celerite2:
                kernel = terms.SHOTerm(
                    S0=np.exp(tpars[ks].value),
                    Q=1 / np.sqrt(2),
                    w0=np.exp(tpars[ko].value),
                )
                gp = GP(kernel, mean=0)
                err2 = dataset.lc["flux_err"] ** 2 + np.exp(tpars[kw].value) ** 2
                gp.compute(dataset.lc["time"], diag=err2, quiet=True)
            else:
                kernel = terms.SHOTerm(
                    log_S0=tpars[ks].value,
                    log_omega0=tpars[ko].value,
                    log_Q=np.log(1 / np.sqrt(2)),
                )
                kernel.freeze_parameter("log_Q")
                kernel += terms.JitterTerm(log_sigma=tpars[kw].value)
                gp = GP(kernel, mean=0, fit_mean=False)
                gp.compute(dataset.lc["time"], dataset.lc["flux_err"])
            gps.append(gp)
        else:
            gps.append(None)
        # take parameters from last fit
        single_pars.append(lmfit.params.copy())

    for gn in gnames:
        params_fit[gn].value = np.mean(
            [s[gn] for s in single_pars]
        )  # np.median([s[gn] for s in single_pars])
    for tn in tnames:
        for i, spar in enumerate(single_pars):
            if tn in spar:
                pn = "{}_v{}".format(tn, i + 1)
                params_fit[pn].value = spar[tn].value

    pfit = []  # fitting values
    for p in params:
        if params_fit[p].vary:
            pfit.append(params_fit[p].value)

    return params_fit, pfit, gps


# =====================================================================
# CHECK PARAMETERS BOUNDARIES
def check_boundaries(pfit, pnames, params):

    within = True
    for i, k in enumerate(pnames):
        v = pfit[i]
        if (v < params[k].min) or (v > params[k].max):
            within = False

    return within


# =====================================================================
# SET PYCHEOPS PARAMETERS FOR A SINGLE LC FROM FITTING PARAMETES
# OF MULTI-LCS
def set_pycheops_par(pfit, pnames, gnames, tnames, params, dataset, visit):

    v = "v{:d}".format(visit)
    pars = dataset.model.make_params()
    for k in pars:
        kk = "{:s}_{:s}".format(k, v)
        if k in gnames:
            pars[k] = params[k]
            if k in pnames:
                pars[k].value = pfit[pnames.index(k)]
        elif k in tnames:
            if kk in pnames:
                pars[k] = params[kk]
                pars[k].value = pfit[pnames.index(kk)]

    return pars


# =====================================================================
# MULTI-TRANSIT MODEL: USES IT DURING FIT
def multi_model(pfit, pnames, gnames, tnames, params_in, datasets):

    within = check_boundaries(pfit, pnames, params_in)
    if not within:
        return -np.inf, -np.inf, -np.inf, -np.inf

    wss, lnlike, lnprior = [], [], []
    flux_model = []
    for i, d in enumerate(datasets):
        time, flux, eflux = d.lc["time"], d.lc["flux"], d.lc["flux_err"]
        pars = set_pycheops_par(pfit, pnames, gnames, tnames, params_in, d, i + 1)

        ff = d.model.eval(pars, t=time)
        if not np.all(np.isfinite(ff)):
            return -np.inf, -np.inf, -np.inf, -np.inf

        flux_model.append(ff)

        k = "log_sigma_v{}".format(i + 1)
        if k in pnames:
            j = np.exp(pfit[pnames.index(k)])
        else:
            j = 0.0
        ef2 = eflux ** 2
        s2 = ef2 + j ** 2

        lnp = 0.0
        for k in pars:
            v = pars[k].value
            u = pars[k].user_data
            if isinstance(u, UFloat):
                lnp += -0.5 * ((u.n - v) / u.s) ** 2

        res = flux - ff
        wss.append(np.sum(res * res / ef2))
        lnlike.append(-0.5 * np.sum(res * res / s2 + np.log(2 * np.pi * s2)))
        lnprior.append(_log_prior(pars["D"], pars["W"], pars["b"]) + lnp)

    return flux_model, wss, lnlike, lnprior


# =====================================================================
# MULTI-TRANSIT MODEL: USES IT DURING PLOT OR MODELLING FINAL PARAMETERS
def components_model(
    pfit, pnames, gnames, tnames, params, datasets, gps, fill_gaps=False
):

    wss, lnlike, lnprior = [], [], []
    flux_all, flux_model = [], []
    flux_trend, flux_glint = [], []
    time_fill = []
    flux_all_fill, flux_model_fill = [], []
    flux_trend_fill, flux_glint_fill = [], []
    lnlike_gp = []
    flux_gp, flux_gp_fill = [], []
    for i, d in enumerate(datasets):
        time, flux, eflux = d.lc["time"], d.lc["flux"], d.lc["flux_err"]
        pars = set_pycheops_par(pfit, pnames, gnames, tnames, params, d, i + 1)
        gp = gps[i]

        ff = d.model.eval(pars, t=time)
        flux_all.append(ff)

        if "glint_scale" in pars:
            flux_model.append(d.model.left.left.eval(pars, t=time))
            flux_trend.append(d.model.left.right.eval(pars, t=time))
            flux_glint.append(d.model.right.eval(pars, t=time))
            if fill_gaps:
                texp = d.exptime * cst.sec2day
                tt = np.arange(time[0], time[-1] + 0.5 * texp, texp)
                time_fill.append(tt)
                flux_all_fill.append(d.model.eval(pars, t=tt))
                flux_model_fill.append(d.model.left.left.eval(pars, t=tt))
                flux_trend_fill.append(d.model.left.right.eval(pars, t=tt))
                flux_glint_fill.append(d.model.right.eval(pars, t=tt))

        else:
            flux_model.append(d.model.left.eval(pars, t=time))
            flux_trend.append(d.model.right.eval(pars, t=time))
            flux_glint.append(np.zeros((len(time))))
            if fill_gaps:
                texp = d.exptime * cst.sec2day
                tt = np.arange(time[0], time[-1] + 0.5 * texp, texp)
                time_fill.append(tt)
                flux_all_fill.append(d.model.eval(pars, t=tt))
                flux_model_fill.append(d.model.left.eval(pars, t=tt))
                flux_trend_fill.append(d.model.right.eval(pars, t=tt))
                flux_glint_fill.append(np.zeros((len(time_fill[i]))))

        k = "log_sigma_v{}".format(i + 1)
        if k in pnames:
            j = np.exp(pfit[pnames.index(k)])
        else:
            j = 0.0

        ef2 = eflux ** 2
        s2 = ef2 + j ** 2

        lnp = 0.0
        for k in pars:
            v = pars[k].value
            u = pars[k].user_data
            if isinstance(u, UFloat):
                lnp += -0.5 * ((u.n - v) / u.s) ** 2

        if gp is not None:
            res = flux - ff
            #             res = flux - flux_model[i]
            ks = "log_S0_v{}".format(i + 1)
            ko = "log_omega0_v{}".format(i + 1)
            kw = "log_sigma_v{}".format(i + 1)
            gp.set_parameter("kernel:terms[0]:log_S0", pfit[pnames.index(ks)])
            gp.set_parameter("kernel:terms[0]:log_omega0", pfit[pnames.index(ko)])
            gp.set_parameter("kernel:terms[1]:log_sigma", pfit[pnames.index(kw)])

            lnlgp = gp.log_likelihood(res)
            lnlike_gp.append(lnlgp)

            pred_mean = gp.predict(res, time, return_cov=False, return_var=False)
            flux_gp.append(pred_mean)
            if fill_gaps:
                pred_mean = gp.predict(
                    res, time_fill[i], return_cov=False, return_var=False
                )
                flux_gp_fill.append(pred_mean)
        else:
            lnlike_gp.append(0.0)
            flux_gp.append(np.zeros((len(time))))
            if fill_gaps:
                flux_gp_fill.append(np.zeros((len(time_fill[i]))))

        wss.append(np.sum((flux - ff) ** 2 / ef2))
        lnlike.append(-0.5 * np.sum((flux - ff) ** 2 / s2 + np.log(2 * np.pi * s2)))
        lnprior.append(_log_prior(pars["D"], pars["W"], pars["b"]) + lnp)

    # ---- output
    fluxes = {
        "all": flux_all,
        "model": flux_model,
        "trend": flux_trend,
        "glint": flux_glint,
        "gp": flux_gp,
    }

    stats = {
        "WSS_single": wss,
        "lnLike_single": lnlike,
        "lnLike_gp_single": lnlike_gp,
        "lnPrior_single": lnprior,
        "WSS": np.sum(wss),
        "lnLike": np.sum(lnlike),
        "lnLike_gp": np.sum(lnlike_gp),
        "lnPrior": np.sum(lnprior),
        "lnL": np.sum(lnlike) + np.sum(lnprior),
    }
    if fill_gaps:
        fluxes["time_fill"] = time_fill
        fluxes["all_fill"] = flux_all_fill
        fluxes["model_fill"] = flux_model_fill
        fluxes["trend_fill"] = flux_trend_fill
        fluxes["glint_fill"] = flux_glint_fill
        fluxes["gp_fill"] = flux_gp_fill

    return fluxes, stats


# =====================================================================
# PLOT OF LC GIVEN BEST-PARAMETER SET
# IT ALSO PLOTS THE RANDOM SAMPLES FROM POSTERIOR DISTRIBUTION
# PLOT EACH LC AND ONE PHASE LC
def plot_final_params(
    pbest,
    pnames,
    gnames,
    tnames,
    params,
    datasets,
    gps,
    out_folder,
    fill_gaps=True,
    emcee=True,
    pars_type="mle",
    nsamples=0,
    flatchain=None,
):

    output_folder = Path(out_folder).resolve()

    fluxes, stats = components_model(
        pbest, pnames, gnames, tnames, params, datasets, gps, fill_gaps=fill_gaps
    )

    print(fluxes.keys())
    if emcee:
        estr = "emcee_"
    else:
        estr = ""

    if nsamples > 0 and flatchain is not None:
        print("Computing samples ...")
        samples = []
        npost, nfit = np.shape(flatchain)
        smp_idx = np.random.choice(npost, size=nsamples, replace=False)
        print("...indexes for {} samples ...".format(nsamples))
        for idx in smp_idx:
            psmp = flatchain[idx, :]
            smp_f, _ = components_model(
                psmp, pnames, gnames, tnames, params, datasets, gps, fill_gaps=fill_gaps
            )
            samples.append(smp_f)
        print("...done")
        # alpha samples
        al_s = max(1.0 / nsamples, 0.05)
    else:
        samples = None
        smp_idx = None
        al_s = 0.0

    wss = stats["WSS"]
    lnlike = stats["lnLike"]
    lnlike_gp = stats["lnLike_gp"]
    lnprior = stats["lnPrior"]
    lnl = stats["lnL"]
    nfit = len(pbest)
    ndata = np.sum([len(d.lc["time"]) for d in datasets])
    bicc = nfit * np.log(ndata)
    rchis = wss / (ndata - nfit)
    bic_wss = bicc + wss
    bic_lnl = bicc - 2 * lnl
    print("W. Sum Squares  = {}".format(wss))
    print("Red. Chi Square = {}".format(rchis))
    print("lnLike          = {}".format(lnlike))
    print("lnLike GP       = {}".format(lnlike_gp))
    print("lnPrior         = {}".format(lnprior))
    print("lnL             = {}".format(lnl))
    print("BIC (WSS)       = {}".format(bic_wss))
    print("BIC (lnL)       = {}".format(bic_lnl))

    times = []
    res_tra = []
    res_all = []

    # figure
    xt_font = 6
    lt_font = 8

    # zorder
    zo_d = 6  # data
    zo_a = 9  # all
    zo_m = 10  # model transit
    zo_s = 8  # samples
    zo_t = 7  # trend
    # lweight
    lw_m = 1.2  # model transit
    lw_a = 1.0  # all
    lw_t = 0.9  # trend

    nrows = 5
    ncols = 1

    vir_map = plt.cm.get_cmap("viridis")
    nlc = len(datasets)
    icol = np.linspace(0, 1, num=nlc, endpoint=True)

    fig_phi = plt.figure()
    # -------------------
    # DETRENDED MODEL W//O TREND IN PHASE
    axd_phi = plt.subplot2grid(
        (nrows - 2, ncols), (0, 0), rowspan=2, colspan=1, fig=fig_phi
    )
    axd_phi.ticklabel_format(useOffset=False)
    axd_phi.tick_params(labelsize=xt_font, labelbottom=False)
    title = "{:s} - PHASE Light-Curve".format(datasets[0].target)
    axd_phi.set_title(title, fontsize=8)
    axd_phi.set_ylabel("flux", fontsize=lt_font)
    # -------------------
    # RESIDUALS IN PHASE
    axr_phi = plt.subplot2grid(
        (nrows - 2, ncols), (2, 0), rowspan=1, colspan=1, fig=fig_phi
    )
    axr_phi.ticklabel_format(useOffset=False)
    axr_phi.tick_params(labelsize=xt_font, labelbottom=True)
    axr_phi.axhline(0.0, color="black", ls="-", lw=0.5, zorder=5)
    axr_phi.set_ylabel("res.", fontsize=lt_font)
    axr_phi.set_xlabel(r"$\phi$", fontsize=lt_font)

    # ====================
    # plot data and model!
    for i, dataset in enumerate(datasets):

        lc = dataset.lc
        t, f, ef = lc["time"], lc["flux"], lc["flux_err"]

        if "P" in pnames:
            P = pbest[pnames.index("P")]
        else:
            P = params["P"].value

        kT0 = "T_0_v{:d}".format(i + 1)
        if kT0 in pnames:
            T0 = pbest[pnames.index(kT0)]
        else:
            T0 = params[kT0].value

        phi = (((t - T0) / P) % 1 + 0.5) % 1
        phi_fill = (((fluxes["time_fill"][i] - T0) / P) % 1 + 0.5) % 1
        cphi = vir_map(icol[i])

        fd = f / fluxes["trend"][i]

        ydet = "flux/trend"
        if fluxes["glint"][i] is not None:
            fgl_fill = fluxes["glint_fill"][i]
            fd -= fluxes["glint"][i]
            ydet += " - glint"
        else:
            fgl_fill = np.zeros(np.shape(fluxes["time_fill"][i]))
        fr = f - fluxes["all"][i]

        if gps[i] is not None:
            print("visit {} has gp /= None".format(i + 1))
            fgp = fluxes["gp"][i]
            ydet += " - GP"
            if fill_gaps:
                fgp_fill = fluxes["gp_fill"][i]
        else:
            print("visit {} has gp == None".format(i + 1))
            fgp = np.zeros(np.shape(t))
            if fill_gaps:
                fgp_fill = np.zeros(np.shape(fluxes["time_fill"][i]))
        fd -= fgp
        fr -= fgp

        title = r"{:s} - visit #{:d} - aperture {:s} ({:.0f}px) - {:s}".format(
            dataset.target, i + 1, lc["aperture"], dataset.ap_rad, pars_type.upper()
        )

        fig = plt.figure()

        # -------------------
        # FULL MODEL W/ TREND
        axf = plt.subplot2grid((nrows, ncols), (0, 0), rowspan=2, colspan=1)
        axf.ticklabel_format(useOffset=False)
        axf.tick_params(labelsize=xt_font, labelbottom=False)
        axf.set_title(title, fontsize=8)
        # data
        axf.errorbar(
            t,
            f,
            yerr=ef,
            color="C0",
            marker="o",
            ms=2,
            mec=out_color,
            mew=0.5,
            ls="",
            ecolor=out_color,
            elinewidth=0.5,
            capsize=0,
            zorder=zo_d,
        )

        # transit model
        axf.plot(
            fluxes["time_fill"][i],
            fluxes["model_fill"][i],
            color="black",
            marker="None",
            ls="-",
            lw=lw_m,
            zorder=zo_m,
        )

        # transit + trend + glint + gp model
        axf.plot(
            fluxes["time_fill"][i],
            fluxes["all_fill"][i] + fgp_fill,
            color="C1",
            marker="None",
            ls="-",
            lw=lw_a,
            zorder=zo_a,
        )
        # trend + glint + gp model
        axf.plot(
            fluxes["time_fill"][i],
            fluxes["trend_fill"][i] + fgl_fill + fgp_fill,
            color="C2",
            marker="None",
            ls="-",
            lw=lw_t,
            alpha=0.6,
            zorder=zo_t,
        )

        axf.set_ylabel("flux")

        # -------------------
        # DETRENDED MODEL W/O TREND
        axd = plt.subplot2grid((nrows, ncols), (2, 0), rowspan=2, colspan=1)
        axd.ticklabel_format(useOffset=False)
        axd.tick_params(labelsize=xt_font, labelbottom=False)

        # data
        axd.errorbar(
            t,
            fd,
            yerr=ef,
            color="C0",
            marker="o",
            ms=2,
            mec=out_color,
            mew=0.5,
            ls="",
            ecolor=out_color,
            elinewidth=0.5,
            capsize=0,
            zorder=zo_d,
        )
        # transit model
        axd.plot(
            fluxes["time_fill"][i],
            fluxes["model_fill"][i],
            color="black",
            marker="None",
            ls="-",
            lw=lw_m,
            zorder=zo_m,
        )
        axd.set_ylabel(ydet)
        # in phase ...
        axd_phi.errorbar(
            phi,
            fd,
            yerr=ef,
            color=cphi,
            marker="o",
            ms=2,
            mec=out_color,
            mew=0.5,
            ls="",
            ecolor=out_color,
            elinewidth=0.5,
            capsize=0,
            zorder=zo_d,
            label="LC#{:d}".format(i + 1),
        )
        axd_phi.plot(
            phi_fill,
            fluxes["model_fill"][i],
            color="black",
            marker="None",
            ls="-",
            lw=lw_m,
            zorder=zo_m,
            alpha=1.0 / nlc,
        )

        # -------------------
        # RESIDUALS
        axr = plt.subplot2grid((nrows, ncols), (4, 0), rowspan=1, colspan=1)
        axr.ticklabel_format(useOffset=False)
        axr.tick_params(labelsize=xt_font, labelbottom=True)

        axr.errorbar(
            t,
            fr,
            yerr=ef,
            color="C0",
            marker="o",
            ms=2,
            mec=out_color,
            mew=0.5,
            ls="",
            ecolor=out_color,
            elinewidth=0.5,
            capsize=0,
            zorder=zo_d,
        )
        axr.axhline(0.0, color="black", ls="-", lw=0.5, zorder=5)
        axr.set_ylabel("res.")
        axr.set_xlabel(r"BJD$_\mathrm{{TDB}} - {}$".format(lc["bjd_ref"]))
        # in phase ...
        axr_phi.errorbar(
            phi,
            fr,
            yerr=ef,
            color=cphi,
            marker="o",
            ms=2,
            mec=out_color,
            mew=0.5,
            ls="",
            ecolor=out_color,
            elinewidth=0.5,
            capsize=0,
            zorder=zo_d,
        )

        # --------------------
        # SAMPLES
        if samples is not None:
            for sample in samples:
                tt = fluxes["time_fill"][i]
                s_gp = sample["gp_fill"][i]
                if s_gp is None:
                    s_gp = np.zeros(np.shape(tt))
                s_all = sample["all_fill"][i] + s_gp
                s_tra = sample["model_fill"][i]
                # transit + trend + glint + gp model
                axf.plot(
                    tt,
                    s_all,
                    color="gray",
                    marker="None",
                    ls="-",
                    lw=lw_a,
                    alpha=al_s,
                    zorder=zo_s,
                )
                axd.plot(
                    tt,
                    s_tra,
                    color="gray",
                    marker="None",
                    ls="-",
                    lw=lw_a,
                    alpha=al_s,
                    zorder=zo_s,
                )
                # in phase ...
                axd_phi.plot(
                    phi_fill,
                    s_tra,
                    color="gray",
                    marker="None",
                    ls="-",
                    lw=lw_a,
                    alpha=al_s,
                    zorder=zo_s,
                )

        fig.tight_layout()
        fig.show()
        fig.savefig(
            output_folder.joinpath(
                "01_lc_v{:02d}_{:s}{:s}.png".format(i + 1, estr, pars_type.lower())
            ),
            bbox_inches="tight",
        )
        fig.savefig(
            output_folder.joinpath(
                "01_lc_v{:02d}_{:s}{:s}.pdf".format(i + 1, estr, pars_type.lower())
            ),
            bbox_inches="tight",
        )

        # RMS COMPUTATION
        # per visit
        print("\nRMS visit {}".format(i + 1))
        # flux - transit
        print("flux-transit")
        rtra = f - fluxes["model"][i]
        rms_unbin = np.std(rtra, ddof=1) * 1.0e6
        print("RMS ({:8s}) = {:8.2f}".format("unbinned", rms_unbin))
        for binw in rms_time:
            _, _, e_bin, n_bin = lcbin(t, rtra, binwidth=binw * cst.hour2day)
            rms_bin = e_bin * np.sqrt(n_bin - 1)
            nrms = len(rms_bin)
            rms = np.mean(rms_bin) * 1.0e6
            std = np.std(rms_bin, ddof=1) * 1.0e6
            if binw >= 1.0:
                binw_str = "{:1.0f}hr".format(binw)
            else:
                binw_str = "{:2.0f}min".format(binw * 60.0)
            print(
                "RMS ({:8s}) = {:8.2f} +/- {:5.2f} (n_bin = {})".format(
                    binw_str, rms, std, nrms
                )
            )
        print("flux-all")
        rms_unbin = np.std(fr, ddof=1) * 1.0e6
        print("RMS ({:8s}) = {:8.2f}".format("unbinned", rms_unbin))
        for binw in rms_time:
            _, _, e_bin, n_bin = lcbin(t, fr, binwidth=binw * cst.hour2day)
            rms_bin = e_bin * np.sqrt(n_bin - 1)
            nrms = len(rms_bin)
            rms = np.mean(rms_bin) * 1.0e6
            std = np.std(rms_bin, ddof=1) * 1.0e6 / np.sqrt(nrms - 1)
            if binw >= 1.0:
                binw_str = "{:1.0f}hr".format(binw)
            else:
                binw_str = "{:2.0f}min".format(binw * 60.0)
            print(
                "RMS ({:8s}) = {:8.2f} +/- {:5.2f} (n_bin = {})".format(
                    binw_str, rms, std, nrms
                )
            )
        print()
        times.append(t)
        res_tra.append(rtra)
        res_all.append(fr)
    # END FOR LOOP

    axd_phi.legend(loc="best", fontsize=xt_font)
    fig_phi.tight_layout()
    fig_phi.show()
    # plt.draw()
    fig_phi.savefig(
        output_folder.joinpath(
            "01_lc_phase_{:s}{:s}.png".format(estr, pars_type.lower())
        ),
        bbox_inches="tight",
    )
    fig_phi.savefig(
        output_folder.joinpath(
            "01_lc_phase_{:s}{:s}.pdf".format(estr, pars_type.lower())
        ),
        bbox_inches="tight",
    )
    plt.close("all")
    # RMS COMPUTATION
    # full
    times = np.concatenate(times)
    sort_idx = np.argsort(times)
    times = times[sort_idx]
    res_tra = np.concatenate(res_tra)[sort_idx]
    res_all = np.concatenate(res_all)[sort_idx]
    print("==========================")
    print("RMS Combined {} visits".format(len(datasets)))
    print("flux-transit")
    rms_unbin = np.std(res_tra, ddof=1) * 1.0e6
    print("RMS ({:8s}) = {:8.2f}".format("unbinned", rms_unbin))
    for binw in rms_time:
        _, _, e_bin, n_bin = lcbin(times, res_tra, binwidth=binw * cst.hour2day)
        rms_bin = e_bin * np.sqrt(n_bin - 1)
        nrms = len(rms_bin)
        rms = np.mean(rms_bin) * 1.0e6
        std = np.std(rms_bin, ddof=1) * 1.0e6 / np.sqrt(nrms - 1)
        if binw >= 1.0:
            binw_str = "{:1.0f}hr".format(binw)
        else:
            binw_str = "{:2.0f}min".format(binw * 60.0)
        print(
            "RMS ({:8s}) = {:8.2f} +/- {:5.2f} (n_bin = {})".format(
                binw_str, rms, std, nrms
            )
        )
    print("flux-all")
    rms_unbin = np.std(res_all, ddof=1) * 1.0e6
    print("RMS ({:8s}) = {:8.2f}".format("unbinned", rms_unbin))
    for binw in rms_time:
        _, _, e_bin, n_bin = lcbin(times, res_all, binwidth=binw * cst.hour2day)
        rms_bin = e_bin * np.sqrt(n_bin - 1)
        nrms = len(rms_bin)
        rms = np.mean(rms_bin) * 1.0e6
        std = np.std(rms_bin, ddof=1) * 1.0e6 / np.sqrt(nrms - 1)
        if binw >= 1.0:
            binw_str = "{:1.0f}hr".format(binw)
        else:
            binw_str = "{:2.0f}min".format(binw * 60.0)
        print(
            "RMS ({:8s}) = {:8.2f} +/- {:5.2f} (n_bin = {})".format(
                binw_str, rms, std, nrms
            )
        )
    print("==========================")
    print()

    return


# =====================================================================
# LNPROBABILITY: NO GAUSSIAN PROCESS
def lnprob_woGP(pfit, pnames, gnames, tnames, params, datasets):

    _, _, lnlike, lnprior = multi_model(pfit, pnames, gnames, tnames, params, datasets)
    if not np.all(np.isfinite(lnlike)):
        lnl = -np.inf
    else:
        lnl = np.sum(lnlike) + np.sum(lnprior)

    return lnl


# =====================================================================
# LNPROBABILITY: WITH GAUSSIAN PROCESS
def lnprob_wGP(pfit, pnames, gnames, tnames, params, datasets, gps):

    flux_model, _, lnlike, lnprior = multi_model(
        pfit, pnames, gnames, tnames, params, datasets
    )
    if not np.all(np.isfinite(lnlike)):
        lnl = -np.inf
    else:
        lnlgp = 0.0
        for i, d in enumerate(datasets):
            gp = gps[i]
            if gp is not None:
                res = d.lc["flux"] - flux_model[i]
                ks = "log_S0_v{}".format(i + 1)
                ko = "log_omega0_v{}".format(i + 1)
                kw = "log_sigma_v{}".format(i + 1)
                gp.set_parameter("kernel:terms[0]:log_S0", pfit[pnames.index(ks)])
                gp.set_parameter("kernel:terms[0]:log_omega0", pfit[pnames.index(ko)])
                gp.set_parameter("kernel:terms[1]:log_sigma", pfit[pnames.index(kw)])
                lnlgp += gp.log_likelihood(res)

        lnl = lnlgp + np.sum(lnprior)

    return lnl


# =====================================================================
# INITIALISES WALKERS FOR EMCEE
def init_walkers(lnprob, pfit, pscale, nwalkers, args=(), init_scale=1.0e-3):

    nfit = len(pfit)
    # init walkers
    pos = []
    # gaussian hyperball close to initial solution
    for _ in range(nwalkers):
        lnlike_i = -np.inf
        while lnlike_i == -np.inf:
            pos_i = pfit + pscale * np.random.randn(nfit) * init_scale
            lnlike_i = lnprob(pos_i, *args)
        pos.append(pos_i)

    # # initial solution and random within boundaries
    # pmin = np.array([params_fit[n].min for n in pnames])
    # pmax = np.array([params_fit[n].max for n in pnames])
    # dbnd = pmax-pmin
    # for i in range(nwalkers-1):
    #     lnlike_i = -np.inf
    #     while lnlike_i == -np.inf:
    #         pos_i = pmin + np.random.random(nfit)*dbnd
    #         lnlike_i = lnprob(pos_i, *args)
    #     pos.append(pos_i)
    # pos.append(pfit)

    return pos


# =====================================================================
# CONVERT EMCEE sampler OBJ TO A DICTIONARY
def sampler_to_dict(sampler, nburn=0, nprerun=0, nthin=1):

    out_dict = {}
    out_dict["chain"] = sampler.get_chain()
    out_dict["log_prob"] = sampler.get_log_prob()

    nsteps, nwalkers, nfit = np.shape(out_dict["chain"])

    out_dict["flatchain"] = sampler.get_chain(
        flat=True, discard=nburn, thin=nthin
    ).reshape((-1, nfit))
    out_dict["log_prob_post"] = sampler.get_log_prob(
        flat=True, discard=nburn, thin=nthin
    )

    out_dict["nwalkers"] = nwalkers
    out_dict["nfit"] = nfit
    out_dict["nprerun"] = nprerun
    out_dict["nsteps"] = nsteps
    out_dict["nthin"] = nthin
    out_dict["nburn"] = nburn

    return out_dict


# =====================================================================
# DO EMCEE ANALYSIS!
def do_emcee_analysis(
    lnprob,
    pfit,
    pscale,
    out_folder,
    args=(),
    nwalkers=64,
    nprerun=0,
    nsteps=512,
    nthin=4,
    nburn=128,
    progress=True,
    run_emcee=True,
    read_sampler=False,
):

    output_folder = Path(out_folder).resolve()
    # Set up the backend
    # Don't forget to clear it in case the file already exists
    filename = output_folder.joinpath("emcee_sampler.h5")

    nfit = len(pfit)
    # init walkers
    pos = init_walkers(lnprob, pfit, pscale, nwalkers, args=args)

    # run emcee
    if run_emcee:
        print("\nEMCEE READY TO RUN!")
        backend = emcee.backends.HDFBackend(filename)
        backend.reset(nwalkers, nfit)

        sampler = emcee.EnsembleSampler(
            nwalkers, nfit, lnprob, args=args, backend=backend
        )
        if progress:
            print("Running burn-in ..")
            sys.stdout.flush()
        if nprerun > 0:
            pos, _, _ = sampler.run_mcmc(
                pos,
                nprerun,
                store=False,
                skip_initial_state_check=True,
                progress=progress,
            )
            sampler.reset()
        if progress:
            print("Running sampler ..")
            sys.stdout.flush()
        _ = sampler.run_mcmc(
            pos, nsteps, thin_by=nthin, skip_initial_state_check=True, progress=progress
        )

    # read emcee sampler file
    if read_sampler:
        print("\nREAD EMCEE FILE {}".format(filename))
        sampler = emcee.backends.HDFBackend(filename, read_only=True)

    # convert to sampler_dictionary
    sampler_dict = sampler_to_dict(sampler, nburn=nburn, nprerun=nprerun, nthin=nthin)

    return sampler_dict


# =====================================================================
# COMPUTES THE HDI/HPD FROM PYASTRONOMY
def high_posterior_density(trace, cred=0.6827):
    """
    Estimate the highest probability density interval.

    This function determines the shortest, continuous interval
    containing the specified fraction (cred) of steps of
    the Markov chain. Note that multi-modal distribution
    may require further scrutiny.

    # cred = 0.6827 <-> -/+ 1 sigma
    # cred = 0.9545 <-> -/+ 2 sigma
    # cred = 0.9973 <-> -/+ 3 sigma
    # cred = 0.999999426696856 <-> -/+ 5 sigma

    Parameters
    ----------
    trace : array
        The steps of the Markov chain.
    cred : float
        The probability mass to be included in the
        interval (between 0 and 1).

    Returns
    -------
    start, end : float
        The start and end points of the interval.
    """
    cred_def = 0.6827
    if (cred > 1.0) or (cred < 0.0):
        print("CRED HAS TO BE: 0 < cred < 1 ==> setting to cred = {}".format(cred_def))
        cred = cred_def

    # Sort the trace steps in ascending order
    st = np.sort(trace)

    # Number of steps in the chain
    n = len(st)
    # Number of steps to be included in the interval
    nin = int(n * cred)

    # All potential intervals must be 1) continuous and 2) cover
    # the given number of trace steps. Potential start and end
    # points of the HPD are given by
    starts = st[0:-nin]
    ends = st[nin:]
    # All possible widths are
    widths = ends - starts
    # The density is highest in the shortest one
    imin = np.argmin(widths)

    return starts[imin], ends[imin]


# =====================================================================
# COMPUTES THE MEDIAN, MLE, AND CI-HDI/HPD @ 68/27% FOR ONE SINGLE CHAIN
def single_chain_summary(singlechain, idx):

    median = np.median(singlechain)
    ci_16p = np.percentile(singlechain, 15.87)
    ci_84p = np.percentile(singlechain, 84.13)
    median_err = 0.5 * (ci_84p - ci_16p)

    mle = singlechain[idx]
    hdi_low, hdi_upp = high_posterior_density(singlechain)
    hdi_err = 0.5 * (hdi_upp - hdi_low)

    return mle, hdi_err, hdi_low, hdi_upp, median, median_err, ci_16p, ci_84p


# =====================================================================
# COMPUTES BEST PARAMETERS FROM POSTERIOR DISTRIBUTION
# AS MEDIAND AND MLE
# COMPUTES DERIVED PARAMETERS
# PRINT THEM ALL
def summary_parameters(samples, pnames, params, mle_within_hdi=True):

    nfit = len(pnames)

    flatchain = samples["flatchain"]
    lnL = samples["log_prob_post"]

    lnL_low, lnL_upp = high_posterior_density(lnL)
    # MLE == Maximum Likelihood Estimation or MAP
    # but the MLE/MAP is not the centre of the lnL,
    # that is the parameter set associated to the MLE
    # is out of the HDI of the single parameter posterior distribution
    # so computes the MLE as:
    # lnL_post within HDI, median of this selection lnL_med_hdi
    # computes the delta_lnL = |lnL - lnL_med_hdi|
    # sort the delta_lnL and takes the parameter set associated
    # with the first element (closes to the lnL_med_hdi)
    if mle_within_hdi:
        mask_ok = np.logical_and(lnL >= lnL_low, lnL <= lnL_upp)
        lnL_post_hdi = lnL[mask_ok]
        lnL_med_hdi = np.percentile(lnL_post_hdi, 50, interpolation="midpoint")
        dlnL = np.abs(lnL - lnL_med_hdi)
        lnL_idx = np.argmin(dlnL)
    else:
        lnL_idx = np.argmax(lnL)
    lnL_mle = lnL[lnL_idx]

    lnL_med = np.median(lnL)
    lnL_q16 = np.percentile(lnL, 15.87, axis=0, interpolation="midpoint")
    lnL_q84 = np.percentile(lnL, 84.13, axis=0, interpolation="midpoint")

    params_mle = copy_parameters(params)
    pmle = flatchain[lnL_idx, :]

    params_median = copy_parameters(params)
    pmedian = np.median(flatchain, axis=0)
    pci_16p = np.percentile(flatchain, 15.87, axis=0, interpolation="midpoint")
    pci_84p = np.percentile(flatchain, 84.13, axis=0, interpolation="midpoint")
    pmedian_err = 0.5 * (pci_84p - pci_16p)

    phdi_low = np.zeros((nfit))
    phdi_upp = np.zeros((nfit))
    phdi_err = np.zeros((nfit))
    print("\nFITTING PARAMETERS")
    print(
        "{:20s} {:>11s} +/- {:>11s} ({:>11s} {:>11s}) {:>11s} +/- {:>11s} ({:>11s} {:>11s})".format(
            "parameter",
            "MLE",
            "HDI_ERR",
            "HDI_LOW",
            "HDI_UPP",
            "MEDIAN",
            "MEDIAN_ERR",
            "CI_16p",
            "CI_84p",
        )
    )

    for i, n in enumerate(pnames):
        low, upp = high_posterior_density(flatchain[:, i])
        phdi_low[i], phdi_upp[i] = low, upp
        phdi_err[i] = 0.5 * (upp - low)
        print(
            "{:20s} {:11s} +/- {:11s} ({:11s} {:11s}) {:11s} +/- {:11s} ({:11s} {:11s})".format(
                n,
                gformat(pmle[i]),
                gformat(phdi_err[i]),
                gformat(low),
                gformat(upp),
                gformat(pmedian[i]),
                gformat(pmedian_err[i]),
                gformat(pci_16p[i]),
                gformat(pci_84p[i]),
            )
        )
        params_mle[n].value = pmle[i]
        params_mle[n].stderr = phdi_err[i]
        params_median[n].value = pmedian[i]
        params_median[n].stderr = pmedian_err[i]

    print("\nDERIVED PARAMETERS")

    # d -> derived
    dnames = []
    dchains = []
    dmle, dhdi_err, dhdi_low, dhdi_upp = [], [], [], []
    dmedian, dmedian_err, dci_16p, dci_84p = [], [], [], []

    # k = sqrt(D)
    dnames.append("k")
    kchain = np.sqrt(flatchain[:, pnames.index("D")])
    dchains.append(kchain)
    (
        mle,
        hdi_err,
        hdi_low,
        hdi_upp,
        median,
        median_err,
        ci_16p,
        ci_84p,
    ) = single_chain_summary(kchain, lnL_idx)
    print(
        "{:20s} {:11s} +/- {:11s} ({:11s} {:11s}) {:11s} +/- {:11s} ({:11s} {:11s})".format(
            "k",
            gformat(mle),
            gformat(hdi_err),
            gformat(hdi_low),
            gformat(hdi_upp),
            gformat(median),
            gformat(median_err),
            gformat(ci_16p),
            gformat(ci_84p),
        )
    )
    dmle.append(mle)
    dhdi_err.append(hdi_err)
    dhdi_low.append(hdi_low)
    dhdi_upp.append(hdi_upp)
    dmedian.append(median)
    dmedian_err.append(median_err)
    dci_16p.append(ci_16p)
    dci_84p.append(ci_84p)

    # W in days
    if "W" in pnames:
        dnames.append("W_d")
        if "P" in pnames:
            P = flatchain[:, pnames.index("P")]
        else:
            P = params["P"].value
        Wdchain = flatchain[:, pnames.index("W")] * P
        dchains.append(Wdchain)
        (
            mle,
            hdi_err,
            hdi_low,
            hdi_upp,
            median,
            median_err,
            ci_16p,
            ci_84p,
        ) = single_chain_summary(Wdchain, lnL_idx)
        print(
            "{:20s} {:11s} +/- {:11s} ({:11s} {:11s}) {:11s} +/- {:11s} ({:11s} {:11s})".format(
                "W_d",
                gformat(mle),
                gformat(hdi_err),
                gformat(hdi_low),
                gformat(hdi_upp),
                gformat(median),
                gformat(median_err),
                gformat(ci_16p),
                gformat(ci_84p),
            )
        )
        dmle.append(mle)
        dhdi_err.append(hdi_err)
        dhdi_low.append(hdi_low)
        dhdi_upp.append(hdi_upp)
        dmedian.append(median)
        dmedian_err.append(median_err)
        dci_16p.append(ci_16p)
        dci_84p.append(ci_84p)

    # a/Rs
    dnames.append("aRs")
    bchain = flatchain[:, pnames.index("b")]
    if "W" in pnames:
        W = flatchain[:, pnames.index("W")]
    else:
        W = params["W"].value
    aRschain = np.sqrt((1 + kchain) ** 2 - bchain * bchain) / (np.pi * W)
    dchains.append(aRschain)
    (
        mle,
        hdi_err,
        hdi_low,
        hdi_upp,
        median,
        median_err,
        ci_16p,
        ci_84p,
    ) = single_chain_summary(aRschain, lnL_idx)
    print(
        "{:20s} {:11s} +/- {:11s} ({:11s} {:11s}) {:11s} +/- {:11s} ({:11s} {:11s})".format(
            "aRs",
            gformat(mle),
            gformat(hdi_err),
            gformat(hdi_low),
            gformat(hdi_upp),
            gformat(median),
            gformat(median_err),
            gformat(ci_16p),
            gformat(ci_84p),
        )
    )
    dmle.append(mle)
    dhdi_err.append(hdi_err)
    dhdi_low.append(hdi_low)
    dhdi_upp.append(hdi_upp)
    dmedian.append(median)
    dmedian_err.append(median_err)
    dci_16p.append(ci_16p)
    dci_84p.append(ci_84p)

    # sini
    dnames.append("sini")
    sinichain = np.sqrt(1 - (bchain / aRschain) ** 2)
    dchains.append(sinichain)
    (
        mle,
        hdi_err,
        hdi_low,
        hdi_upp,
        median,
        median_err,
        ci_16p,
        ci_84p,
    ) = single_chain_summary(sinichain, lnL_idx)
    print(
        "{:20s} {:11s} +/- {:11s} ({:11s} {:11s}) {:11s} +/- {:11s} ({:11s} {:11s})".format(
            "sini",
            gformat(mle),
            gformat(hdi_err),
            gformat(hdi_low),
            gformat(hdi_upp),
            gformat(median),
            gformat(median_err),
            gformat(ci_16p),
            gformat(ci_84p),
        )
    )
    dmle.append(mle)
    dhdi_err.append(hdi_err)
    dhdi_low.append(hdi_low)
    dhdi_upp.append(hdi_upp)
    dmedian.append(median)
    dmedian_err.append(median_err)
    dci_16p.append(ci_16p)
    dci_84p.append(ci_84p)
    # inc
    dnames.append("inc")
    ichain = np.arcsin(sinichain) * cst.rad2deg
    dchains.append(ichain)
    (
        mle,
        hdi_err,
        hdi_low,
        hdi_upp,
        median,
        median_err,
        ci_16p,
        ci_84p,
    ) = single_chain_summary(ichain, lnL_idx)
    print(
        "{:20s} {:11s} +/- {:11s} ({:11s} {:11s}) {:11s} +/- {:11s} ({:11s} {:11s})".format(
            "inc",
            gformat(mle),
            gformat(hdi_err),
            gformat(hdi_low),
            gformat(hdi_upp),
            gformat(median),
            gformat(median_err),
            gformat(ci_16p),
            gformat(ci_84p),
        )
    )
    dmle.append(mle)
    dhdi_err.append(hdi_err)
    dhdi_low.append(hdi_low)
    dhdi_upp.append(hdi_upp)
    dmedian.append(median)
    dmedian_err.append(median_err)
    dci_16p.append(ci_16p)
    dci_84p.append(ci_84p)

    # ecc
    if "f_c" in pnames and "f_s" in pnames:
        dnames.append("ecc")
        echain = (
            flatchain[:, pnames.index("f_c")] ** 2
            + flatchain[:, pnames.index("f_s")] ** 2
        )
        dchains.append(echain)
        (
            mle,
            hdi_err,
            hdi_low,
            hdi_upp,
            median,
            median_err,
            ci_16p,
            ci_84p,
        ) = single_chain_summary(echain, lnL_idx)
        print(
            "{:20s} {:11s} +/- {:11s} ({:11s} {:11s}) {:11s} +/- {:11s} ({:11s} {:11s})".format(
                "ecc",
                gformat(mle),
                gformat(hdi_err),
                gformat(hdi_low),
                gformat(hdi_upp),
                gformat(median),
                gformat(median_err),
                gformat(ci_16p),
                gformat(ci_84p),
            )
        )
        dmle.append(mle)
        dhdi_err.append(hdi_err)
        dhdi_low.append(hdi_low)
        dhdi_upp.append(hdi_upp)
        dmedian.append(median)
        dmedian_err.append(median_err)
        dci_16p.append(ci_16p)
        dci_84p.append(ci_84p)

    # sigma_jitter for each lc
    for p in pnames:
        if "log_sigma" in p:
            n = p.replace("log_", "")
            dnames.append(n)
            sigmachain = np.exp(flatchain[:, pnames.index(p)])
            dchains.append(sigmachain)
            (
                mle,
                hdi_err,
                hdi_low,
                hdi_upp,
                median,
                median_err,
                ci_16p,
                ci_84p,
            ) = single_chain_summary(sigmachain, lnL_idx)
            print(
                "{:20s} {:11s} +/- {:11s} ({:11s} {:11s}) {:11s} +/- {:11s} ({:11s} {:11s})".format(
                    n,
                    gformat(mle),
                    gformat(hdi_err),
                    gformat(hdi_low),
                    gformat(hdi_upp),
                    gformat(median),
                    gformat(median_err),
                    gformat(ci_16p),
                    gformat(ci_84p),
                )
            )
            dmle.append(mle)
            dhdi_err.append(hdi_err)
            dhdi_low.append(hdi_low)
            dhdi_upp.append(hdi_upp)
            dmedian.append(median)
            dmedian_err.append(median_err)
            dci_16p.append(ci_16p)
            dci_84p.append(ci_84p)

    dchains = np.array(dchains).T

    summary = {
        "names": pnames,
        "flatchain": flatchain,
        "mle": pmle,
        "hdi_err": phdi_err,
        "hdi_low": phdi_low,
        "hdi_upp": phdi_upp,
        "params_mle": params_mle,
        "median": pmedian,
        "median_err": pmedian_err,
        "ci_16p": pci_16p,
        "ci_84p": pci_84p,
        "params_median": params_median,
        "lnL": lnL,
        "lnL_mle": lnL_mle,
        "lnL_low": lnL_low,
        "lnL_upp": lnL_upp,
        "lnL_med": lnL_med,
        "lnL_q16": lnL_q16,
        "lnL_q84": lnL_q84,
        "dnames": dnames,
        "dchains": dchains,
        "dmle": dmle,
        "dhdi_err": dhdi_err,
        "dhdi_low": dhdi_low,
        "dhdi_upp": dhdi_upp,
        "dmedian": dmedian,
        "dmedian_err": dmedian_err,
        "dci_16p": dci_16p,
        "dci_84p": dci_84p,
    }

    return summary


def trace_plot(sampler_dict, summary, out_folder):

    output_folder = Path(out_folder).resolve()

    nsteps = sampler_dict["nsteps"]
    nthin = sampler_dict["nthin"]
    nburn = sampler_dict["nburn"]

    pnames = summary["names"]
    mod_names = []
    temp = global_names + ["T_0"]
    for p in pnames:
        for t in temp:
            if p[: len(t)] == t:
                mod_names.append(p)
    nmod = len(mod_names)

    det_names = []
    temp = transit_names.copy()
    temp.remove("dT")
    temp.remove("T_0")
    for p in pnames:
        for t in temp:
            if p[: len(t)] == t:
                det_names.append(p)
    ndet = len(det_names)

    pmle = summary["mle"]
    phdi_err = summary["hdi_err"]
    phdi_low, phdi_upp = summary["hdi_low"], summary["hdi_upp"]
    pmedian = summary["median"]
    pmedian_err = summary["median_err"]
    pci_16p, pci_84p = summary["ci_16p"], summary["ci_84p"]

    # plot lnL trace
    print("-Loading full lnL")
    lnL_all = sampler_dict["log_prob"]
    # lnL_post = lnL_all[nburn+nthin-1:nsteps:nthin, :]

    lnL_mle = summary["lnL_mle"]
    lnL_med = summary["lnL_med"]
    lnL_low = summary["lnL_low"]
    lnL_upp = summary["lnL_upp"]

    # fontsize
    tk_font = 5
    l_font = 6
    # axes definition
    x0, y0, width = 0.125, 0.10, 0.775
    height_max = 0.85
    hplt = 0.5  # in

    # TRANSIT MODELS
    print("-Preparing figure for transit models")
    sys.stdout.flush()
    # a4 paper = 8.3 x 11.7 in = 210 x 297 mm
    nplt = nmod + 1
    fw, fh = 8.3, nplt * hplt
    if nplt > 12:
        print("--!!Figure will have huge height ({:.1f} in)!!".format(fh))

    # axes definition
    height_max_ax = height_max / nplt
    height = height_max_ax * 0.90
    space = height_max_ax * 0.10

    print("-Transit models full trace plot")
    # full - transit models
    fig = plt.figure(figsize=(fw, fh))

    print("-lnL chain", end=" ")
    ax = fig.add_axes([x0, y0, width, height])
    ax.ticklabel_format(useOffset=False)
    ax.tick_params(labelsize=tk_font, labelbottom=True)
    ax.set_ylabel(r"{}".format("lnL"), fontsize=l_font)
    print("full", end=" ")
    # plot chain
    ax.plot(lnL_all, color="black", marker="None", ls="-", lw=0.4, alpha=0.33, zorder=9)
    # plot hdi
    ax.axhspan(lnL_low, lnL_upp, fc="C2", ec="None", alpha=0.33, zorder=7)
    # plot mle
    ax.axhline(lnL_mle, color="C0", ls="-", lw=1.0, alpha=1.0, zorder=8)
    # plot median
    ax.axhline(lnL_med, color="C1", ls="-", lw=1.0, alpha=1.0, zorder=8)
    # nburn
    ax.axvline(nburn, color="darkgray", ls="-", lw=1.0, alpha=1.0, zorder=8)

    ax.set_xlabel("nsteps", fontsize=8)
    print("done", end=" ")
    sys.stdout.flush()

    for i, n in enumerate(mod_names):
        idx = pnames.index(n)
        increase_y0 = (i + 1) * (height + space)
        yr = y0 + increase_y0

        print("-{} chain".format(n), end=" ")
        chains = sampler_dict["chain"][:, :, idx]
        # full
        ax = fig.add_axes([x0, yr, width, height])
        ax.ticklabel_format(useOffset=False)
        ax.tick_params(labelsize=tk_font, labelbottom=False)
        ax.set_ylabel(r"{}".format(n), fontsize=l_font)
        print("full", end=" ")
        # plot chain
        ax.plot(
            chains[:, :],
            color="black",
            marker="None",
            ls="-",
            lw=0.4,
            alpha=0.33,
            zorder=9,
        )
        # plot mle w/ error
        ax.axhspan(
            phdi_low[idx],
            phdi_upp[idx],
            fc="C0",
            ec="None",
            alpha=0.33,
            zorder=7,
            label="mle HDI68.27%",
        )
        ax.axhline(
            pmle[idx], color="C0", ls="-", lw=1.0, alpha=1.0, zorder=8, label="mle"
        )
        ax.axhline(
            pmle[idx] - phdi_err[idx],
            color="C0",
            ls="--",
            lw=1.0,
            alpha=1.0,
            zorder=8,
            label="mle+/-err",
        )
        ax.axhline(
            pmle[idx] + phdi_err[idx], color="C0", ls="--", lw=1.0, alpha=1.0, zorder=8
        )
        # plot median w/ error
        ax.axhspan(
            pci_16p[idx],
            pci_84p[idx],
            fc="C1",
            ec="None",
            alpha=0.33,
            zorder=7,
            label="CI(16p-84p)",
        )
        ax.axhline(
            pmedian[idx],
            color="C1",
            ls="-",
            lw=1.0,
            alpha=1.0,
            zorder=8,
            label="median",
        )
        ax.axhline(
            pmedian[idx] - pmedian_err[idx],
            color="C1",
            ls="--",
            lw=1.0,
            alpha=1.0,
            zorder=8,
            label="median+/-err",
        )
        ax.axhline(
            pmedian[idx] + pmedian_err[idx],
            color="C1",
            ls="--",
            lw=1.0,
            alpha=1.0,
            zorder=8,
        )
        # burnin
        ax.axvline(nburn, color="darkgray", ls="-", lw=1.0, alpha=1.0, zorder=8)
        print("done")
        sys.stdout.flush()

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.5), ncol=6, fontsize=6)

    print("-Saving transit models full png ...", end=" ")
    sys.stdout.flush()
    fig.savefig(
        output_folder.joinpath("08_trace_full_emcee_transit_models.png"),
        bbox_inches="tight",
    )
    # print('and pdf ... ', end=' ')
    # sys.stdout.flush()
    # fig.savefig(output_folder.joinpath('08_trace_full_emcee_transit_models.pdf'), bbox_inches='tight')
    plt.close(fig)
    print("done")
    # gc.collect()

    print("-Transit models posterior trace plot")
    # posterior
    fig = plt.figure(figsize=(fh, fh))
    print("-lnL chain", end=" ")
    ax = fig.add_axes([x0, y0, width, height])
    ax.ticklabel_format(useOffset=False)
    ax.tick_params(labelsize=tk_font, labelbottom=True)
    ax.set_ylabel(r"{}".format("lnL"), fontsize=l_font)
    print("posterior", end=" ")
    # plot chain
    ax.plot(
        lnL_all[nburn + nthin - 1 : nsteps : nthin, :],
        color="black",
        marker="None",
        ls="-",
        lw=0.4,
        alpha=0.33,
        zorder=9,
    )
    # plot hdi
    ax.axhspan(lnL_low, lnL_upp, fc="C2", ec="None", alpha=0.33, zorder=7)
    # plot mle
    ax.axhline(lnL_mle, color="C0", ls="-", lw=1.0, alpha=1.0, zorder=8)
    # plot median
    ax.axhline(lnL_med, color="C1", ls="-", lw=1.0, alpha=1.0, zorder=8)
    # # nburn
    # ax.axvline(nburn, color='darkgray', ls='-', lw=1.0, alpha=1.0, zorder=8)

    ax.set_xlabel(
        "nsteps posterior (burnin = {} thinning = {})".format(nburn, nthin), fontsize=8
    )
    print("done")
    sys.stdout.flush()

    for i, n in enumerate(mod_names):
        idx = pnames.index(n)
        increase_y0 = (i + 1) * (height + space)
        yr = y0 + increase_y0

        print("-{} chain".format(n), end=" ")
        chains = sampler_dict["chain"][nburn + nthin - 1 : nsteps : nthin, :, idx]

        # posterior
        ax = fig.add_axes([x0, yr, width, height])
        ax.ticklabel_format(useOffset=False)
        ax.tick_params(labelsize=tk_font, labelbottom=False)
        ax.set_ylabel(r"{}".format(n), fontsize=l_font)
        print("posterior", end=" ")
        # plot chain
        ax.plot(
            chains, color="black", marker="None", ls="-", lw=0.4, alpha=0.33, zorder=9
        )
        # plot mle w/ error
        ax.axhspan(
            phdi_low[idx],
            phdi_upp[idx],
            fc="C0",
            ec="None",
            alpha=0.33,
            zorder=7,
            label="mle HDI68.27%",
        )
        ax.axhline(
            pmle[idx], color="C0", ls="-", lw=1.0, alpha=1.0, zorder=8, label="mle"
        )
        ax.axhline(
            pmle[idx] - phdi_err[idx],
            color="C0",
            ls="--",
            lw=1.0,
            alpha=1.0,
            zorder=8,
            label="mle+/-err",
        )
        ax.axhline(
            pmle[idx] + phdi_err[idx], color="C0", ls="--", lw=1.0, alpha=1.0, zorder=8
        )
        # plot median w/ error
        ax.axhspan(
            pci_16p[idx],
            pci_84p[idx],
            fc="C1",
            ec="None",
            alpha=0.33,
            zorder=7,
            label="CI(16p-84p)",
        )
        ax.axhline(
            pmedian[idx],
            color="C1",
            ls="-",
            lw=1.0,
            alpha=1.0,
            zorder=8,
            label="median",
        )
        ax.axhline(
            pmedian[idx] - pmedian_err[idx],
            color="C1",
            ls="--",
            lw=1.0,
            alpha=1.0,
            zorder=8,
            label="median+/-err",
        )
        ax.axhline(
            pmedian[idx] + pmedian_err[idx],
            color="C1",
            ls="--",
            lw=1.0,
            alpha=1.0,
            zorder=8,
        )
        print("done")
        sys.stdout.flush()

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.5), ncol=6, fontsize=6)

    print("-Saving transit models posterior png ...", end=" ")
    sys.stdout.flush()
    fig.savefig(
        output_folder.joinpath("08_trace_post_emcee_transit_models.png"),
        bbox_inches="tight",
    )
    # print('and pdf ... ', end=' ')
    # sys.stdout.flush()
    # fig.savefig(output_folder.joinpath('08_trace_post_emcee_transit_models.pdf'), bbox_inches='tight')
    plt.close(fig)
    print("done")
    sys.stdout.flush()

    # DETRENDING MODELS
    print("-Preparing figure for detrending models")
    sys.stdout.flush()
    # a4 paper = 8.3 x 11.7 in = 210 x 297 mm
    nplt = ndet
    fw, fh = 8.3, nplt * hplt
    if nplt > 12:
        print("--!!Figure will have huge height ({:.1f} in)!!".format(fh))

    # axes definition
    height_max_ax = height_max / nplt
    height = height_max_ax * 0.90
    space = height_max_ax * 0.10

    print("-Detrending models full trace plot")
    # full - Detrending models
    fig = plt.figure(figsize=(fw, fh))

    for i, n in enumerate(det_names):
        idx = pnames.index(n)
        increase_y0 = (i) * (height + space)
        yr = y0 + increase_y0

        print("-{} chain".format(n), end=" ")
        chains = sampler_dict["chain"][:, :, idx]
        # full
        ax = fig.add_axes([x0, yr, width, height])
        ax.ticklabel_format(useOffset=False)
        ax.tick_params(labelsize=tk_font, labelbottom=False)
        ax.set_ylabel(r"{}".format(n), fontsize=l_font)
        if i == 0:
            ax.tick_params(labelsize=tk_font, labelbottom=True)
            ax.set_xlabel("nsteps", fontsize=8)
        print("full", end=" ")
        # plot chain
        ax.plot(
            chains[:, :],
            color="black",
            marker="None",
            ls="-",
            lw=0.4,
            alpha=0.33,
            zorder=9,
        )
        # plot mle w/ error
        ax.axhspan(
            phdi_low[idx],
            phdi_upp[idx],
            fc="C0",
            ec="None",
            alpha=0.33,
            zorder=7,
            label="mle HDI68.27%",
        )
        ax.axhline(
            pmle[idx], color="C0", ls="-", lw=1.0, alpha=1.0, zorder=8, label="mle"
        )
        ax.axhline(
            pmle[idx] - phdi_err[idx],
            color="C0",
            ls="--",
            lw=1.0,
            alpha=1.0,
            zorder=8,
            label="mle+/-err",
        )
        ax.axhline(
            pmle[idx] + phdi_err[idx], color="C0", ls="--", lw=1.0, alpha=1.0, zorder=8
        )
        # plot median w/ error
        ax.axhspan(
            pci_16p[idx],
            pci_84p[idx],
            fc="C1",
            ec="None",
            alpha=0.33,
            zorder=7,
            label="CI(16p-84p)",
        )
        ax.axhline(
            pmedian[idx],
            color="C1",
            ls="-",
            lw=1.0,
            alpha=1.0,
            zorder=8,
            label="median",
        )
        ax.axhline(
            pmedian[idx] - pmedian_err[idx],
            color="C1",
            ls="--",
            lw=1.0,
            alpha=1.0,
            zorder=8,
            label="median+/-err",
        )
        ax.axhline(
            pmedian[idx] + pmedian_err[idx],
            color="C1",
            ls="--",
            lw=1.0,
            alpha=1.0,
            zorder=8,
        )
        # burnin
        ax.axvline(nburn, color="darkgray", ls="-", lw=1.0, alpha=1.0, zorder=8)
        print("done")
        sys.stdout.flush()

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.5), ncol=6, fontsize=6)

    print("-Saving detrending models full png ...", end=" ")
    sys.stdout.flush()
    fig.savefig(
        output_folder.joinpath("08_trace_full_emcee_detrending_models.png"),
        bbox_inches="tight",
    )
    # print('and pdf ... ', end=' ')
    # sys.stdout.flush()
    # fig.savefig(output_folder.joinpath('08_trace_full_emcee_detrending_models.pdf'), bbox_inches='tight')
    plt.close(fig)
    print("done")
    # gc.collect()

    print("-Detrending models full trace plot")
    fig = plt.figure(figsize=(fw, fh))

    for i, n in enumerate(det_names):
        idx = pnames.index(n)
        increase_y0 = (i) * (height + space)
        yr = y0 + increase_y0

        print("-{} chain".format(n), end=" ")
        chains = sampler_dict["chain"][nburn + nthin - 1 : nsteps : nthin, :, idx]

        # posterior
        ax = fig.add_axes([x0, yr, width, height])
        ax.ticklabel_format(useOffset=False)
        ax.tick_params(labelsize=tk_font, labelbottom=False)
        ax.set_ylabel(r"{}".format(n), fontsize=l_font)
        if i == 0:
            ax.set_xlabel(
                "nsteps posterior (burnin = {} thinning = {})".format(nburn, nthin),
                fontsize=8,
            )
            ax.tick_params(labelsize=tk_font, labelbottom=True)
        print("posterior", end=" ")
        # plot chain
        ax.plot(
            chains, color="black", marker="None", ls="-", lw=0.4, alpha=0.33, zorder=9
        )
        # plot mle w/ error
        ax.axhspan(
            phdi_low[idx],
            phdi_upp[idx],
            fc="C0",
            ec="None",
            alpha=0.33,
            zorder=7,
            label="mle HDI68.27%",
        )
        ax.axhline(
            pmle[idx], color="C0", ls="-", lw=1.0, alpha=1.0, zorder=8, label="mle"
        )
        ax.axhline(
            pmle[idx] - phdi_err[idx],
            color="C0",
            ls="--",
            lw=1.0,
            alpha=1.0,
            zorder=8,
            label="mle+/-err",
        )
        ax.axhline(
            pmle[idx] + phdi_err[idx], color="C0", ls="--", lw=1.0, alpha=1.0, zorder=8
        )
        # plot median w/ error
        ax.axhspan(
            pci_16p[idx],
            pci_84p[idx],
            fc="C1",
            ec="None",
            alpha=0.33,
            zorder=7,
            label="CI(16p-84p)",
        )
        ax.axhline(
            pmedian[idx],
            color="C1",
            ls="-",
            lw=1.0,
            alpha=1.0,
            zorder=8,
            label="median",
        )
        ax.axhline(
            pmedian[idx] - pmedian_err[idx],
            color="C1",
            ls="--",
            lw=1.0,
            alpha=1.0,
            zorder=8,
            label="median+/-err",
        )
        ax.axhline(
            pmedian[idx] + pmedian_err[idx],
            color="C1",
            ls="--",
            lw=1.0,
            alpha=1.0,
            zorder=8,
        )
        print("done")
        sys.stdout.flush()

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.5), ncol=6, fontsize=6)

    print("-Saving detrending models posterior png ...", end=" ")
    sys.stdout.flush()
    fig.savefig(
        output_folder.joinpath("08_trace_post_emcee_detrending_models.png"),
        bbox_inches="tight",
    )
    # print('and pdf ... ', end=' ')
    # sys.stdout.flush()
    # fig.savefig(output_folder.joinpath('08_trace_post_emcee_detrending_models.pdf'), bbox_inches='tight')
    plt.close(fig)
    print("done")
    sys.stdout.flush()

    return


# ======================================================================
# ======================================================================


def print_pycheops_pars(pars, user_data=True, expr=False):

    print("\n")
    header = "{:<16s} {:>6s} {:>16s} {:>16s} {:>16s} {:>16s}".format(
        "parameter", "vary", "value", "stderr", "min", "max"
    )
    if user_data:
        header += " {:>16s}".format("user_data")
    if expr:
        header += " {:>16s}".format("expr")
    print(header)

    for p in pars:
        vv = "{}".format(pars[p].vary)

        va = "{:16.6g}".format(pars[p].value)

        try:
            st = "{:16.6g}".format(pars[p].stderr)
        except:
            st = "0"

        mi = "{:16.6g}".format(pars[p].min)
        ma = "{:16.6g}".format(pars[p].max)

        line = "{:<16s} {:>6s} {:>s} {:>16s} {:>s} {:>s}".format(p, vv, va, st, mi, ma)

        if user_data:
            try:
                ud = "{}".format(pars[p].user_data)
            except:
                ud = "None"
            line += " {:>16s}".format(ud)

        if expr:
            try:
                ex = "{}".format(pars[p].expr)
            except:
                ex = "None"
            line += " {:>s}".format(ex)

        print(line)

    return


# ======================================================================
# ======================================================================


def print_analysis_stats(analysis_stats):
    print("")
    print(" ** SUMMARY **")
    h_st = "{:20s}".format("ANALYSIS_ID")
    h_lm = "{:14s} {:>14s} {:>14s} {:>14s}".format("LMFIT_TYPE", "CHISQ", "BIC", "RMS")
    h_em = "{:14s} {:>14s} {:>14s} {:>14s} {:8s}".format(
        "EMCEE", "CHISQ", "BIC", "RMS", "err_T0_s"
    )
    h_eg = "{:14s} {:>14s} {:>14s} {:>14s} {:8s}".format(
        "EMCEE_GP", "CHISQ", "BIC", "RMS", "err_T0_s"
    )
    head = "{:s} {:s} {:s} {:s}".format(h_st, h_lm, h_em, h_eg)
    print(head)
    for k_stats, v_stats in analysis_stats.items():
        fit_types = v_stats.keys()
        glint_fit = np.any([True if ("GLINT" in ft) else False for ft in fit_types])
        if glint_fit:
            lmk = "LMFIT w/ GLINT"
        else:
            lmk = "LMFIT"
        lmv = v_stats[lmk]["flux-all (w/o GP)"]

        emk = "EMCEE"
        emv = v_stats[emk]["flux-all (w/o GP)"]
        eT0e = v_stats["err_T0_s"]

        gpk = "EMCEE w/ GP"
        gpv = v_stats[gpk]["flux-all (w/ GP)"]
        eT0g = v_stats["err_T0_s w/ GP"]

        l_st = "{:20s}".format(k_stats)
        l_lm = "{:14s} {:14.4f} {:14.4f} {:14.4f}".format(
            lmk, lmv["ChiSqr"], lmv["BIC"], lmv["RMS (unbinned)"][0]
        )
        l_em = "{:14s} {:14.4f} {:14.4f} {:14.4f} {:8.1f}".format(
            emk, emv["ChiSqr"], emv["BIC"], emv["RMS (unbinned)"][0], eT0e
        )
        l_eg = "{:14s} {:14.4f} {:14.4f} {:14.4f} {:8.1f}".format(
            gpk, gpv["ChiSqr"], gpv["BIC"], gpv["RMS (unbinned)"][0], eT0g
        )
        line = "{:s} {:s} {:s} {:s}".format(l_st, l_lm, l_em, l_eg)
        print(line)
    print("")

    return


# ======================================================================
# ======================================================================


def plot_fft(dataset, pars_in, star=None, gsmooth=5, logxlim=(1.5, 4.5)):
    """

    Lomb-Scargle power-spectrum of the residuals.

    If the previous fit included a GP then this is _not_ included in the
    calculation of the residuals, i.e., the power spectrum includes the
    power "fitted-out" using the GP. The assumption here is that the GP
    has been used to model stellar variability that we wish to
    characterize using the power spectrum.

    The red vertical dotted lines show the CHEOPS  orbital frequency and
    its first two harmonics.

    If star is a pycheops starproperties object and star.teff is <7000K,
    then the likely range of nu_max is shown using green dashed lines.

    """

    time = np.array(dataset.lc["time"])
    flux = np.array(dataset.lc["flux"])
    flux_err = np.array(dataset.lc["flux_err"])

    model = get_full_model(dataset, params_in=pars_in, time=time)
    res = flux - model.all_nogp

    # print('nu_max = {:0.0f} muHz'.format(nu_max))
    t_s = time * 86400
    y = 1e6 * res
    ls = LombScargle(t_s, y, normalization="psd")
    frequency, power = ls.autopower()
    p_smooth = convolve(power, Gaussian1DKernel(gsmooth))

    fig, ax = plt.subplots()
    ax.loglog(frequency * 1e6, power / 1e6, c="gray", alpha=0.5)
    ax.loglog(frequency * 1e6, p_smooth / 1e6, c="darkcyan")
    # nu_max from Campante et al. (2016) eq (20)
    if star is not None:
        if star.teff < 7000:
            nu_max = 3090 * 10 ** (star.logg - 4.438) * um.sqrt(star.teff / 5777)
            ax.axvline(nu_max.n - nu_max.s, ls="--", c="C3")  # ,c='g')
            ax.axvline(nu_max.n + nu_max.s, ls="--", c="C3")  # ,c='g')
    f_cheops = 1e6 / (CHEOPS_ORBIT_MINUTES * 60)
    for h in range(1, 4):
        ax.axvline(h * f_cheops, ls=":", c="C2")  # ,c='darkred')
    ax.set_xlim(10 ** logxlim[0], 10 ** logxlim[1])
    ax.set_xlabel(r"Frequency [$\mu$Hz]")
    ax.set_ylabel("Power [ppm$^2$ $\mu$Hz$^{-1}$]")
    # ax.set_title(title)
    return fig


# ======================================================================
# ======================================================================
def multi_detrending_model(detrending_args):
    # ====================================================================
    # NOW LOOP IN THE DIFFERENT DETRENDING VERSIONS
    # AND CREATES A FOLDER FOR EACH ONE
    print("\nSTARTING ANALYSIS WITH DIFFERENT DETRENDING CRITERIA")

    analysis_stats = {}

    visit_folder = detrending_args["visit_folder"]
    file_key = detrending_args["file_key"]
    clipping = detrending_args["clipping"]
    star_name = detrending_args["star_name"]
    aperture = detrending_args["aperture"]
    # shape        = detrending_args['shape']
    analysis_todo = detrending_args["analysis_todo"]
    glint_type = detrending_args["glint_type"]
    in_par = detrending_args["in_par"]
    nwalkers = detrending_args["nwalkers"]
    nprerun = detrending_args["nprerun"]
    nsteps = detrending_args["nsteps"]
    nthin = detrending_args["nthin"]
    nburn = detrending_args["nburn"]
    progress = detrending_args["progress"]
    Mstar = detrending_args["Mstar"]
    Rstar = detrending_args["Rstar"]
    P = detrending_args["P"]
    Kms = detrending_args["Kms"]
    star = detrending_args["star"]

    if "all" in analysis_todo:
        detrending = detrend.copy()
    else:
        try:
            detrending = {}
            for k, v in detrend.items():
                if k in analysis_todo:
                    detrending[k] = v
            if len(detrending) == 0:
                print("len(detrending) == 0: all")
                detrending = detrend.copy()
        except:
            detrending = detrend.copy()

    print()
    print("===============================================================")
    print("Selected analysis (keywords):")
    print(detrending.keys())
    print("===============================================================")
    print()

    for analysis, det_par in detrending.items():

        print()
        print("=============================================================")
        print("=============================================================")
        print("ANALYSIS NUMBER {:s}".format(analysis))

        analysis_folder = visit_folder.joinpath("analysis_{:s}".format(analysis))
        print("-Creating folder {}".format(analysis_folder.resolve()))
        if not analysis_folder.is_dir():
            analysis_folder.mkdir(parents=True, exist_ok=True)

        analysis_id = "{}_id{}".format(star_name.replace(" ", "_"), analysis)
        analysis_stats[analysis_id] = {}

        print("-Reloading dataset ... ")
        dataset = Dataset(
            file_key=file_key,
            target=star_name,
            download_all=False,
            view_report_on_download=False,
        )
        _ = dataset.get_lightcurve(aperture=aperture, reject_highpoints=False)
        _ = dataset.clip_outliers(verbose=True)
        if clipping:
            print(
                "Further clipping on {} with {}-sigma wrt the median.".format(
                    clipping[0], clipping[1]
                )
            )
            if clipping[0] in dataset.lc.keys():
                x = dataset.lc[clipping[0]]
                mask = mask_data_clipping(x, clipping[1], clip_type="median")
                _ = dataset.mask_data(mask)
        lc = dataset.lc

        print("\n-Parameters set:")
        # in_par.pretty_print(colwidth=16, precision=6, fmt='g', columns=['value', 'vary', 'min', 'max', 'user_data'])
        print_pycheops_pars(in_par, user_data=True, expr=True)
        print("\n-Detrending set")
        for k, v in det_par.items():
            print("{:20s} = {}".format(k, v))

        # LMFIT-------------------------------------------------------------
        print("\n-LMFIT")
        # Fit with lmfit
        lmfit = dataset.lmfit_transit(
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
        )

        print(dataset.lmfit_report(min_correl=0.5))

        print("\n-Plot lmfit output")

        fig = dataset.plot_lmfit(figsize=(5, 5), binwidth=15 / 1440.0)
        #  plt.draw()
        fig.savefig(
            analysis_folder.joinpath(
                "{}_04_lc_lmfit_nodetrend.png".format(analysis_id)
            ),
            bbox_inches="tight",
        )
        plt.close(fig)

        fig = dataset.plot_lmfit(figsize=(5, 5), binwidth=15 / 1440.0, detrend=True)
        #  plt.draw()
        fig.savefig(
            analysis_folder.joinpath("{}_05_lc_lmfit_detrend.png".format(analysis_id)),
            bbox_inches="tight",
        )
        plt.close(fig)

        fig = dataset.rollangle_plot(figsize=(5, 5), fontsize=8)
        fig.savefig(
            analysis_folder.joinpath(
                "{}_06_roll_angle_vs_residual.png".format(analysis_id)
            ),
            bbox_inches="tight",
        )
        #  plt.draw()
        plt.close(fig)

        print("-PARAMS: LMFIT")
        params_lm = lmfit.params.copy()
        # params_lm.pretty_print(colwidth=20, precision=6, columns=['value', 'stderr', 'vary', 'min', 'max', 'expr'])
        print_pycheops_pars(params_lm, user_data=True, expr=True)
        dataset.gp = None  # force gp = None in the dataset
        stats_lm = computes_rms(dataset, params_best=params_lm, glint=False)
        analysis_stats[analysis_id]["LMFIT"] = stats_lm
        fig, _ = model_plot_fit(
            dataset,
            params_lm,
            par_type="lm",
            nsamples=0,
            flatchains=None,
            model_filename=analysis_folder.joinpath(
                "{}_05_lc_lmfit.dat".format(analysis_id)
            ),
        )
        fig.savefig(
            analysis_folder.joinpath("{}_05_lc_lmfit.png".format(analysis_id)),
            bbox_inches="tight",
        )
        plt.close(fig)

        # check if detrend has not None glint
        # WARNING: MODIFY IT PROPERLY
        if det_par["glint_scale"] is not None:
            if glint_type == "moon":
                _ = dataset.add_glint(show_plot=False, moon=True)
            # _ = dataset.add_glint(moon=True, nspline=21, binwidth=7)
            else:
                _ = dataset.add_glint(show_plot=False)

            # LMFIT w/ GLINT--------------------------------------------------
            print("\n-LMFIT with GLINT")
            lmfit = dataset.lmfit_transit(
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
                glint_scale=det_par["glint_scale"],
            )

            print(dataset.lmfit_report(min_correl=0.5))

            print("\n-Plot lmfit output")
            fig = dataset.plot_lmfit(binwidth=15 / 1440.0)
            #  plt.draw()
            fig.savefig(
                analysis_folder.joinpath(
                    "{}_04_lc_lmfit_nodetrend_glint.png".format(analysis_id)
                ),
                bbox_inches="tight",
            )
            plt.close(fig)

            fig = dataset.plot_lmfit(binwidth=15 / 1440.0, detrend=True)
            #  plt.draw()
            fig.savefig(
                analysis_folder.joinpath(
                    "{}_05_lc_lmfit_detrend_glint.png".format(analysis_id)
                ),
                bbox_inches="tight",
            )
            plt.close(fig)

            fig = dataset.rollangle_plot(figsize=(6, 4), fontsize=8)
            #  plt.draw()
            fig.savefig(
                analysis_folder.joinpath(
                    "{}_06_roll_angle_vs_residual_glint.png".format(analysis_id)
                ),
                bbox_inches="tight",
            )
            plt.close(fig)

            print("-PARAMS: LMFIT w/ GLINT")
            params_lm = lmfit.params.copy()
            # params_lm.pretty_print(colwidth=20, precision=6, columns=['value', 'stderr', 'vary', 'min', 'max', 'expr'])
            print_pycheops_pars(params_lm, user_data=True, expr=True)
            dataset.gp = None  # force gp = None in the dataset
            stats_lm_glint = computes_rms(dataset, params_best=params_lm, glint=False)
            analysis_stats[analysis_id]["LMFIT w/ GLINT"] = stats_lm_glint
            fig, _ = model_plot_fit(
                dataset,
                params_lm,
                par_type="lm+glint",
                nsamples=0,
                flatchains=None,
                model_filename=analysis_folder.joinpath(
                    "{}_05_lc_lmfit_glint.dat".format(analysis_id)
                ),
            )
            fig.savefig(
                analysis_folder.joinpath(
                    "{}_05_lc_lmfit_glint.png".format(analysis_id)
                ),
                bbox_inches="tight",
            )
            plt.close(fig)
        # END IF GLINT

        # ===== EMCEE =====
        # Run emcee from last best fit
        print("\n-Run emcee from last best fit with:")
        print(" nwalkers = {}".format(nwalkers))
        print(" nprerun  = {}".format(nprerun))
        print(" nsteps   = {}".format(nsteps))
        print(" nburn    = {}".format(nburn))
        print(" nthin    = {}".format(nthin))

        # EMCEE-------------------------------------------------------------
        result = dataset.emcee_sampler(
            params=params_lm,
            nwalkers=nwalkers,
            burn=nprerun,
            steps=nsteps,
            thin=nthin,
            add_shoterm=False,
            progress=progress,
        )

        print(dataset.emcee_report(min_correl=0.5))

        title = "emcee model (no gp)"
        print("\n-Plot pycheops {}".format(title))
        fig = dataset.plot_emcee(
            title=title,
            figsize=(5, 5),
            fontsize=10,
            nsamples=64,
            binwidth=15 / 1440.0,
            detrend=False,
        )
        fig.savefig(
            analysis_folder.joinpath(
                "{}_07_lc_emcee_nodetrend.png".format(analysis_id)
            ),
            bbox_inches="tight",
        )
        #  plt.draw()
        plt.close(fig)
        fig = dataset.plot_emcee(
            title=title,
            figsize=(5, 5),
            fontsize=10,
            nsamples=64,
            binwidth=15 / 1440.0,
            detrend=True,
        )
        fig.savefig(
            analysis_folder.joinpath("{}_07_lc_emcee_detrend.png".format(analysis_id)),
            bbox_inches="tight",
        )
        #  plt.draw()
        plt.close(fig)
        fig = dataset.rollangle_plot(figsize=(5, 5), fontsize=8)
        fig.savefig(
            analysis_folder.joinpath(
                "{}_07_emcee_roll_angle_vs_residual.png".format(analysis_id)
            ),
            bbox_inches="tight",
        )
        #  plt.draw()
        plt.close(fig)

        print("\n-Plot trace of the chains")
        fig = dataset.trail_plot("all")  # add 'all' for all traces!
        #  plt.draw()
        fig.savefig(
            analysis_folder.joinpath("{}_08_trace_emcee_all.png".format(analysis_id)),
            bbox_inches="tight",
        )
        plt.close(fig)

        print("\n-Plot corner full from pycheops (not removed nburn)")
        fig = dataset.corner_plot(plotkeys="all")
        #  plt.draw()
        fig.savefig(
            analysis_folder.joinpath("{}_09_corner_emcee_all.png".format(analysis_id)),
            bbox_inches="tight",
        )
        plt.close(fig)

        print("\n-Computing my parameters and plot models with random samples")
        params_med, stats_med, params_best, stats_mle = get_best_parameters(
            result, dataset, nburn=nburn
        )
        analysis_stats[analysis_id]["EMCEE"] = stats_mle
        # update emcee.params -> median and emcee.params_best -> mle
        for p in dataset.emcee.params:
            dataset.emcee.params[p] = params_med[p]
            dataset.emcee.params_best[p] = params_best[p]

        fig, _ = model_plot_fit(
            dataset,
            params_med,
            par_type="median",
            nsamples=nwalkers,
            flatchains=result.chain,
            model_filename=analysis_folder.joinpath(
                "{}_07_lc_emcee_median.dat".format(analysis_id)
            ),
        )
        fig.savefig(
            analysis_folder.joinpath("{}_07_lc_emcee_median.png".format(analysis_id)),
            bbox_inches="tight",
        )
        plt.close(fig)
        fig = plot_fft(dataset, params_med, star=star)
        fig.savefig(
            analysis_folder.joinpath("{}_07_fft_emcee_median.png".format(analysis_id)),
            bbox_inches="tight",
        )
        plt.close(fig)

        fig, _ = model_plot_fit(
            dataset,
            params_best,
            par_type="mle",
            nsamples=nwalkers,
            flatchains=result.chain,
            model_filename=analysis_folder.joinpath(
                "{}_07_lc_emcee_mle.dat".format(analysis_id)
            ),
        )
        fig.savefig(
            analysis_folder.joinpath("{}_07_lc_emcee_mle.png".format(analysis_id)),
            bbox_inches="tight",
        )
        plt.close(fig)
        fig = plot_fft(dataset, params_best, star=star)
        fig.savefig(
            analysis_folder.joinpath("{}_07_fft_emcee_mle.png".format(analysis_id)),
            bbox_inches="tight",
        )
        plt.close(fig)

        print("-massaradius with MEDIAN")
        T0_best = ufloat(params_med["T_0"].value, params_med["T_0"].stderr)
        BJD_best = T0_best + lc["bjd_ref"]
        si = ufloat(params_med["sini"].value, params_med["sini"].stderr)
        inc_best = ufloat(params_med["inc"].value, params_med["inc"].stderr)
        W_best = ufloat(params_med["W"].value, params_med["W"].stderr) * P
        W_h, W_m = W_best * cst.day2hour, W_best * cst.day2min
        k = ufloat(params_med["k"].value, params_med["k"].stderr)
        aRs = ufloat(params_med["aR"].value, params_med["aR"].stderr)

        print(
            " T0     = {:.6f} +/- {:.6f} days = {:.6f}+/-{:.6f} BJD_TDB".format(
                T0_best.n, T0_best.s, BJD_best.n, BJD_best.s
            )
        )
        print(
            " err_T0 = {:.3f} m = {:.1f} s".format(
                T0_best.s * cst.day2min, T0_best.s * cst.day2sec
            )
        )
        print(" inc    = {:.6f} +/- {:.6f} deg".format(inc_best.n, inc_best.s))
        print(
            " dur    = {:.6f} +/- {:.6f} days = {:.6f} +/- {:.6f} hour = {:.6f} +/- {:.6f} min".format(
                W_best.n, W_best.s, W_h.n, W_h.s, W_m.n, W_m.s
            )
        )
        print(
            " b      = {:.4f} +/- {:.4f}".format(
                params_med["b"].value, params_med["b"].stderr
            )
        )
        print(" k      = {:.6f} +/- {:.6f}".format(k.n, k.s))
        print(" aRs    = {:.6f} +/- {:.6f}".format(aRs.n, aRs.s))

        try:
            print("\n-trying with tepcat=True")
            _, fig = massradius(
                m_star=Mstar,
                r_star=Rstar,
                k=k,
                K=Kms,
                aR=aRs,
                sini=si,
                P=P,
                jovian=True,
                tepcat=True,
                figsize=(5, 5),
            )
        except:
            print("\n-Error with tepcat=True: using tepcat=False")
            _, fig = massradius(
                m_star=Mstar,
                r_star=Rstar,
                k=k,
                K=Kms,
                aR=aRs,
                sini=si,
                P=P,
                jovian=True,
                tepcat=False,
                figsize=(5, 5),
            )
        #  plt.draw()
        fig.savefig(
            analysis_folder.joinpath("{}_10_massradius_median.png".format(analysis_id)),
            bbox_inches="tight",
        )
        plt.close(fig)
        print(" Saved massradius median")

        print("-massradius with MLE")
        T0_best = ufloat(params_best["T_0"].value, params_best["T_0"].stderr)
        BJD_best = T0_best + lc["bjd_ref"]
        si = ufloat(params_best["sini"].value, params_best["sini"].stderr)
        inc_best = ufloat(params_best["inc"].value, params_best["inc"].stderr)
        W_best = ufloat(params_best["W"].value, params_best["W"].stderr) * P
        W_h, W_m = W_best * cst.day2hour, W_best * cst.day2min
        k = ufloat(params_best["k"].value, params_best["k"].stderr)
        aRs = ufloat(params_best["aR"].value, params_best["aR"].stderr)

        print(
            " T0     = {:.6f} +/- {:.6f} days = {:.6f}+/-{:.6f} BJD_TDB".format(
                T0_best.n, T0_best.s, BJD_best.n, BJD_best.s
            )
        )
        print(
            " err_T0 = {:.3f} m = {:.1f} s".format(
                T0_best.s * cst.day2min, T0_best.s * cst.day2sec
            )
        )
        print(" inc    = {:.6f} +/- {:.6f} deg".format(inc_best.n, inc_best.s))
        print(
            " dur    = {:.6f} +/- {:.6f} days = {:.6f} +/- {:.6f} hour = {:.6f} +/- {:.6f} min".format(
                W_best.n, W_best.s, W_h.n, W_h.s, W_m.n, W_m.s
            )
        )
        print(
            " b      = {:.4f} +/- {:.4f}".format(
                params_best["b"].value, params_best["b"].stderr
            )
        )
        print(" k      = {:.6f} +/- {:.6f}".format(k.n, k.s))
        print(" aRs    = {:.6f} +/- {:.6f}".format(aRs.n, aRs.s))

        analysis_stats[analysis_id]["err_T0_s"] = T0_best.s * cst.day2sec

        try:
            print("\n-trying with tepcat=True")
            _, fig = massradius(
                m_star=Mstar,
                r_star=Rstar,
                k=k,
                K=Kms,
                aR=aRs,
                sini=si,
                P=P,
                jovian=True,
                tepcat=True,
                figsize=(5, 5),
            )
        except:
            print("\n-Error with tepcat=True: using tepcat=False")
            _, fig = massradius(
                m_star=Mstar,
                r_star=Rstar,
                k=k,
                K=Kms,
                aR=aRs,
                sini=si,
                P=P,
                jovian=True,
                tepcat=False,
                figsize=(5, 5),
            )
        #  plt.draw()
        fig.savefig(
            analysis_folder.joinpath("{}_10_massradius_mle.png".format(analysis_id)),
            bbox_inches="tight",
        )
        plt.close(fig)
        print(" Saved massradius mle")

        file_emcee = save_dataset(
            dataset, analysis_folder.resolve(), star_name, file_key, gp=False
        )
        print("-Dumped dataset into file {}".format(file_emcee))

        # EMCEE WITH GAUSSIAN PROCESS 1-------------------------------------
        # Run emcee with SHOTerm kernel for Gaussian Process
        print(
            "\n-Run emcee with GP-SHOTerm with parameters as median of previous posterior distribution"
        )

        # if(shape == 'fix'):
        #   result_gp = dataset.emcee_sampler(
        #     params      = params_med,
        #     nwalkers    = nwalkers,
        #     burn        = nprerun,
        #     steps       = nsteps,
        #     thin        = nthin,
        #     add_shoterm = True,
        #     progress    = False
        #   )

        # else: # shape == 'fit'

        print("*FIRST: we have to train the gp noise, so I fix the transit parameters")
        # Fix the transit parameters
        params_fixed = copy_parameters(params_best)
        for p in ["T_0", "D", "W", "b"]:
            params_fixed[p].set(vary=False)

        result_gp = dataset.emcee_sampler(
            params=params_fixed,
            nwalkers=nwalkers,
            burn=nprerun,
            steps=nsteps,
            thin=nthin,
            add_shoterm=True,
            progress=progress,
        )

        print("\n-Computing my parameters and plot models with random samples")
        params_med_gp, _, params_best_gp, _ = get_best_parameters(
            result_gp, dataset, nburn=nburn
        )
        # analysis_stats[analysis_id]["EMCEE w/ GP"] = stats_med_gp

        fig, _ = model_plot_fit(
            dataset,
            params_med_gp,
            par_type="median",
            nsamples=nwalkers,
            flatchains=result_gp.chain,
            model_filename=analysis_folder.joinpath(
                "{}_07_lc_emcee_median_gp_train.dat".format(analysis_id)
            ),
        )
        fig.savefig(
            analysis_folder.joinpath(
                "{}_07_lc_emcee_median_gp_train.png".format(analysis_id)
            ),
            bbox_inches="tight",
        )
        plt.close(fig)
        fig, _ = model_plot_fit(
            dataset,
            params_best_gp,
            par_type="mle",
            nsamples=nwalkers,
            flatchains=result_gp.chain,
            model_filename=analysis_folder.joinpath(
                "{}_07_lc_emcee_mle_gp_train.dat".format(analysis_id)
            ),
        )
        fig.savefig(
            analysis_folder.joinpath(
                "{}_07_lc_emcee_mle_gp_train.png".format(analysis_id)
            ),
            bbox_inches="tight",
        )
        plt.close(fig)

        # EMCEE WITH GAUSSIAN PROCESS 2-------------------------------------
        print(
            "\n*SECOND: we have to use priors on gp noise, and set to fit again the transit parameters"
        )
        params_fit_gp = copy_parameters(params_best_gp)
        # set priors for gp
        for p in ["log_S0", "log_omega0", "log_sigma"]:
            params_fit_gp[p].user_data = ufloat(
                params_best_gp[p].value, 2 * params_best_gp[p].stderr
            )
        # Restoring the transit model parameters as free parameters
        # for p in ['T_0','D','W','b']:
        # params_fit_gp[p].set(vary=True)
        for p in in_par:
            params_fit_gp[p].set(
                vary=params_best[p].vary, min=in_par[p].min, max=in_par[p].max
            )
            params_fit_gp[p].stderr = params_best[p].stderr

        result_gp = dataset.emcee_sampler(
            params=params_fit_gp,
            nwalkers=nwalkers,
            burn=nprerun,
            steps=nsteps,
            thin=nthin,
            # add_shoterm = True, # not needed the second time
            progress=progress,
        )

        print(dataset.emcee_report(min_correl=0.5))

        title = "emcee model (gp)"
        print("\n-Plot pycheops {}".format(title))
        fig = dataset.plot_emcee(
            title=title,
            figsize=(5, 5),
            fontsize=10,
            nsamples=64,
            binwidth=15 / 1440.0,
            detrend=False,
        )
        fig.savefig(
            analysis_folder.joinpath(
                "{}_07_lc_emcee_nodetrend_gp.png".format(analysis_id)
            ),
            bbox_inches="tight",
        )
        plt.close(fig)
        fig = dataset.plot_emcee(
            title=title,
            figsize=(5, 5),
            fontsize=10,
            nsamples=64,
            binwidth=15 / 1440.0,
            detrend=True,
        )
        fig.savefig(
            analysis_folder.joinpath(
                "{}_07_lc_emcee_detrend_gp.png".format(analysis_id)
            ),
            bbox_inches="tight",
        )
        plt.close(fig)
        fig = dataset.rollangle_plot(figsize=(5, 5), fontsize=8)
        fig.savefig(
            analysis_folder.joinpath(
                "{}_07_emcee_roll_angle_vs_residual_gp.png".format(analysis_id)
            ),
            bbox_inches="tight",
        )
        plt.close(fig)

        print("\n-Plot trace of the chains")
        fig = dataset.trail_plot("all")  # add 'all' for all traces!
        #  plt.draw()
        fig.savefig(
            analysis_folder.joinpath(
                "{}_08_trace_emcee_all_gp.png".format(analysis_id)
            ),
            bbox_inches="tight",
        )
        plt.close(fig)

        print("\n-Plot corner full from pycheops (not removed nburn)")
        fig = dataset.corner_plot(plotkeys="all")
        #  plt.draw()
        fig.savefig(
            analysis_folder.joinpath(
                "{}_09_corner_emcee_all_gp.png".format(analysis_id)
            ),
            bbox_inches="tight",
        )
        plt.close(fig)

        print("\n-Computing my parameters and plot models with random samples")
        params_med_gp, stats_med_gp, params_best_gp, stats_mle_gp = get_best_parameters(
            result_gp, dataset, nburn=nburn
        )
        analysis_stats[analysis_id]["EMCEE w/ GP"] = stats_mle_gp
        # update emcee.params -> median and emcee.params_best -> mle
        for p in dataset.emcee.params:
            dataset.emcee.params[p] = params_med_gp[p]
            dataset.emcee.params_best[p] = params_best_gp[p]

        fig, _ = model_plot_fit(
            dataset,
            params_med_gp,
            par_type="median",
            nsamples=nwalkers,
            flatchains=result_gp.chain,
            model_filename=analysis_folder.joinpath(
                "{}_07_lc_emcee_median_gp.dat".format(analysis_id)
            ),
        )
        fig.savefig(
            analysis_folder.joinpath(
                "{}_07_lc_emcee_median_gp.png".format(analysis_id)
            ),
            bbox_inches="tight",
        )
        plt.close(fig)
        fig, _ = model_plot_fit(
            dataset,
            params_best_gp,
            par_type="mle",
            nsamples=nwalkers,
            flatchains=result_gp.chain,
            model_filename=analysis_folder.joinpath(
                "{}_07_lc_emcee_mle_gp.dat".format(analysis_id)
            ),
        )
        fig.savefig(
            analysis_folder.joinpath("{}_07_lc_emcee_mle_gp.png".format(analysis_id)),
            bbox_inches="tight",
        )
        plt.close(fig)

        print("-massaradius with MEDIAN")
        T0_best = ufloat(params_med_gp["T_0"].value, params_med_gp["T_0"].stderr)
        BJD_best = T0_best + lc["bjd_ref"]
        si = ufloat(params_med_gp["sini"].value, params_med_gp["sini"].stderr)
        inc_best = ufloat(params_med_gp["inc"].value, params_med_gp["inc"].stderr)
        W_best = ufloat(params_med_gp["W"].value, params_med_gp["W"].stderr) * P
        W_h, W_m = W_best * cst.day2hour, W_best * cst.day2min
        k = ufloat(params_med_gp["k"].value, params_med_gp["k"].stderr)
        aRs = ufloat(params_med_gp["aR"].value, params_med_gp["aR"].stderr)

        print(
            " T0     = {:.6f} +/- {:.6f} days = {:.6f}+/-{:.6f} BJD_TDB".format(
                T0_best.n, T0_best.s, BJD_best.n, BJD_best.s
            )
        )
        print(
            " err_T0 = {:.3f} m = {:.1f} s".format(
                T0_best.s * cst.day2min, T0_best.s * cst.day2sec
            )
        )
        print(" inc    = {:.6f} +/- {:.6f} deg".format(inc_best.n, inc_best.s))
        print(
            " dur    = {:.6f} +/- {:.6f} days = {:.6f} +/- {:.6f} hour = {:.6f} +/- {:.6f} min".format(
                W_best.n, W_best.s, W_h.n, W_h.s, W_m.n, W_m.s
            )
        )
        print(
            " b      = {:.4f} +/- {:.4f}".format(
                params_med_gp["b"].value, params_med_gp["b"].stderr
            )
        )
        print(" k      = {:.6f} +/- {:.6f}".format(k.n, k.s))
        print(" aRs    = {:.6f} +/- {:.6f}".format(aRs.n, aRs.s))

        try:
            print("\n-trying with tepcat=True")
            _, fig = massradius(
                m_star=Mstar,
                r_star=Rstar,
                k=k,
                K=Kms,
                aR=aRs,
                sini=si,
                P=P,
                jovian=True,
                tepcat=True,
                figsize=(5, 5),
            )
        except:
            print("\n-Error with tepcat=True: using tepcat=False")
            _, fig = massradius(
                m_star=Mstar,
                r_star=Rstar,
                k=k,
                K=Kms,
                aR=aRs,
                sini=si,
                P=P,
                jovian=True,
                tepcat=False,
                figsize=(5, 5),
            )
        #  plt.draw()
        fig.savefig(
            analysis_folder.joinpath(
                "{}_10_massradius_median_gp.png".format(analysis_id)
            ),
            bbox_inches="tight",
        )
        plt.close(fig)
        print(" Saved massradius median with gp")

        print("-massradius with MLE")
        T0_best = ufloat(params_best_gp["T_0"].value, params_best_gp["T_0"].stderr)
        BJD_best = T0_best + lc["bjd_ref"]
        si = ufloat(params_best_gp["sini"].value, params_best_gp["sini"].stderr)
        inc_best = ufloat(params_best_gp["inc"].value, params_best_gp["inc"].stderr)
        W_best = ufloat(params_best_gp["W"].value, params_best_gp["W"].stderr) * P
        W_h, W_m = W_best * cst.day2hour, W_best * cst.day2min
        k = ufloat(params_best_gp["k"].value, params_best_gp["k"].stderr)
        aRs = ufloat(params_best_gp["aR"].value, params_best_gp["aR"].stderr)

        print(
            " T0     = {:.6f} +/- {:.6f} days = {:.6f}+/-{:.6f} BJD_TDB".format(
                T0_best.n, T0_best.s, BJD_best.n, BJD_best.s
            )
        )
        print(
            " err_T0 = {:.3f} m = {:.1f} s".format(
                T0_best.s * cst.day2min, T0_best.s * cst.day2sec
            )
        )
        print(" inc    = {:.6f} +/- {:.6f} deg".format(inc_best.n, inc_best.s))
        print(
            " dur    = {:.6f} +/- {:.6f} days = {:.6f} +/- {:.6f} hour = {:.6f} +/- {:.6f} min".format(
                W_best.n, W_best.s, W_h.n, W_h.s, W_m.n, W_m.s
            )
        )
        print(
            " b      = {:.4f} +/- {:.4f}".format(
                params_best_gp["b"].value, params_best_gp["b"].stderr
            )
        )
        print(" k      = {:.6f} +/- {:.6f}".format(k.n, k.s))
        print(" aRs    = {:.6f} +/- {:.6f}".format(aRs.n, aRs.s))

        analysis_stats[analysis_id]["err_T0_s w/ GP"] = T0_best.s * cst.day2sec

        try:
            print("\n-trying with tepcat=True")
            _, fig = massradius(
                m_star=Mstar,
                r_star=Rstar,
                k=k,
                K=Kms,
                aR=aRs,
                sini=si,
                P=P,
                jovian=True,
                tepcat=True,
                figsize=(5, 5),
            )
        except:
            print("\n-Error with tepcat=True: using tepcat=False")
            _, fig = massradius(
                m_star=Mstar,
                r_star=Rstar,
                k=k,
                K=Kms,
                aR=aRs,
                sini=si,
                P=P,
                jovian=True,
                tepcat=False,
                figsize=(5, 5),
            )

        #  plt.draw()
        fig.savefig(
            analysis_folder.joinpath("{}_10_massradius_mle_gp.png".format(analysis_id)),
            bbox_inches="tight",
        )
        plt.close(fig)
        print(" Saved massradius mle with gp")

        file_emcee_gp = save_dataset(
            dataset, analysis_folder.resolve(), star_name, file_key, gp=True
        )
        print("-Dumped dataset into file {}".format(file_emcee_gp))

        print("ANALYSIS {} DONE".format(analysis))
        plt.close("all")
        print("=============================================================")

    # END LOOP ON ANALYSIS
    return analysis_stats


# ======================================================================
# ======================================================================
def should_I_decorr_detrending_model(detrending_args):
    # ====================================================================
    # NOW LOOP IN THE DIFFERENT DETRENDING VERSIONS
    # AND CREATES A FOLDER FOR EACH ONE
    print("\nSTARTING ANALYSIS WITH DIFFERENT DETRENDING CRITERIA")

    analysis_stats = {}

    visit_folder = detrending_args["visit_folder"]
    file_key = detrending_args["file_key"]
    glint_type = detrending_args["glint_type"]
    clipping = detrending_args["clipping"]
    star_name = detrending_args["star_name"]
    aperture = detrending_args["aperture"]
    in_par = detrending_args["in_par"]
    nwalkers = detrending_args["nwalkers"]
    nprerun = detrending_args["nprerun"]
    nsteps = detrending_args["nsteps"]
    nthin = detrending_args["nthin"]
    nburn = detrending_args["nburn"]
    progress = detrending_args["progress"]
    Mstar = detrending_args["Mstar"]
    Rstar = detrending_args["Rstar"]
    P = detrending_args["P"]
    T_0 = detrending_args["T_0"]
    Kms = detrending_args["Kms"]
    star = detrending_args["star"]

    glint_list = [False]
    if glint_type in ["moon", "glint"]:
        glint_list += [glint_type]

    for glint_value in glint_list:
        print("\nRunning analysis decorr with glint: {}".format(glint_value))

        print("Loading dataset ... ")
        dataset = Dataset(
            file_key=file_key,
            target=star_name,
            download_all=False,
            view_report_on_download=False,
        )
        _ = dataset.get_lightcurve(aperture=aperture, reject_highpoints=False)
        _ = dataset.clip_outliers(verbose=True)
        if clipping:
            print(
                "Further clipping on {} with {}-sigma wrt the median.".format(
                    clipping[0], clipping[1]
                )
            )
            if clipping[0] in dataset.lc.keys():
                x = dataset.lc[clipping[0]]
                mask = mask_data_clipping(x, clipping[1], clip_type="median")
                _ = dataset.mask_data(mask)

        mask_centre_value = T_0[1]
        mask_width_value = T_0[2] - T_0[0]
        _, decorr_params = dataset.should_I_decorr(mask_centre_value, mask_width_value)

        print()
        print("===============================================================")
        print("Selected analysis (keywords):")
        print(decorr_params)
        print("===============================================================")
        print()

        if len(decorr_params) == 0:
            print("NO DETRENDING PARAMETERS TO USE")

        analysis = "decorr"
        glint_str = ""
        if glint_value:
            glint_str = "{}".format(glint_value.lower())
            analysis = "decorr_{}".format(glint_str)
        analysis_folder = visit_folder.joinpath("analysis_{:s}".format(analysis))
        print("-Creating folder {}".format(analysis_folder.resolve()))
        if not analysis_folder.is_dir():
            analysis_folder.mkdir(parents=True, exist_ok=True)

        analysis_id = "{}_id{}".format(star_name.replace(" ", "_"), analysis)
        analysis_stats[analysis_id] = {}

        lc = dataset.lc

        print("\n-Parameters set:")
        # in_par.pretty_print(colwidth=16, precision=6, fmt='g', columns=['value', 'vary', 'min', 'max', 'user_data'])
        print_pycheops_pars(in_par, user_data=True, expr=True)
        print("\n-Detrending set")
        det_par = detrend_default.copy()
        for k, v in detrend["07b"].items():
            if k in decorr_params:
                det_par[k] = v
        # if(glint_type in ['moon', 'glint']):
        #   det_par['glint_scale'] = detrend["07b"]["glint_scale"]
        for k, v in det_par.items():
            print("{:20s} = {}".format(k, v))

        # LMFIT-------------------------------------------------------------
        print("\n-LMFIT")
        # Fit with lmfit
        lmfit = dataset.lmfit_transit(
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
        )

        print(dataset.lmfit_report(min_correl=0.5))

        print("\n-Plot lmfit output")

        fig = dataset.plot_lmfit(figsize=(5, 5), binwidth=15 / 1440.0)
        #  plt.draw()
        fig.savefig(
            analysis_folder.joinpath(
                "{}_04_lc_lmfit_nodetrend.png".format(analysis_id)
            ),
            bbox_inches="tight",
        )
        plt.close(fig)

        fig = dataset.plot_lmfit(figsize=(5, 5), binwidth=15 / 1440.0, detrend=True)
        #  plt.draw()
        fig.savefig(
            analysis_folder.joinpath("{}_05_lc_lmfit_detrend.png".format(analysis_id)),
            bbox_inches="tight",
        )
        plt.close(fig)

        fig = dataset.rollangle_plot(figsize=(5, 5), fontsize=8)
        fig.savefig(
            analysis_folder.joinpath(
                "{}_06_roll_angle_vs_residual.png".format(analysis_id)
            ),
            bbox_inches="tight",
        )
        #  plt.draw()
        plt.close(fig)

        print("-PARAMS: LMFIT")
        params_lm = lmfit.params.copy()
        # params_lm.pretty_print(colwidth=20, precision=6, columns=['value', 'stderr', 'vary', 'min', 'max', 'expr'])
        print_pycheops_pars(params_lm, user_data=True, expr=True)
        dataset.gp = None  # force gp = None in the dataset
        stats_lm = computes_rms(dataset, params_best=params_lm, glint=False)
        analysis_stats[analysis_id]["LMFIT"] = stats_lm
        fig, _ = model_plot_fit(
            dataset,
            params_lm,
            par_type="lm",
            nsamples=0,
            flatchains=None,
            model_filename=analysis_folder.joinpath(
                "{}_05_lc_lmfit.dat".format(analysis_id)
            ),
        )
        fig.savefig(
            analysis_folder.joinpath("{}_05_lc_lmfit.png".format(analysis_id)),
            bbox_inches="tight",
        )
        plt.close(fig)

        # check if detrend has not None glint
        # WARNING: MODIFY IT PROPERLY
        if glint_value:
            det_par["glint_scale"] = detrend["07b"]["glint_scale"]
            if glint_value == "moon":
                # _ = dataset.add_glint(moon=True, nspline=21, binwidth=7)
                _ = dataset.add_glint(
                    moon=True, show_plot=False, nspline=21, binwidth=7
                )
            else:
                _ = dataset.add_glint(show_plot=False, nspline=21, binwidth=7)

            # LMFIT w/ GLINT--------------------------------------------------
            print("\n-LMFIT with GLINT")
            lmfit = dataset.lmfit_transit(
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
                glint_scale=det_par["glint_scale"],
            )

            print(dataset.lmfit_report(min_correl=0.5))

            print("\n-Plot lmfit output")
            fig = dataset.plot_lmfit(binwidth=15 / 1440.0)
            #  plt.draw()
            fig.savefig(
                analysis_folder.joinpath(
                    "{}_04_lc_lmfit_nodetrend_glint.png".format(analysis_id)
                ),
                bbox_inches="tight",
            )
            plt.close(fig)

            fig = dataset.plot_lmfit(binwidth=15 / 1440.0, detrend=True)
            #  plt.draw()
            fig.savefig(
                analysis_folder.joinpath(
                    "{}_05_lc_lmfit_detrend_glint.png".format(analysis_id)
                ),
                bbox_inches="tight",
            )
            plt.close(fig)

            fig = dataset.rollangle_plot(figsize=(6, 4), fontsize=8)
            #  plt.draw()
            fig.savefig(
                analysis_folder.joinpath(
                    "{}_06_roll_angle_vs_residual_glint.png".format(analysis_id)
                ),
                bbox_inches="tight",
            )
            plt.close(fig)

            print("-PARAMS: LMFIT w/ GLINT")
            params_lm = lmfit.params.copy()
            # params_lm.pretty_print(colwidth=20, precision=6, columns=['value', 'stderr', 'vary', 'min', 'max', 'expr'])
            print_pycheops_pars(params_lm, user_data=True, expr=True)
            dataset.gp = None  # force gp = None in the dataset
            stats_lm_glint = computes_rms(dataset, params_best=params_lm, glint=False)
            analysis_stats[analysis_id]["LMFIT w/ GLINT"] = stats_lm_glint
            fig, _ = model_plot_fit(
                dataset,
                params_lm,
                par_type="lm+glint",
                nsamples=0,
                flatchains=None,
                model_filename=analysis_folder.joinpath(
                    "{}_05_lc_lmfit_glint.dat".format(analysis_id)
                ),
            )
            fig.savefig(
                analysis_folder.joinpath(
                    "{}_05_lc_lmfit_glint.png".format(analysis_id)
                ),
                bbox_inches="tight",
            )
            plt.close(fig)
        # END IF GLINT

        # ===== EMCEE =====
        # Run emcee from last best fit
        print("\n-Run emcee from last best fit with:")
        print(" nwalkers = {}".format(nwalkers))
        print(" nprerun  = {}".format(nprerun))
        print(" nsteps   = {}".format(nsteps))
        print(" nburn    = {}".format(nburn))
        print(" nthin    = {}".format(nthin))
        print()

        # EMCEE-------------------------------------------------------------
        result = dataset.emcee_sampler(
            params=params_lm,
            nwalkers=nwalkers,
            burn=nprerun,
            steps=nsteps,
            thin=nthin,
            add_shoterm=False,
            progress=progress,
        )

        print(dataset.emcee_report(min_correl=0.5))

        title = "emcee model (no gp)"
        print("\n-Plot pycheops {}".format(title))
        fig = dataset.plot_emcee(
            title=title,
            figsize=(5, 5),
            fontsize=10,
            nsamples=64,
            binwidth=15 / 1440.0,
            detrend=False,
        )
        fig.savefig(
            analysis_folder.joinpath(
                "{}_07_lc_emcee_nodetrend.png".format(analysis_id)
            ),
            bbox_inches="tight",
        )
        #  plt.draw()
        plt.close(fig)
        fig = dataset.plot_emcee(
            title=title,
            figsize=(5, 5),
            fontsize=10,
            nsamples=64,
            binwidth=15 / 1440.0,
            detrend=True,
        )
        fig.savefig(
            analysis_folder.joinpath("{}_07_lc_emcee_detrend.png".format(analysis_id)),
            bbox_inches="tight",
        )
        #  plt.draw()
        plt.close(fig)
        fig = dataset.rollangle_plot(figsize=(5, 5), fontsize=8)
        fig.savefig(
            analysis_folder.joinpath(
                "{}_07_emcee_roll_angle_vs_residual.png".format(analysis_id)
            ),
            bbox_inches="tight",
        )
        #  plt.draw()
        plt.close(fig)

        print("\n-Plot trace of the chains")
        fig = dataset.trail_plot("all")  # add 'all' for all traces!
        #  plt.draw()
        fig.savefig(
            analysis_folder.joinpath("{}_08_trace_emcee_all.png".format(analysis_id)),
            bbox_inches="tight",
        )
        plt.close(fig)

        print("\n-Plot corner full from pycheops (not removed nburn)")
        fig = dataset.corner_plot(plotkeys="all")
        #  plt.draw()
        fig.savefig(
            analysis_folder.joinpath("{}_09_corner_emcee_all.png".format(analysis_id)),
            bbox_inches="tight",
        )
        plt.close(fig)

        print("\n-Computing my parameters and plot models with random samples")
        params_med, _, params_best, stats_mle = get_best_parameters(
            result, dataset, nburn=nburn
        )
        # analysis_stats[analysis_id]["EMCEE"] = stats_med
        analysis_stats[analysis_id]["EMCEE"] = stats_mle
        # update emcee.params -> median and emcee.params_best -> mle
        for p in dataset.emcee.params:
            dataset.emcee.params[p] = params_med[p]
            dataset.emcee.params_best[p] = params_best[p]

        fig, _ = model_plot_fit(
            dataset,
            params_med,
            par_type="median",
            nsamples=nwalkers,
            flatchains=result.chain,
            model_filename=analysis_folder.joinpath(
                "{}_07_lc_emcee_median.dat".format(analysis_id)
            ),
        )
        fig.savefig(
            analysis_folder.joinpath("{}_07_lc_emcee_median.png".format(analysis_id)),
            bbox_inches="tight",
        )
        plt.close(fig)
        fig = plot_fft(dataset, params_med, star=star)
        fig.savefig(
            analysis_folder.joinpath("{}_07_fft_emcee_median.png".format(analysis_id)),
            bbox_inches="tight",
        )
        plt.close(fig)

        fig, _ = model_plot_fit(
            dataset,
            params_best,
            par_type="mle",
            nsamples=nwalkers,
            flatchains=result.chain,
            model_filename=analysis_folder.joinpath(
                "{}_07_lc_emcee_mle.dat".format(analysis_id)
            ),
        )
        fig.savefig(
            analysis_folder.joinpath("{}_07_lc_emcee_mle.png".format(analysis_id)),
            bbox_inches="tight",
        )
        plt.close(fig)
        fig = plot_fft(dataset, params_best, star=star)
        fig.savefig(
            analysis_folder.joinpath("{}_07_fft_emcee_mle.png".format(analysis_id)),
            bbox_inches="tight",
        )
        plt.close(fig)

        print("-massaradius with MEDIAN")
        T0_best = ufloat(params_med["T_0"].value, params_med["T_0"].stderr)
        BJD_best = T0_best + lc["bjd_ref"]
        si = ufloat(params_med["sini"].value, params_med["sini"].stderr)
        inc_best = ufloat(params_med["inc"].value, params_med["inc"].stderr)
        W_best = ufloat(params_med["W"].value, params_med["W"].stderr) * P
        W_h, W_m = W_best * cst.day2hour, W_best * cst.day2min
        k = ufloat(params_med["k"].value, params_med["k"].stderr)
        aRs = ufloat(params_med["aR"].value, params_med["aR"].stderr)

        print(
            " T0     = {:.6f} +/- {:.6f} days = {:.6f}+/-{:.6f} BJD_TDB".format(
                T0_best.n, T0_best.s, BJD_best.n, BJD_best.s
            )
        )
        print(
            " err_T0 = {:.3f} m = {:.1f} s".format(
                T0_best.s * cst.day2min, T0_best.s * cst.day2sec
            )
        )
        print(" inc    = {:.6f} +/- {:.6f} deg".format(inc_best.n, inc_best.s))
        print(
            " dur    = {:.6f} +/- {:.6f} days = {:.6f} +/- {:.6f} hour = {:.6f} +/- {:.6f} min".format(
                W_best.n, W_best.s, W_h.n, W_h.s, W_m.n, W_m.s
            )
        )
        print(
            " b      = {:.4f} +/- {:.4f}".format(
                params_med["b"].value, params_med["b"].stderr
            )
        )
        print(" k      = {:.6f} +/- {:.6f}".format(k.n, k.s))
        print(" aRs    = {:.6f} +/- {:.6f}".format(aRs.n, aRs.s))

        analysis_stats[analysis_id]["err_T0_s"] = T0_best.s * cst.day2sec

        try:
            print("\n-trying with tepcat=True")
            _, fig = massradius(
                m_star=Mstar,
                r_star=Rstar,
                k=k,
                K=Kms,
                aR=aRs,
                sini=si,
                P=P,
                jovian=True,
                tepcat=True,
                figsize=(5, 5),
            )
        except:
            print("\n-Error with tepcat=True: using tepcat=False")
            _, fig = massradius(
                m_star=Mstar,
                r_star=Rstar,
                k=k,
                K=Kms,
                aR=aRs,
                sini=si,
                P=P,
                jovian=True,
                tepcat=False,
                figsize=(5, 5),
            )
        #  plt.draw()
        fig.savefig(
            analysis_folder.joinpath("{}_10_massradius_median.png".format(analysis_id)),
            bbox_inches="tight",
        )
        plt.close(fig)
        print(" Saved massradius median")

        print("-massradius with MLE")
        T0_best = ufloat(params_best["T_0"].value, params_best["T_0"].stderr)
        BJD_best = T0_best + lc["bjd_ref"]
        si = ufloat(params_best["sini"].value, params_best["sini"].stderr)
        inc_best = ufloat(params_best["inc"].value, params_best["inc"].stderr)
        W_best = ufloat(params_best["W"].value, params_best["W"].stderr) * P
        W_h, W_m = W_best * cst.day2hour, W_best * cst.day2min
        k = ufloat(params_best["k"].value, params_best["k"].stderr)
        aRs = ufloat(params_best["aR"].value, params_best["aR"].stderr)

        print(
            " T0     = {:.6f} +/- {:.6f} days = {:.6f}+/-{:.6f} BJD_TDB".format(
                T0_best.n, T0_best.s, BJD_best.n, BJD_best.s
            )
        )
        print(
            " err_T0 = {:.3f} m = {:.1f} s".format(
                T0_best.s * cst.day2min, T0_best.s * cst.day2sec
            )
        )
        print(" inc    = {:.6f} +/- {:.6f} deg".format(inc_best.n, inc_best.s))
        print(
            " dur    = {:.6f} +/- {:.6f} days = {:.6f} +/- {:.6f} hour = {:.6f} +/- {:.6f} min".format(
                W_best.n, W_best.s, W_h.n, W_h.s, W_m.n, W_m.s
            )
        )
        print(
            " b      = {:.4f} +/- {:.4f}".format(
                params_best["b"].value, params_best["b"].stderr
            )
        )
        print(" k      = {:.6f} +/- {:.6f}".format(k.n, k.s))
        print(" aRs    = {:.6f} +/- {:.6f}".format(aRs.n, aRs.s))

        try:
            print("\n-trying with tepcat=True")
            _, fig = massradius(
                m_star=Mstar,
                r_star=Rstar,
                k=k,
                K=Kms,
                aR=aRs,
                sini=si,
                P=P,
                jovian=True,
                tepcat=True,
                figsize=(5, 5),
            )
        except:
            print("\n-Error with tepcat=True: using tepcat=False")
            _, fig = massradius(
                m_star=Mstar,
                r_star=Rstar,
                k=k,
                K=Kms,
                aR=aRs,
                sini=si,
                P=P,
                jovian=True,
                tepcat=False,
                figsize=(5, 5),
            )
        #  plt.draw()
        fig.savefig(
            analysis_folder.joinpath("{}_10_massradius_mle.png".format(analysis_id)),
            bbox_inches="tight",
        )
        plt.close(fig)
        print(" Saved massradius mle")

        # update params within dataset with my determination!!

        file_emcee = save_dataset(
            dataset, analysis_folder.resolve(), star_name, file_key, gp=False
        )
        print("-Dumped dataset into file {}".format(file_emcee))

        # EMCEE WITH GAUSSIAN PROCESS 1-------------------------------------
        # Run emcee with SHOTerm kernel for Gaussian Process
        print(
            "\n-Run emcee with GP-SHOTerm with parameters of previous posterior distribution"
        )

        # if(shape == 'fix'):
        #   result_gp = dataset.emcee_sampler(
        #     params      = params_med,
        #     nwalkers    = nwalkers,
        #     burn        = nprerun,
        #     steps       = nsteps,
        #     thin        = nthin,
        #     add_shoterm = True,
        #     progress    = False
        #   )

        # else: # shape == 'fit'

        print("*FIRST: we have to train the gp noise, so I fix the transit parameters")
        # Fix the transit parameters
        # params_fixed = copy_parameters(params_med)
        params_fixed = copy_parameters(params_best)
        for p in ["T_0", "D", "W", "b"]:
            params_fixed[p].set(vary=False)

        result_gp = dataset.emcee_sampler(
            params=params_fixed,
            nwalkers=nwalkers,
            burn=nprerun,
            steps=nsteps,
            thin=nthin,
            add_shoterm=True,
            progress=progress,
        )

        print("\n-Computing my parameters and plot models with random samples")
        params_med_gp, _, params_best_gp, stats_mle_gp = get_best_parameters(
            result_gp, dataset, nburn=nburn
        )
        # analysis_stats[analysis_id]["EMCEE w/ GP"] = stats_med_gp
        analysis_stats[analysis_id]["EMCEE w/ GP"] = stats_mle_gp

        fig, _ = model_plot_fit(
            dataset,
            params_med_gp,
            par_type="median",
            nsamples=nwalkers,
            flatchains=result_gp.chain,
            model_filename=analysis_folder.joinpath(
                "{}_07_lc_emcee_median_gp_train.dat".format(analysis_id)
            ),
        )
        fig.savefig(
            analysis_folder.joinpath(
                "{}_07_lc_emcee_median_gp_train.png".format(analysis_id)
            ),
            bbox_inches="tight",
        )
        plt.close(fig)
        fig, _ = model_plot_fit(
            dataset,
            params_best_gp,
            par_type="mle",
            nsamples=nwalkers,
            flatchains=result_gp.chain,
            model_filename=analysis_folder.joinpath(
                "{}_07_lc_emcee_mle_gp_train.dat".format(analysis_id)
            ),
        )
        fig.savefig(
            analysis_folder.joinpath(
                "{}_07_lc_emcee_mle_gp_train.png".format(analysis_id)
            ),
            bbox_inches="tight",
        )
        plt.close(fig)

        # EMCEE WITH GAUSSIAN PROCESS 2-------------------------------------
        print(
            "\n*SECOND: we have to use priors on gp noise, and set to fit again the transit parameters"
        )
        # params_fit_gp = copy_parameters(params_med_gp)
        params_fit_gp = copy_parameters(params_best_gp)
        # set priors for gp
        for p in ["log_S0", "log_omega0", "log_sigma"]:
            params_fit_gp[p].user_data = ufloat(
                params_best_gp[p].value, 2 * params_best_gp[p].stderr
            )
        # Restoring the transit model parameters as free parameters
        # for p in ['T_0','D','W','b']:
        # params_fit_gp[p].set(vary=True)
        for p in in_par:
            params_fit_gp[p].set(
                vary=params_best[p].vary, min=in_par[p].min, max=in_par[p].max
            )
            params_fit_gp[p].stderr = params_best[p].stderr

        result_gp = dataset.emcee_sampler(
            params=params_fit_gp,
            nwalkers=nwalkers,
            burn=nprerun,
            steps=nsteps,
            thin=nthin,
            # add_shoterm = True, # not needed the second time
            progress=progress,
        )

        print(dataset.emcee_report(min_correl=0.5))

        title = "emcee model (gp)"
        print("\n-Plot pycheops {}".format(title))
        fig = dataset.plot_emcee(
            title=title,
            figsize=(5, 5),
            fontsize=10,
            nsamples=64,
            binwidth=15 / 1440.0,
            detrend=False,
        )
        fig.savefig(
            analysis_folder.joinpath(
                "{}_07_lc_emcee_nodetrend_gp.png".format(analysis_id)
            ),
            bbox_inches="tight",
        )
        plt.close(fig)
        fig = dataset.plot_emcee(
            title=title,
            figsize=(5, 5),
            fontsize=10,
            nsamples=64,
            binwidth=15 / 1440.0,
            detrend=True,
        )
        fig.savefig(
            analysis_folder.joinpath(
                "{}_07_lc_emcee_detrend_gp.png".format(analysis_id)
            ),
            bbox_inches="tight",
        )
        plt.close(fig)
        fig = dataset.rollangle_plot(figsize=(5, 5), fontsize=8)
        fig.savefig(
            analysis_folder.joinpath(
                "{}_07_emcee_roll_angle_vs_residual_gp.png".format(analysis_id)
            ),
            bbox_inches="tight",
        )
        plt.close(fig)

        print("\n-Plot trace of the chains")
        fig = dataset.trail_plot("all")  # add 'all' for all traces!
        #  plt.draw()
        fig.savefig(
            analysis_folder.joinpath(
                "{}_08_trace_emcee_all_gp.png".format(analysis_id)
            ),
            bbox_inches="tight",
        )
        plt.close(fig)

        print("\n-Plot corner full from pycheops (not removed nburn)")
        fig = dataset.corner_plot(plotkeys="all")
        #  plt.draw()
        fig.savefig(
            analysis_folder.joinpath(
                "{}_09_corner_emcee_all_gp.png".format(analysis_id)
            ),
            bbox_inches="tight",
        )
        plt.close(fig)

        print("\n-Computing my parameters and plot models with random samples")
        params_med_gp, _, params_best_gp, stats_mle_gp = get_best_parameters(
            result_gp, dataset, nburn=nburn
        )
        analysis_stats[analysis_id]["EMCEE w/ GP"] = stats_mle_gp
        # update emcee.params -> median and emcee.params_best -> mle
        for p in dataset.emcee.params:
            dataset.emcee.params[p] = params_med_gp[p]
            dataset.emcee.params_best[p] = params_best_gp[p]

        fig, _ = model_plot_fit(
            dataset,
            params_med_gp,
            par_type="median",
            nsamples=nwalkers,
            flatchains=result_gp.chain,
            model_filename=analysis_folder.joinpath(
                "{}_07_lc_emcee_median_gp.dat".format(analysis_id)
            ),
        )
        fig.savefig(
            analysis_folder.joinpath(
                "{}_07_lc_emcee_median_gp.png".format(analysis_id)
            ),
            bbox_inches="tight",
        )
        plt.close(fig)
        fig, _ = model_plot_fit(
            dataset,
            params_best_gp,
            par_type="mle",
            nsamples=nwalkers,
            flatchains=result_gp.chain,
            model_filename=analysis_folder.joinpath(
                "{}_07_lc_emcee_mle_gp.dat".format(analysis_id)
            ),
        )
        fig.savefig(
            analysis_folder.joinpath("{}_07_lc_emcee_mle_gp.png".format(analysis_id)),
            bbox_inches="tight",
        )
        plt.close(fig)

        print("-massaradius with MEDIAN")
        T0_best = ufloat(params_med_gp["T_0"].value, params_med_gp["T_0"].stderr)
        BJD_best = T0_best + lc["bjd_ref"]
        si = ufloat(params_med_gp["sini"].value, params_med_gp["sini"].stderr)
        inc_best = ufloat(params_med_gp["inc"].value, params_med_gp["inc"].stderr)
        W_best = ufloat(params_med_gp["W"].value, params_med_gp["W"].stderr) * P
        W_h, W_m = W_best * cst.day2hour, W_best * cst.day2min
        k = ufloat(params_med_gp["k"].value, params_med_gp["k"].stderr)
        aRs = ufloat(params_med_gp["aR"].value, params_med_gp["aR"].stderr)

        print(
            " T0     = {:.6f} +/- {:.6f} days = {:.6f}+/-{:.6f} BJD_TDB".format(
                T0_best.n, T0_best.s, BJD_best.n, BJD_best.s
            )
        )
        print(
            " err_T0 = {:.3f} m = {:.1f} s".format(
                T0_best.s * cst.day2min, T0_best.s * cst.day2sec
            )
        )
        print(" inc    = {:.6f} +/- {:.6f} deg".format(inc_best.n, inc_best.s))
        print(
            " dur    = {:.6f} +/- {:.6f} days = {:.6f} +/- {:.6f} hour = {:.6f} +/- {:.6f} min".format(
                W_best.n, W_best.s, W_h.n, W_h.s, W_m.n, W_m.s
            )
        )
        print(
            " b      = {:.4f} +/- {:.4f}".format(
                params_med_gp["b"].value, params_med_gp["b"].stderr
            )
        )
        print(" k      = {:.6f} +/- {:.6f}".format(k.n, k.s))
        print(" aRs    = {:.6f} +/- {:.6f}".format(aRs.n, aRs.s))

        analysis_stats[analysis_id]["err_T0_s w/ GP"] = T0_best.s * cst.day2sec

        try:
            print("\n-trying with tepcat=True")
            _, fig = massradius(
                m_star=Mstar,
                r_star=Rstar,
                k=k,
                K=Kms,
                aR=aRs,
                sini=si,
                P=P,
                jovian=True,
                tepcat=True,
                figsize=(5, 5),
            )
        except:
            print("\n-Error with tepcat=True: using tepcat=False")
            _, fig = massradius(
                m_star=Mstar,
                r_star=Rstar,
                k=k,
                K=Kms,
                aR=aRs,
                sini=si,
                P=P,
                jovian=True,
                tepcat=False,
                figsize=(5, 5),
            )
        #  plt.draw()
        fig.savefig(
            analysis_folder.joinpath(
                "{}_10_massradius_median_gp.png".format(analysis_id)
            ),
            bbox_inches="tight",
        )
        plt.close(fig)
        print(" Saved massradius median with gp")

        print("-massradius with MLE")
        T0_best = ufloat(params_best_gp["T_0"].value, params_best_gp["T_0"].stderr)
        BJD_best = T0_best + lc["bjd_ref"]
        si = ufloat(params_best_gp["sini"].value, params_best_gp["sini"].stderr)
        inc_best = ufloat(params_best_gp["inc"].value, params_best_gp["inc"].stderr)
        W_best = ufloat(params_best_gp["W"].value, params_best_gp["W"].stderr) * P
        W_h, W_m = W_best * cst.day2hour, W_best * cst.day2min
        k = ufloat(params_best_gp["k"].value, params_best_gp["k"].stderr)
        aRs = ufloat(params_best_gp["aR"].value, params_best_gp["aR"].stderr)

        print(
            " T0     = {:.6f} +/- {:.6f} days = {:.6f}+/-{:.6f} BJD_TDB".format(
                T0_best.n, T0_best.s, BJD_best.n, BJD_best.s
            )
        )
        print(
            " err_T0 = {:.3f} m = {:.1f} s".format(
                T0_best.s * cst.day2min, T0_best.s * cst.day2sec
            )
        )
        print(" inc    = {:.6f} +/- {:.6f} deg".format(inc_best.n, inc_best.s))
        print(
            " dur    = {:.6f} +/- {:.6f} days = {:.6f} +/- {:.6f} hour = {:.6f} +/- {:.6f} min".format(
                W_best.n, W_best.s, W_h.n, W_h.s, W_m.n, W_m.s
            )
        )
        print(
            " b      = {:.4f} +/- {:.4f}".format(
                params_best_gp["b"].value, params_best_gp["b"].stderr
            )
        )
        print(" k      = {:.6f} +/- {:.6f}".format(k.n, k.s))
        print(" aRs    = {:.6f} +/- {:.6f}".format(aRs.n, aRs.s))

        try:
            print("\n-trying with tepcat=True")
            _, fig = massradius(
                m_star=Mstar,
                r_star=Rstar,
                k=k,
                K=Kms,
                aR=aRs,
                sini=si,
                P=P,
                jovian=True,
                tepcat=True,
                figsize=(5, 5),
            )
        except:
            print("\n-Error with tepcat=True: using tepcat=False")
            _, fig = massradius(
                m_star=Mstar,
                r_star=Rstar,
                k=k,
                K=Kms,
                aR=aRs,
                sini=si,
                P=P,
                jovian=True,
                tepcat=False,
                figsize=(5, 5),
            )

        #  plt.draw()
        fig.savefig(
            analysis_folder.joinpath("{}_10_massradius_mle_gp.png".format(analysis_id)),
            bbox_inches="tight",
        )
        plt.close(fig)
        print(" Saved massradius mle with gp")

        file_emcee_gp = save_dataset(
            dataset, analysis_folder.resolve(), star_name, file_key, gp=True
        )
        print("-Dumped dataset into file {}".format(file_emcee_gp))

        print("ANALYSIS {} DONE".format(analysis))
        plt.close("all")
        print("=============================================================")

    return analysis_stats


# ======================================================================
# ======================================================================


class PIPEDataset(Dataset):
    # based on get_lightcurve method of Dataset class
    def get_PIPE_lightcurve(self, PIPE_data, reject_highpoints=False, verbose=False):
        # start updating details within dataset
        ap_rad = 25.0
        aperture = "PSF"
        self.pipe_ver = "PIPE"
        self.ap_rad = ap_rad
        self.aperture = aperture
        self.PIPE_data = PIPE_data

        ok = np.logical_and(PIPE_data["FLUX"] > 0.0, PIPE_data["FLAG"] < 8)
        m = np.isnan(PIPE_data["FLUX"])
        if np.sum(m) > 0:
            msg = "Light curve contains {} NaN values".format(np.sum(m))
            warnings.warn(msg)
            ok = np.logical_and(ok, ~m)

        PIPE_ok = {}
        for k, v in PIPE_data.items():
            PIPE_ok[k] = v[ok]

        bjd = PIPE_ok["BJD_TIME"]
        bjd_ref = np.int(bjd[0])
        self.bjd_ref = bjd_ref

        time = bjd - bjd_ref
        flux = PIPE_ok["FLUX"]
        flux_err = PIPE_ok["FLUXERR"]

        centroid_x, centroid_y = PIPE_ok["XC"], PIPE_ok["YC"]
        xoff = centroid_x - 100.0  # image 200x200px with star centered at (100,100)
        yoff = centroid_y - 100.0  # image 200x200px with star centered at (100,100)

        roll_angle = PIPE_ok["ROLL"]
        bg = PIPE_ok["BG"]

        # set to zeros
        contam = np.zeros_like(bjd)
        smear = np.zeros_like(bjd)

        try:
            deltaT = PIPE_ok["thermFront_2"] + 12
        except:
            deltaT = np.zeros_like(bjd)

        # reject highpoints
        if reject_highpoints:
            C_cut = 2 * np.nanmedian(flux) - np.nanmin(flux)
            okr = (flux < C_cut).nonzero()
            time = time[okr]
            flux = flux[okr]
            flux_err = flux_err[okr]
            xoff = xoff[okr]
            yoff = yoff[okr]
            roll_angle = roll_angle[okr]
            bg = bg[okr]
            contam = contam[okr]
            smear = smear[okr]
            deltaT = deltaT[okr]
            N_cut = len(bjd) - len(time)

        # computes mean and median
        fluxmed = np.nanmedian(flux)
        self.flux_mean = flux.mean()
        self.flux_median = fluxmed
        self.flux_rms = np.std(flux, ddof=1)
        self.flux_mse = np.nanmedian(flux_err)

        if verbose:
            if reject_highpoints:
                print("C_cut = {:0.0f}".format(C_cut))
                print("N(C > C_cut) = {}".format(N_cut))
            print("Mean counts = {:0.1f}".format(self.flux_mean))
            print("Median counts = {:0.1f}".format(fluxmed))
            print(
                "RMS counts = {:0.1f} [{:0.0f} ppm]".format(
                    np.nanstd(flux), 1e6 * np.nanstd(flux) / fluxmed
                )
            )
            print(
                "Median standard error = {:0.1f} [{:0.0f} ppm]".format(
                    np.nanmedian(flux_err), 1e6 * np.nanmedian(flux_err) / fluxmed
                )
            )
            print("Mean contamination = {:0.1f} ppm".format(1e6 * contam.mean()))
            print(
                "Mean smearing correction = {:0.1f} ppm".format(
                    1e6 * smear.mean() / fluxmed
                )
            )
            if np.max(np.abs(deltaT)) > 0:
                f = interp1d(
                    [22.5, 25, 30, 40],
                    [140, 200, 330, 400],
                    bounds_error=False,
                    fill_value="extrapolate",
                )
                ramp = np.ptp(f(ap_rad) * deltaT)
                print("Predicted amplitude of ramp = {:0.0f} ppm".format(ramp))

        flux = flux / fluxmed
        flux_err = flux_err / fluxmed
        #     smear    = smear/fluxmed
        self.lc = {
            "time": time,
            "flux": flux,
            "flux_err": flux_err,
            "bjd_ref": bjd_ref,
            "table": PIPE_data,
            "header": [k for k in PIPE_data.keys()],
            "xoff": xoff,
            "yoff": yoff,
            "bg": bg,
            "contam": contam,
            "smear": smear,
            "deltaT": deltaT,
            "centroid_x": centroid_x,
            "centroid_y": centroid_y,
            "roll_angle": roll_angle,
            "aperture": aperture,
        }

        return time, flux, flux_err


# ======================================================================
# ======================================================================


class FITSDataset(Dataset):
    # based on get_lightcurve method of Dataset class
    def get_FITS_lightcurve(
        self, visit_args, transit, info, reject_highpoints=False, verbose=False
    ):
        # start updating details within dataset
        ap_rad = 25.0
        aperture = visit_args["aperture"].upper()
        FITS_data = transit["data"]
        self.pipe_ver = visit_args["passband"]
        self.ap_rad = ap_rad
        self.aperture = aperture
        self.FITS_data = FITS_data

        btjd = info["BTJD"]

        # TESS "normal lc"
        # TIME TIMECORR CADENCENO SAP_FLUX SAP_FLUX_ERR SAP_BKG SAP_BKG_ERR PDCSAP_FLUX PDCSAP_FLUX_ERR SAP_QUALITY MOM_CENTR1 MOM_CENTR1_ERR MOM_CENTR2 MOM_CENTR2_ERR POS_CORR1 POS_CORR2
        # TESS HLSP/QLP HAS NO PDC AND NO SAP_FLUX_ERR, BUT KSPSAP_FLUX/_ERR

        if aperture == "pdc":
            key_flux = "PCDSAP_FLUX"
            key_flux_err = "PCDSAP_FLUX_ERR"
        else:
            key_flux = "SAP_FLUX"
            key_flux_err = "SAP_FLUX_ERR"

        # CHECK IF key_flux_err in the data keyword, if not it means we are probably using HLSP/QPL lc.
        if key_flux_err not in transit["data"].keys():
            key_flux = "KSPSAP_FLUX"
            key_flux_err = "KSPSAP_FLUX_ERR"

        flux_raw = transit["data"][key_flux]

        # TODO get_quality_combination base on QUALITY_FLAGS
        # ok = np.logical_and(
        #   flux_raw > 0.0,
        # )
        ok = flux_raw > 0.0

        m = np.isnan(flux_raw)
        if np.sum(m) > 0:
            msg = "Light curve contains {} NaN values".format(np.sum(m))
            warnings.warn(msg)
            ok = np.logical_and(ok, ~m)

        data_ok = {}
        for k, v in FITS_data.items():
            data_ok[k] = v[ok]

        bjd = data_ok["TIME"] + btjd
        bjd_ref = np.int(bjd[0])
        self.bjd_ref = bjd_ref

        time = bjd - bjd_ref
        flux = data_ok[key_flux]
        flux_err = data_ok[key_flux_err]

        if "KSP" in key_flux:
            centroid_x, centroid_y = data_ok["SAP_X"], data_ok["SAP_Y"]
            xoff = centroid_x - np.median(centroid_x)
            yoff = centroid_y - np.median(centroid_y)
        else:
            centroid_x, centroid_y = data_ok["MOM_CENTR1"], data_ok["MOM_CENTR2"]
            xoff = data_ok["POS_CORR1"]
            yoff = data_ok["POS_CORR2"]

        bg = data_ok["SAP_BKG"]

        # set to zeros
        roll_angle = np.zeros_like(bjd)
        contam = np.zeros_like(bjd)
        smear = np.zeros_like(bjd)
        deltaT = np.zeros_like(bjd)

        # reject highpoints
        if reject_highpoints:
            C_cut = 2 * np.nanmedian(flux) - np.nanmin(flux)
            okr = (flux < C_cut).nonzero()
            time = time[okr]
            flux = flux[okr]
            flux_err = flux_err[okr]
            xoff = xoff[okr]
            yoff = yoff[okr]
            roll_angle = roll_angle[okr]
            bg = bg[okr]
            contam = contam[okr]
            smear = smear[okr]
            deltaT = deltaT[okr]
            N_cut = len(bjd) - len(time)

        # computes mean and median
        fluxmed = np.nanmedian(flux)
        self.flux_mean = flux.mean()
        self.flux_median = fluxmed
        self.flux_rms = np.std(flux, ddof=1)
        self.flux_mse = np.nanmedian(flux_err)

        if verbose:
            if reject_highpoints:
                print("C_cut = {:0.0f}".format(C_cut))
                print("N(C > C_cut) = {}".format(N_cut))
            print("Mean counts = {:0.1f}".format(self.flux_mean))
            print("Median counts = {:0.1f}".format(fluxmed))
            print(
                "RMS counts = {:0.1f} [{:0.0f} ppm]".format(
                    np.nanstd(flux), 1e6 * np.nanstd(flux) / fluxmed
                )
            )
            print(
                "Median standard error = {:0.1f} [{:0.0f} ppm]".format(
                    np.nanmedian(flux_err), 1e6 * np.nanmedian(flux_err) / fluxmed
                )
            )
            print("Mean contamination = {:0.1f} ppm".format(1e6 * contam.mean()))
            print(
                "Mean smearing correction = {:0.1f} ppm".format(
                    1e6 * smear.mean() / fluxmed
                )
            )
            if np.max(np.abs(deltaT)) > 0:
                f = interp1d(
                    [22.5, 25, 30, 40],
                    [140, 200, 330, 400],
                    bounds_error=False,
                    fill_value="extrapolate",
                )
                ramp = np.ptp(f(ap_rad) * deltaT)
                print("Predicted amplitude of ramp = {:0.0f} ppm".format(ramp))

        flux = flux / fluxmed
        flux_err = flux_err / fluxmed
        #     smear    = smear/fluxmed
        self.lc = {
            "time": time,
            "flux": flux,
            "flux_err": flux_err,
            "bjd_ref": bjd_ref,
            "table": FITS_data,
            "header": [k for k in FITS_data.keys()],
            "xoff": xoff,
            "yoff": yoff,
            "bg": bg,
            "contam": contam,
            "smear": smear,
            "deltaT": deltaT,
            "centroid_x": centroid_x,
            "centroid_y": centroid_y,
            "roll_angle": roll_angle,
            "aperture": aperture,
        }

        return time, flux, flux_err


class AsciiDataset(Dataset):
    # based on get_lightcurve method of Dataset class
    def get_ascii_lightcurve(
        self, ascii_data, normalise=False, reject_highpoints=False, verbose=False
    ):
        # start updating details within dataset
        ap_rad = 0.0
        aperture = "ascii"
        self.pipe_ver = "ascii"
        self.ap_rad = ap_rad
        self.aperture = aperture
        self.ascii_data = ascii_data

        ascii_keys = [k for k in ascii_data.keys()]

        ok = ascii_data["flux"] > 0.0
        m = np.isnan(ascii_data["flux"])
        if np.sum(m) > 0:
            msg = "Light curve contains {} NaN values".format(np.sum(m))
            warnings.warn(msg)
            ok = np.logical_and(ok, ~m)

        ascii_ok = {}
        for k, v in ascii_data.items():
            ascii_ok[k] = v[ok]

        bjd = ascii_ok["time"]
        bjd_ref = np.int(bjd[0])
        self.bjd_ref = bjd_ref

        time = bjd - bjd_ref
        flux = ascii_ok["flux"]
        flux_err = ascii_ok["flux_err"]

        # set default diagnostics to zeros
        centroid_x, centroid_y = np.zeros_like(time), np.zeros_like(time)
        xoff, yoff = np.zeros_like(time), np.zeros_like(time)
        bg = np.zeros_like(time)
        roll_angle = np.zeros_like(time)
        contam = np.zeros_like(time)
        smear = np.zeros_like(time)
        deltaT = np.zeros_like(time)

        if "centroid_x" in ascii_keys:
            centroid_x = ascii_ok["centroid_x"]
        if "centroid_y" in ascii_keys:
            centroid_y = ascii_ok["centroid_y"]
        if "xoff" in ascii_keys:
            xoff = ascii_ok["xoff"]
        if "yoff" in ascii_keys:
            yoff = ascii_ok["yoff"]
        if "bg" in ascii_keys:
            bg = ascii_ok["bg"]
        if "yoff" in ascii_keys:
            yoff = ascii_ok["yoff"]
        if "contam" in ascii_keys:
            contam = ascii_ok["contam"]
        if "smear" in ascii_keys:
            smear = ascii_ok["smear"]

        # reject highpoints
        if reject_highpoints:
            C_cut = 2 * np.nanmedian(flux) - np.nanmin(flux)
            okr = (flux < C_cut).nonzero()
            time = time[okr]
            flux = flux[okr]
            flux_err = flux_err[okr]
            xoff = xoff[okr]
            yoff = yoff[okr]
            roll_angle = roll_angle[okr]
            bg = bg[okr]
            contam = contam[okr]
            smear = smear[okr]
            deltaT = deltaT[okr]
            N_cut = len(bjd) - len(time)

        # computes mean and median
        fluxmed = np.nanmedian(flux)
        self.flux_mean = flux.mean()
        self.flux_median = fluxmed
        self.flux_rms = np.std(flux, ddof=1)
        self.flux_mse = np.nanmedian(flux_err)

        if verbose:
            if reject_highpoints:
                print("C_cut = {:0.0f}".format(C_cut))
                print("N(C > C_cut) = {}".format(N_cut))
            print("Mean counts = {:0.1f}".format(self.flux_mean))
            print("Median counts = {:0.1f}".format(fluxmed))
            print(
                "RMS counts = {:0.1f} [{:0.0f} ppm]".format(
                    np.nanstd(flux), 1e6 * np.nanstd(flux) / fluxmed
                )
            )
            print(
                "Median standard error = {:0.1f} [{:0.0f} ppm]".format(
                    np.nanmedian(flux_err), 1e6 * np.nanmedian(flux_err) / fluxmed
                )
            )
            print("Mean contamination = {:0.1f} ppm".format(1e6 * contam.mean()))
            print(
                "Mean smearing correction = {:0.1f} ppm".format(
                    1e6 * smear.mean() / fluxmed
                )
            )
            if np.max(np.abs(deltaT)) > 0:
                f = interp1d(
                    [22.5, 25, 30, 40],
                    [140, 200, 330, 400],
                    bounds_error=False,
                    fill_value="extrapolate",
                )
                ramp = np.ptp(f(ap_rad) * deltaT)
                print("Predicted amplitude of ramp = {:0.0f} ppm".format(ramp))

        if normalise:
            print("NORMALISING FLUX!!!")
            flux = flux / fluxmed
            flux_err = flux_err / fluxmed
        #     smear    = smear/fluxmed
        self.lc = {
            "time": time,
            "flux": flux,
            "flux_err": flux_err,
            "bjd_ref": bjd_ref,
            "table": ascii_data,
            "header": ascii_keys,
            "xoff": xoff,
            "yoff": yoff,
            "bg": bg,
            "contam": contam,
            "smear": smear,
            "deltaT": deltaT,
            "centroid_x": centroid_x,
            "centroid_y": centroid_y,
            "roll_angle": roll_angle,
            "aperture": aperture,
        }

        return time, flux, flux_err


# ======================================================================
# ======================================================================


def save_dataset(dataset, folder, target, file_key, gp=False):
    """
    Save the current dataset as a pickle file

    :returns: pickle file name
    """
    fd = os.path.abspath(folder)
    f0 = "{}__{}.dataset".format(target.replace(" ", "_"), file_key)

    if gp:
        fn = "{}".format(f0.replace(".dataset", "_gp.dataset"))
    else:
        fn = f0

    fl = os.path.join(fd, fn)
    with open(fl, "wb") as fp:
        pickle.dump(dataset, fp, pickle.HIGHEST_PROTOCOL)

    # dataset.save()
    # os.replace(f0, fl)

    return fl


def load_dataset(filename):
    """
    Load a dataset from a pickle file

    :param filename: pickle file name

    :returns: dataset object

    """
    with open(filename, "rb") as fp:
        dataset = pickle.load(fp)
    return dataset


# ======================================================================
# ======================================================================


class CustomMultiVisit(MultiVisit):
    def __init__(
        self,
        target=None,
        datasets_list=None,
        ident=None,
        id_kws={"dace": True},
        verbose=True,
    ):
        self.target = target
        self.datasets = []

        if target is None:
            return
        if datasets_list is None:
            return

        if datasets_list is None:
            MultiVisit.__init__(
                self,
                target=target,
                datadir=None,
                ident=ident,
                id_kws=id_kws,
                verbose=verbose,
            )
            return

        # 0) original pycheops
        # datatimes = [Dataset.load(i).bjd_ref for i in glob(ptn)]
        # g = [x for _,x in sorted(zip(datatimes,glob(ptn)))]

        # 1) LBo based on pycheops Dataset.load but providing file list
        # datatimes = [Dataset.load(i).bjd_ref for i in datasets_list]
        # g = [x for _,x in sorted(zip(datatimes,datasets_list))]

        # 2) LBo using own load_dataset function and providing file list
        g = [load_dataset(fl) for fl in datasets_list]
        t_ref = [d.bjd_ref for d in g]
        idx = np.argsort(t_ref)
        g = [g[ix] for ix in idx]

        # if ident is not 'none':
        # if ident is None: ident = target
        # self.star = StarProperties(ident, **id_kws)
        if ident is None:
            ident = target
        self.star = StarProperties(ident, **id_kws)

        if verbose:
            print(self.star)
            print("N  file_key                   Aperture last_ GP  Glint pipe_ver")
            print("---------------------------------------------------------------")

        # for case 0) and 1)
        # for n,fl in enumerate(g):
        #   d = Dataset.load(fl)
        # d = load_dataset(fl)

        # for case 2)
        for n, d in enumerate(g):

            # print("\ndataset {}".format(n+1))
            # print("-Loaded status")
            # print("| lmfit T_0: {}          id = {}".format(d.lmfit.params['T_0'], id(d.lmfit.params['T_0'])))
            # print("| emcee T_0: {} (median) id = {}".format(d.emcee.params['T_0'], id(d.emcee.params['T_0'])))
            # print("| emcee T_0: {} (best)   id = {}".format(d.emcee.params_best['T_0'], id(d.emcee.params_best['T_0'])))
            # print("-Modified status")

            # Make time scales consistent
            dBJD = d.bjd_ref - 2457000
            d._old_bjd_ref = d.bjd_ref
            d.bjd_ref = 2457000
            d.lc["time"] += dBJD
            d.lc["bjd_ref"] = dBJD
            if "lmfit" in d.__dict__:
                p = d.lmfit.params["T_0"]
                p._val += dBJD
                p.init_value += dBJD
                p.min += dBJD
                p.max += dBJD
                if "T_0" in d.lmfit.var_names:
                    d.lmfit.init_vals[d.lmfit.var_names.index("T_0")] += dBJD
                if "T_0" in d.lmfit.init_values:
                    d.lmfit.init_values["T_0"] += dBJD
            # print("*)lmfit T_0: {}".format(d.lmfit.params['T_0']))

            if "emcee" in d.__dict__:
                # print("0) lmfit T_0: {}          id = {}".format(d.lmfit.params['T_0'], id(d.lmfit.params['T_0'])))
                # print("0) emcee T_0: {} (median) id = {}".format(d.emcee.params['T_0'], id(d.emcee.params['T_0'])))
                # print("0) emcee T_0: {} (best)   id = {}".format(d.emcee.params_best['T_0'], id(d.emcee.params_best['T_0'])))
                # print('--emcee modifications: params')
                p = d.emcee.params["T_0"]
                p._val += dBJD
                p.init_value += dBJD
                p.min += dBJD
                p.max += dBJD
                # print("1) lmfit T_0: {}          id = {}".format(d.lmfit.params['T_0'], id(d.lmfit.params['T_0'])))
                # print("1) emcee T_0: {} (median) id = {}".format(d.emcee.params['T_0'], id(d.emcee.params['T_0'])))
                # print("1) emcee T_0: {} (best)   id = {}".format(d.emcee.params_best['T_0'], id(d.emcee.params_best['T_0'])))
                # print('--emcee modifications: params_best')
                p = d.emcee.params_best["T_0"]
                p._val += dBJD
                p.init_value += dBJD
                p.min += dBJD
                p.max += dBJD
                # print("2) lmfit T_0: {}          id = {}".format(d.lmfit.params['T_0'], id(d.lmfit.params['T_0'])))
                # print("2) emcee T_0: {} (median) id = {}".format(d.emcee.params['T_0'], id(d.emcee.params['T_0'])))
                # print("2) emcee T_0: {} (best)   id = {}".format(d.emcee.params_best['T_0'], id(d.emcee.params_best['T_0'])))

                if "T_0" in d.emcee.var_names:
                    j = d.emcee.var_names.index("T_0")
                    if "init_vals" in d.emcee.__dict__:
                        # print('--emcee modfications: init_vals and chain of j = {}'.format(j))
                        # print("3) emcee.init_vals[j]: {}".format(d.emcee.init_vals[j]))
                        d.emcee.init_vals[j] += dBJD
                        # print("4) emcee.init_vals[j]: {}".format(d.emcee.init_vals[j]))
                        d.emcee.chain[:, j] += dBJD
                if "T_0" in d.emcee.init_values:
                    # print('--emcee modfications: init_values')
                    # print("5) emcee.init_values['T_0']: {}".format(d.emcee.init_values['T_0']))
                    d.emcee.init_values["T_0"] += dBJD
                    # print("6) emcee.init_values['T_0']: {}".format(d.emcee.init_values['T_0']))

            # print("| lmfit T_0: {}          id = {}".format(d.lmfit.params['T_0'], id(d.lmfit.params['T_0'])))
            # print("| emcee T_0: {} (median) id = {}".format(d.emcee.params['T_0'], id(d.emcee.params['T_0'])))
            # print("| emcee T_0: {} (best)   id = {}".format(d.emcee.params_best['T_0'], id(d.emcee.params_best['T_0'])))

            self.datasets.append(d)
            if verbose:
                dd = d.__dict__
                ap = d.lc["aperture"] if "lc" in dd else "---"
                lf = d.__lastfit__ if "__lastfit__" in dd else "---"
                try:
                    gp = "Yes" if d.gp else "No"
                except AttributeError:
                    gp = "No"
                gl = "Yes" if "f_glint" in dd else "No"
                pv = d.pipe_ver
                print(f" {n+1:2} {d.file_key} {ap:8} {lf:5} {gp:3} {gl:5} {pv}")

        return


def custom_plot_phase(M, result, title=None):

    parbest = result.parbest.copy()

    modpars = M.modpars

    P = parbest["P"].value
    T_0 = parbest["T_0"].value

    zd = 7
    zm = 8 + len(M.datasets)

    x0, y0, width = 0.125, 0.10, 0.775
    height_max = 0.85

    # shf = 0.79
    shr = 0.15  # 0.19
    shx = 0.02

    hx = height_max * shx
    hr = height_max * shr
    hf = 0.5 * (height_max - hx * 2 - hr)  # height_max*shf

    fig = plt.figure()

    xf, yf, wf = x0, y0 + hr + hx + hf + hx, width
    axf = fig.add_axes([xf, yf, wf, hf])
    if title is not None:
        axf.set_title(title)
    axf.ticklabel_format(useOffset=False)
    axf.tick_params(labelbottom=False)

    xd, yd, wd = x0, y0 + hr + hx, width
    axd = fig.add_axes([xd, yd, wd, hf])
    axd.ticklabel_format(useOffset=False)
    axd.tick_params(labelbottom=False)

    xr, yr, wr = x0, y0, width
    axr = fig.add_axes([xr, yr, wr, hr])
    axr.ticklabel_format(useOffset=False)
    axr.tick_params(labelbottom=True)

    gcol_min = 0.2
    gcol_max = 0.9
    gcol = np.linspace(gcol_min, gcol_max, num=len(M.datasets), endpoint=True)

    time, flux, flux_err = [], [], []
    phase = []
    flux_all, flux_tra, flux_trend = [], [], []
    residuals = []
    lc_id = []

    nvis = len(M.datasets)
    vir_map = plt.cm.get_cmap("viridis")
    icol = np.linspace(0, 1, num=nvis, endpoint=True)

    for i, d in enumerate(M.datasets):
        t = d.lc["time"]
        f = d.lc["flux"]
        ef = d.lc["flux_err"]
        ph = (((t - T_0) / P) % 1 + 0.5) % 1
        models = M.models[i]

        # needed to have a true transit model
        for dp in (
            "c",
            "dfdbg",
            "dfdcontam",
            "glint_scale",
            "dfdx",
            "d2fdx2",
            "dfdy",
            "d2fdy2",
            "dfdt",
            "d2fdt2",
        ):
            p = f"{dp}_{i+1:02d}"
            if p in result.var_names:
                modpars[i][dp].value = 1 if dp == "c" else 0

        f_tra = models.eval(modpars[i], t=t)
        f_all = result.bestfit[i]
        f_trend = f_all - f_tra
        res = result.residual[i]

        n = len(t)
        lc_id.append(np.zeros((n)) + i + 1)
        time.append(t)
        flux.append(f)
        flux_err.append(ef)
        phase.append(ph)
        flux_all.append(f_all)
        flux_tra.append(f_tra)
        flux_trend.append(f_trend)
        residuals.append(res)

        dcol = (gcol[i], gcol[i], gcol[i], 1)

        axf.errorbar(
            ph,
            f,
            yerr=ef,
            color=dcol,
            marker="o",
            ms=2,
            mec="gray",
            mew=0.3,
            ecolor="gray",
            elinewidth=0.8,
            capsize=0,
            ls="",
            zorder=zd + i,
            label="data visit#{:02d}".format(i + 1),
        )
        axf.plot(
            ph,
            f_all,
            color=vir_map(icol[i]),
            marker="o",
            ms=1,
            alpha=0.1 + 1 / len(M.datasets),
            ls="",
            zorder=zm + i,
            label="model visit#{:02d}".format(i + 1),
        )
        axf.set_ylabel("normalized flux")

        axd.errorbar(
            ph,
            f - f_trend,
            yerr=ef,
            color=dcol,
            marker="o",
            ms=2,
            mec="gray",
            mew=0.3,
            ecolor="gray",
            elinewidth=0.8,
            capsize=0,
            ls="",
            zorder=zd + i,
        )
        axd.plot(
            ph,
            f_tra,
            color=vir_map(icol[i]),
            marker="o",
            ms=1,
            alpha=0.1 + 1 / len(M.datasets),
            ls="",
            zorder=zm + i,
        )
        axd.set_ylabel("detrended flux")

        axr.axhline(0.0, color="black", ls="-", lw=1, zorder=6)
        axr.errorbar(
            ph,
            res,
            yerr=ef,
            color=dcol,
            marker="o",
            ms=2,
            mec="gray",
            mew=0.3,
            ecolor="gray",
            elinewidth=0.8,
            capsize=0,
            ls="",
            zorder=zd + i,
        )
        axr.set_xlabel(r"$\phi$")
        axr.set_ylabel("residuals")

    # axf.legend(loc='best', fontsize=6)
    axf.legend(loc="upper left", bbox_to_anchor=(1.001, 0.66), ncol=1, fontsize=6)

    out = {}
    out["time"] = np.concatenate(time)
    out["flux"] = np.concatenate(flux)
    out["flux_err"] = np.concatenate(flux_err)
    out["phase"] = np.concatenate(phase)
    out["flux_all"] = np.concatenate(flux_all)
    out["flux_tra"] = np.concatenate(flux_tra)
    out["flux_trend"] = np.concatenate(flux_trend)
    out["residuals"] = np.concatenate(residuals)
    out["lc_id"] = np.concatenate(lc_id)

    return fig, out


def custom_plot_phase_from_model(model, title=None):

    times = model["time"]
    fluxs = model["flux"]
    flux_errs = model["flux_err"]
    phases = model["phase"]
    flux_alls = model["flux_all"]
    flux_tras = model["flux_tra"]
    flux_trends = model["flux_trend"]
    residuals = model["residuals"]
    lc_id = model["lc_id"].astype(int)

    vis = np.unique(lc_id)
    nvis = len(vis)

    # markersize
    # data
    msdata = 3.0
    # model
    msmod = 1.0
    # alpha model
    base_alpha = 0.4
    tot_alpha = 1.0 - base_alpha
    # alpha data
    data_alpha = 1.0

    zd = 7
    zm = 8 + nvis

    x0, y0, width = 0.125, 0.10, 0.775
    height_max = 0.85

    # shf = 0.79
    shr = 0.15  # 0.19
    shx = 0.02

    hx = height_max * shx
    hr = height_max * shr
    hf = 0.5 * (height_max - hx * 2 - hr)  # height_max*shf

    fig = plt.figure()
    labelx = -0.2

    xf, yf, wf = x0, y0 + hr + hx + hf + hx, width
    axf = fig.add_axes([xf, yf, wf, hf])
    axf.yaxis.set_label_coords(labelx, 0.5)
    if title is not None:
        axf.set_title(title)
    axf.ticklabel_format(useOffset=False)
    axf.tick_params(labelbottom=False)

    xd, yd, wd = x0, y0 + hr + hx, width
    axd = fig.add_axes([xd, yd, wd, hf])
    axd.yaxis.set_label_coords(labelx, 0.5)
    axd.ticklabel_format(useOffset=False)
    axd.tick_params(labelbottom=False)

    xr, yr, wr = x0, y0, width
    axr = fig.add_axes([xr, yr, wr, hr])
    axr.yaxis.set_label_coords(labelx, 0.5)
    axr.ticklabel_format(useOffset=False)
    axr.tick_params(labelbottom=True)

    gcol_min = 0.2
    gcol_max = 0.9
    gcol = np.linspace(gcol_min, gcol_max, num=nvis, endpoint=True)

    vir_map = plt.cm.get_cmap("viridis_r")
    icol = np.linspace(0, 1, num=nvis, endpoint=True)

    for i in range(nvis):
        vn = vis[i]
        sel = lc_id == vn
        t = times[sel]
        f = fluxs[sel]
        ef = flux_errs[sel]
        ph = phases[sel]

        f_tra = flux_tras[sel]
        f_all = flux_alls[sel]
        f_trend = flux_trends[sel]
        res = residuals[sel]

        n = len(t)

        dcol = (gcol[i], gcol[i], gcol[i], 1)

        axf.errorbar(
            ph,
            f,
            yerr=ef,
            color=dcol,
            marker="o",
            ms=msdata,
            mec="gray",
            mew=0.3,
            ecolor="gray",
            elinewidth=0.8,
            capsize=0,
            ls="",
            alpha=data_alpha,
            zorder=zd + i,
            label="data visit#{:02d}".format(i + 1),
        )
        axf.plot(
            ph,
            f_all,
            color=vir_map(icol[i]),
            marker=".",
            ms=msmod,
            alpha=base_alpha + tot_alpha / nvis,
            ls="",
            zorder=zm + i,
            label="model visit#{:02d}".format(i + 1),
        )
        axf.set_ylabel("normalized flux")

        axd.errorbar(
            ph,
            f - f_trend,
            yerr=ef,
            color=dcol,
            marker="o",
            ms=msdata,
            mec="gray",
            mew=0.3,
            ecolor="gray",
            elinewidth=0.8,
            capsize=0,
            ls="",
            alpha=data_alpha,
            zorder=zd + i,
        )
        axd.plot(
            ph,
            f_tra,
            color=vir_map(icol[i]),
            marker=".",
            ms=msmod,
            alpha=base_alpha + tot_alpha / nvis,
            ls="",
            zorder=zm + i,
        )
        axd.set_ylabel("detrended flux")

        axr.axhline(0.0, color="black", ls="-", lw=1, zorder=6)
        axr.errorbar(
            ph,
            res,
            yerr=ef,
            color=dcol,
            marker="o",
            ms=msdata,
            mec="gray",
            mew=0.3,
            ecolor="gray",
            elinewidth=0.8,
            capsize=0,
            ls="",
            alpha=data_alpha,
            zorder=zd + i,
        )
        axr.set_xlabel(r"$\phi$")
        axr.set_ylabel("residuals")

    # axf.legend(loc='best', fontsize=6)
    axf.legend(
        loc="upper left",
        bbox_to_anchor=(1.001, 0.66),
        ncol=1,
        fontsize=plt.rcParams["xtick.labelsize"] - 2,
    )

    return fig


# ======================================================================
def mask_data_clipping(x, k, clip_type="median"):

    if clip_type == "mean":
        mu = np.mean(x)
    else:
        mu = np.median(x)

    ares = np.abs(x - mu)
    rms = np.percentile(ares, 68.27, interpolation="midpoint")
    mask = ares > k * rms
    return mask


# ======================================================================
def computes_bayes_factor(params):

    # pycheops
    # for p in params:
    #   u = params[p].user_data
    #   if (isinstance(u, UFloat) and
    #           (p.startswith('dfd') or p.startswith('d2f') or
    #             (p == 'ramp') or (p == 'glint_scale') ) ):
    #     v = params[p].value
    #     s = params[p].stderr
    #     if s is not None:
    #       B = np.exp(-0.5*((v-u.n)/s)**2) * u.s/s

    BFx = {}

    for p in params:
        u = params[p].user_data
        if isinstance(u, UFloat):
            if (
                p.startswith("dfd")
                or p.startswith("d2f")
                or p in ["ramp", "glint_scale"]
            ):
                v = params[p].value
                s = params[p].stderr
                if s is not None:
                    B = np.exp(-0.5 * ((v - u.n) / s) ** 2) * u.s / s
                    BFx[p] = B

    # sort the BF dictionary
    # reverse sorting of dictionary
    # x = {1: 2, 3: 4, 4: 3, 2: 1, 0: 0}
    # y = dict(sorted(x.items(), key=lambda item: item[1], reverse=True))
    BF = dict(sorted(BFx.items(), key=lambda item: item[1], reverse=True))

    return BF


# ======================================================================
def planet_check(dataset, olog=None):
    bjd = Time(dataset.bjd_ref + dataset.lc["time"][0], format="jd", scale="tdb")
    target_coo = SkyCoord(dataset.ra, dataset.dec, unit=("hour", "degree"))
    printlog(f"BJD = {bjd}", olog=olog)
    printlog("Body     R.A.         Declination  Sep(deg)", olog=olog)
    printlog("-------------------------------------------", olog=olog)
    for p in ("moon", "mars", "jupiter", "saturn", "uranus", "neptune"):
        c = get_body(p, bjd)
        ra = c.ra.to_string(precision=2, unit="hour", sep=":", pad=True)
        dec = c.dec.to_string(
            precision=1, sep=":", unit="degree", alwayssign=True, pad=True
        )
        sep = target_coo.separation(c).degree
        printlog(f"{p.capitalize():8s} {ra:12s} {dec:12s} {sep:8.1f}", olog=olog)
    return


# ======================================================================
def copy_DRP_report(dataset, output_folder, olog=None):

    out_folder = os.path.abspath(output_folder)

    pdfFile = "{}_DataReduction.pdf".format(dataset.file_key)
    pdfPath = Path(dataset.tgzfile).parent / pdfFile
    if not pdfPath.is_file():
        tar = tarfile.open(dataset.tgzfile)
        r = re.compile("(.*_RPT_COR_DataReduction_.*.pdf)")
        report = list(filter(r.match, dataset.list))
        try:
            printlog(report[0], olog=olog)
            if len(report) == 0:
                printlog("Dataset does not contain DRP report.", olog=olog)
                raise Exception("Dataset does not contain DRP report.")
            if len(report) > 1:
                printlog("Multiple reports in dataset", olog=olog)
                raise Exception("Multiple reports in dataset")
            printlog("Extracting report from .tgz file ...", olog)
            # with tar.extractfile(report[0]) as fin:
            #   with open(pdfPath,'wb') as fout:
            #     for line in fin:
            #       fout.write(line)
            tar.extract(report[0], path=output_folder)
        except:
            printlog("Probably DRP report not in tgz file...", olog=olog)
        tar.close()

    return


# ======================================================================
# FROM STARPROPERTIES.PY WITHIN PYCHEOPS AND ADAPTED

#   pycheops - Tools for the analysis of data from the ESA CHEOPS mission
#
#   Copyright (C) 2018  Dr Pierre Maxted, Keele University
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
"""
CustomStarProperties
==============
 Object class to obtain/store observed properties of a star and to infer
 parameters such as radius and density.

"""

# from __future__ import (absolute_import, division, print_function,
#                                 unicode_literals)
# import numpy as np
# from astropy.table import Table
# from astropy.coordinates import SkyCoord
# import requests
# from .core import load_config
# from pathlib import Path
# from os.path import getmtime
# from time import localtime, mktime
# from uncertainties import ufloat, UFloat
# from .ld import stagger_power2_interpolator, atlas_h1h2_interpolator
# from .ld import phoenix_h1h2_interpolator
# from numpy.random import normal


class CustomStarProperties(object):
    """
    CHEOPS StarProperties object

    The observed properties T_eff, log_g and [Fe/H] are obtained from
    DACE or SWEET-Cat, or can be specified by the user.

    Set match_arcsec=None to skip extraction of parameters from SWEET-Cat.

    By default properties are obtained from SWEET-Cat.

    Set dace=True to obtain parameters from the stellar properties table at
    DACE.

    User-defined properties are specified either as a ufloat or as a 2-tuple
    (value, error), e.g., teff=(5000,100).

    User-defined properties over-write values obtained from SWEET-Cat or DACE.

    The stellar density is estimated using an linear relation between log(rho)
    and log(g) derived using the method of Moya et al. (2018ApJS..237...21M)

    Limb darkening parameters in the CHEOPS band are interpolated from Table 2
    of Maxted (2018A&A...616A..39M). The error on these parameters is
    propogated from the errors in Teff, log_g and [Fe/H] plus an additional
    error of 0.01 for h_1 and 0.05 for h_2, as recommended in Maxted (2018).
    If [Fe/H] for the star is not specified, the value 0.0 +/- 0.3 is assumed.

    If the stellar parameters are outside the range covered by Table 2 of
    Maxted (2018), then the results from ATLAS model from Table 10 of Claret
    (2019RNAAS...3...17C) are used instead. For stars cooler than 3500K the
    PHOENIX models for solar metalicity  from Table 5 of Claret (2019) are
    used. The parameters h_1 and h_2 are both given nominal errors of 0.1 for
    both ATLAS model, and 0.15 for PHOENIX models.

    """

    def __init__(
        self,
        identifier,
        force_download=False,
        dace=False,
        match_arcsec=None,
        configFile=None,
        teff=None,
        logg=None,
        metal=None,
        passband="CHEOPS",
        verbose=True,
    ):

        self.identifier = identifier
        coords = SkyCoord.from_name(identifier)
        self.ra = coords.ra.to_string(precision=2, unit="hour", sep=":", pad=True)
        self.dec = coords.dec.to_string(
            precision=1, sep=":", unit="degree", alwayssign=True, pad=True
        )

        config = load_config(configFile)
        _cache_path = config["DEFAULT"]["data_cache_path"]
        sweetCatPath = Path(_cache_path, "sweetcat.tsv")

        if force_download:
            download_sweetcat = True
        elif dace:
            download_sweetcat = False
        elif sweetCatPath.is_file():
            file_age = mktime(localtime()) - getmtime(sweetCatPath)
            if file_age > int(config["SWEET-Cat"]["update_interval"]):
                download_sweetcat = True
            else:
                download_sweetcat = False
        else:
            download_sweetcat = True

        if download_sweetcat:
            url = config["SWEET-Cat"]["download_url"]
            req = requests.post(url)
            with open(sweetCatPath, "wb") as file:
                file.write(req.content)
            if verbose:
                print("SWEET-Cat data downloaded from \n {}".format(url))

        if dace:
            from dace.cheops import Cheops

            db = Cheops.query_catalog("stellar")
            cat_c = SkyCoord(
                db["obj_pos_ra_deg"], db["obj_pos_dec_deg"], unit="degree,degree"
            )
            idx, sep, _ = coords.match_to_catalog_sky(cat_c)
            if sep.arcsec[0] > match_arcsec:
                raise ValueError("No matching star in DACE stellar properties table")
            self.teff = ufloat(db["obj_phys_teff_k"][idx], 99)
            self.teff_note = "DACE"
            self.logg = ufloat(db["obj_phys_logg"][idx], 0.09)
            self.logg_note = "DACE"
            self.metal = ufloat(db["obj_phys_feh"][idx], 0.09)
            self.metal_note = "DACE"
            self.gaiadr2 = db["obj_id_gaiadr2"][idx]

        else:
            names = [
                "star",
                "hd",
                "ra",
                "dec",
                "vmag",
                "e_vmag",
                "par",
                "e_par",
                "parsource",
                "teff",
                "e_teff",
                "logg",
                "e_logg",
                "logglc",
                "e_logglc",
                "vt",
                "e_vt",
                "metal",
                "e_metal",
                "mass",
                "e_mass",
                "author",
                "source",
                "update",
                "comment",
            ]
            sweetCat = Table.read(
                sweetCatPath,
                format="ascii.no_header",
                delimiter="\t",
                fast_reader=False,
                names=names,
                encoding="utf-8",
            )

            if match_arcsec is None:
                entry = None
            else:
                cat_c = SkyCoord(sweetCat["ra"], sweetCat["dec"], unit="hour,degree")
                idx, sep, _ = coords.match_to_catalog_sky(cat_c)
                if sep.arcsec[0] > match_arcsec:
                    raise ValueError("No matching star in SWEET-Cat")
                entry = sweetCat[idx]

            try:
                self.teff = ufloat(float(entry["teff"]), float(entry["e_teff"]))
                self.teff_note = "SWEET-Cat"
            except:
                self.teff = None
            try:
                self.logg = ufloat(float(entry["logg"]), float(entry["e_logg"]))
                self.logg_note = "SWEET-Cat"
            except:
                self.logg = None
            try:
                self.metal = ufloat(float(entry["metal"]), float(entry["e_metal"]))
                self.metal_note = "SWEET-Cat"
            except:
                self.metal = None

        # User defined values
        if teff:
            self.teff = teff if isinstance(teff, UFloat) else ufloat(*teff)
            self.teff_note = "User"
        if logg:
            self.logg = logg if isinstance(logg, UFloat) else ufloat(*logg)
            self.logg_note = "User"
        if metal:
            self.metal = metal if isinstance(metal, UFloat) else ufloat(*metal)
            self.metal_note = "User"

        # log rho from log g using method of Moya et al.
        # (2018ApJS..237...21M). Accuracy is 4.4%
        self.logrho = None
        if self.logg:
            if (self.logg.n > 3.697) and (self.logg.n < 4.65):
                logrho = -7.352 + 1.6580 * self.logg
                self.logrho = ufloat(logrho.n, np.hypot(logrho.s, 0.044))

        self.h_1 = None
        self.h_2 = None
        self.ld_ref = None
        if self.teff and self.logg:
            metal = self.metal if self.metal else ufloat(0, 0.3)
            power2 = stagger_power2_interpolator(passband=passband)
            _, _, h_1, h_2 = power2(self.teff.n, self.logg.n, metal.n)
            if not np.isnan(h_1):
                self.ld_ref = "Stagger"
                Xteff = np.random.normal(self.teff.n, self.teff.s, 256)
                Xlogg = np.random.normal(self.logg.n, self.logg.s, 256)
                Xmetal = np.random.normal(metal.n, metal.s, 256)
                X = power2(Xteff, Xlogg, Xmetal)
                # Additional error derived in Maxted, 2019
                e_h_1 = np.hypot(0.01, np.sqrt(np.nanmean((X[:, 2] - h_1) ** 2)))
                e_h_2 = np.hypot(0.05, np.sqrt(np.nanmean((X[:, 3] - h_2) ** 2)))
                self.h_1 = ufloat(round(h_1, 3), round(e_h_1, 3))
                self.h_2 = ufloat(round(h_2, 3), round(e_h_2, 3))
            if self.ld_ref is None:
                atlas = atlas_h1h2_interpolator()
                h_1, h_2 = atlas(self.teff.n, self.logg.n, metal.n)
                if not np.isnan(h_1):
                    self.h_1 = ufloat(round(h_1, 3), 0.1)
                    self.h_2 = ufloat(round(h_2, 3), 0.1)
                    self.ld_ref = "ATLAS"
            if self.ld_ref is None:
                phoenix = phoenix_h1h2_interpolator()
                h_1, h_2 = phoenix(self.teff.n, self.logg.n)
                if not np.isnan(h_1):
                    self.h_1 = ufloat(round(h_1, 3), 0.15)
                    self.h_2 = ufloat(round(h_2, 3), 0.15)
                    self.ld_ref = "PHOENIX-COND"

    def __repr__(self):
        s = "Identifier : {}\n".format(self.identifier)
        s += "Coordinates: {} {}\n".format(self.ra, self.dec)
        if self.teff:
            s += "T_eff : {:5.0f} +/- {:3.0f} K   [{}]\n".format(
                self.teff.n, self.teff.s, self.teff_note
            )
        if self.logg:
            s += "log g : {:5.2f} +/- {:0.2f}    [{}]\n".format(
                self.logg.n, self.logg.s, self.logg_note
            )
        if self.metal:
            s += "[M/H] : {:+5.2f} +/- {:0.2f}    [{}]\n".format(
                self.metal.n, self.metal.s, self.metal_note
            )
        if self.logrho:
            s += "log rho : {:5.2f} +/- {:0.2f}  (solar units)\n".format(
                self.logrho.n, self.logrho.s
            )
        if self.ld_ref:
            s += "h_1 : {:5.3f} +/- {:0.3f}     [{}]\n".format(
                self.h_1.n, self.h_1.s, self.ld_ref
            )
            s += "h_2 : {:5.3f} +/- {:0.3f}     [{}]\n".format(
                self.h_2.n, self.h_2.s, self.ld_ref
            )
        return s


# ======================================================================
def quick_save_params(out_file, params, bjd_ref):

    of = open(out_file, "w")
    of.write("# param value error vary unit\n")
    for p in params:
        stderr = params[p].stderr
        if stderr is None:
            stderr = 0.0
        try:
            if p[0] in ["d", "r", "g"]:
                unit = params_units["det"]
            else:
                unit = params_units[p]
                if p == "T_0":
                    unit = "{}{}".format(unit, bjd_ref)
        except:
            unit = "-"
        line = "{:20s} {:20.10f} {:20.10f} {:6s} {:>20s}\n".format(
            p, params[p].value, stderr, str(params[p].vary), unit
        )
        of.write(line)
    of.close()

    return


def quick_save_params_ultranest(out_file, params, bjd_ref):

    of = open(out_file, "w")
    of.write("# param value error vary unit\n")
    for p in params:
        stderr = params[p].stderr
        if stderr is None:
            stderr = 0.0
        try:
            if p[0] in ["d", "r", "g"]:
                unit = params_units["det"]
            else:
                unit = params_units[p]
                if p == "T_0":
                    unit = "{}{}".format(unit, bjd_ref)
        except:
            unit = "-"
        line = "{:20s} {:20.10f} {:20.10f} {:6s} {:>20s}\n".format(
            p, params[p].value, stderr, str(params[p].vary), unit
        )
        of.write(line)
    of.close()

    return


# ======================================================================
def quick_save_params_ultra(
    out_file, star_inputs, planet_inputs, params_lm_loop, results, bjd_ref, mod="median"
):
    parlist = [
        "T_0",
        "P",
        "D",
        "W",
        "b",
        "f_c",
        "f_s",
        "h_1",
        "h_2",
        "t_exp",
        "n_over",
        "c",
        "dfdbg",
        "dfdcontam",
        "dfdy",
        "d2fdy2",
        "dfdt",
        "d2fdt2",
        "dfdsinphi",
        "dfdcosphi",
        "dfdsin2phi",
        "dfdcos2phi",
        "dfdsin3phi",
        "dfdcos3phi",
        "k",
        "aR",
        "sini",
        "logrho",
        "e",
        "q_1",
        "q_2",
        "log_sigma",
        "sigma_w",
        "inc",
    ]

    params = {}

    of = open(out_file, "w")
    of.write("# param value error_mean error_up error_low vary unit\n")
    for p in parlist:
        params[p] = {}
        if p in results["paramnames"]:
            res_idx = results["paramnames"].index(p)
            params[p]["value"] = results["posterior"][mod][res_idx]
            params[p]["vary"] = True
            params[p]["err_mean"] = (
                results["posterior"]["errup"][res_idx]
                + results["posterior"]["errlo"][res_idx]
            ) / 2
            params[p]["err_up"] = results["posterior"]["errup"][res_idx]
            params[p]["err_low"] = results["posterior"]["errlo"][res_idx]
            # params[p]["stdev"] = results["stdev"][res_idx]
        else:
            is_parameter = False
            if p in list(params_lm_loop.keys()):
                inpars = params_lm_loop
                is_parameter = True
            elif p in list(planet_inputs.keys()):
                inpars = planet_inputs
            elif p in list(star_inputs.keys()):
                inpars = star_inputs
            else:
                raise (
                    f"{p} key not found in any of the input or the fitting results key list"
                )
            if is_parameter:
                params[p]["value"] = params_lm_loop[p].value
                params[p]["vary"] = False
                params[p]["err_mean"] = params_lm_loop[p].stderr
                params[p]["err_up"] = params_lm_loop[p].stderr
                params[p]["err_low"] = params_lm_loop[p].stderr
            else:
                try:
                    params[p]["value"] = inpars[p + "_user_data"].n
                    params[p]["vary"] = False
                    params[p]["err_mean"] = inpars[p + "_user_data"].s
                    params[p]["err_up"] = inpars[p + "_user_data"].s
                    params[p]["err_low"] = inpars[p + "_user_data"].s
                except AttributeError:
                    params[p]["value"] = 0
                    params[p]["vary"] = False
                    params[p]["err_mean"] = 0
                    params[p]["err_up"] = 0
                    params[p]["err_low"] = 0
            # params[p]["stdev"] = inpars[p + "_user_data"].s

            # TODO continue if parameters is not fitted with ultranest and vary == False
        # if params[p]["stdev"] is None:
        #     params[p]["stdev"] = 0

        if p == "T_0":
            unit = "{} - {}".format("BJD_TDB", bjd_ref)
        elif p == "P" or p == "t_exp":
            unit = "d"
        elif p == "W":
            unit = "day/P"
        elif p == "D" or p == "c" or p == "sigma_w":
            unit = "flux"
        elif p == "logrho":
            unit = "log(rho_sun)"
        elif p == "inc":
            unit = "deg"
        else:
            unit = "-"
        line = "{:20s} {:20.10f} {:20.10f} {:20.10f} {:20.10f} {:6s} {:>20s}\n".format(
            p,
            params[p]["value"],
            params[p]["err_mean"],
            params[p]["err_up"],
            params[p]["err_low"],
            str(params[p]["vary"]),
            unit,
        )
        of.write(line)
    of.close()

    return


# ======================================================================


def u1u2_to_q1q2(u1, u2):

    u12 = u1 + u2
    q1 = u12 * u12
    q2 = u1 / (2.0 * u12)

    return q1, q2


def q1q2_to_u1u2(q1, q2):

    twoq2 = 2.0 * q2
    sq1 = np.sqrt(q1)
    u1 = sq1 * twoq2
    u2 = sq1 * (1.0 - twoq2)

    return u1, u2


def generate_private_key(path):
    key = Fernet.generate_key()
    path = os.path.join(path, "")

    with open(path + "private", "wb") as private:
        private.write(key)
