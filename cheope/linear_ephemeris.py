#!/usr/bin/env python
# coding: utf-8

# # Linear Epehemeris window
import matplotlib.pyplot as plt

import numpy as np

from astropy.time import Time
import scipy.optimize as sciopt
import sklearn.linear_model as sklm
import scipy.odr as sodr
import statsmodels.api as sm

# Epoch or Transit number
def compute_epoch(tref, pref, t):

    epo = np.rint((t - tref) / pref)

    return epo


# Transit time from linear ephemeris
def linear_transit_time(tref, pref, epo):

    tlin = tref + epo * pref

    return tlin


# Predict transit time (linear) and error (w/ MC) on observing window
def transit_prediction(tref, etref, pref, epref, t, n_mc=100):

    epo = compute_epoch(tref, pref, t)
    t_lin = linear_transit_time(tref, pref, epo)

    tref_mc = np.random.normal(loc=tref, scale=etref, size=n_mc)
    pref_mc = np.random.normal(loc=pref, scale=epref, size=n_mc)
    epo_mc = np.repeat(epo, n_mc)

    t_mc = linear_transit_time(tref_mc, pref_mc, epo_mc)

    t_rms = np.percentile(np.abs(t_mc - t_lin), 68.27, interpolation="midpoint")
    t_mean = np.mean(t_mc)
    t_std = np.std(t_mc, ddof=1)

    t_std_prop = np.sqrt(etref * etref + (epo * epref) * (epo * epref))

    return epo, t_lin, t_rms, t_mean, t_std, t_std_prop


# Computes prediction and plot
def plot_transit_prediction(tref, etref, pref, epref, t, dur_min=None, n_mc=100):

    TimeRef = Time(tref, format="jd", scale="tdb")

    print(
        "Linear ephemeris  = {:20.8f} ({:10.8f}) BJD_TDB + N x {:15.8f} ({:10.8f}) d".format(
            tref, etref, pref, epref
        )
    )
    print("Tref              = {:20.8f} BJD_TDB = {:s}".format(tref, TimeRef.iso))
    print("Date & Time Input = {:20.8f} BJD_TDB = {:s}".format(t.jd, t.iso))

    epo, t_lin, t_rms, t_mean, t_std, t_std_prop = transit_prediction(
        tref, etref, pref, epref, t.jd, n_mc=n_mc
    )

    rms_min = t_rms * 1440.0
    std_min = t_std * 1440.0
    std_prop_min = t_std_prop * 1440.0

    print("Predicted times at N = {}".format(epo))
    print("                T0")
    print(
        "linear ephem. = {} +/- {} (rms) [{} (error prop.)]".format(
            t_lin, t_rms, t_std_prop
        )
    )
    print("mean   ephem. = {} +/- {} (std)".format(t_mean, t_std))
    print(
        "Error window  = {} min (rms) , {} min (std) , {} min (error prop)".format(
            rms_min, std_min, std_prop_min
        )
    )

    max_std = np.max([rms_min, std_min, std_prop_min])

    fig = plt.figure(figsize=(6, 2.5))

    lineph = "Lin. eph = {} ({}) + N x {} ({})\n".format(tref, etref, pref, epref)
    timest = "at {} ({}) with N = {}".format(t.iso, t.jd, epo)
    plt.title(lineph + timest, loc="center", fontsize=8)

    ymin = 0.0
    ymax = ymin + 1.0
    plt.fill_betweenx(
        [ymin, ymax],
        -std_prop_min,
        std_prop_min,
        color="lightgray",
        alpha=1.0,
        zorder=6,
        label="rms",
    )
    ymin = ymax
    ymax = ymin + 1.0
    plt.fill_betweenx(
        [ymin, ymax], -rms_min, rms_min, color="C0", alpha=0.6, zorder=7, label="std"
    )
    ymin = ymax
    ymax = ymin + 1.0
    plt.fill_betweenx(
        [ymin, ymax],
        -std_min,
        std_min,
        color="C1",
        alpha=0.6,
        zorder=8,
        label="err. prop.",
    )

    xlim = max_std * 1.03
    if dur_min is not None:
        hdur = 0.5 * dur_min
        pre_x = [-dur_min, -hdur]
        tra_x = [-hdur, hdur]
        pos_x = [hdur, dur_min]
        dur_y = [0.0, 0.0]
        lw_d = 1.5
        plt.plot(
            pre_x,
            dur_y,
            color="black",
            marker="None",
            ls="--",
            lw=lw_d,
            alpha=1.0,
            label="oot",
            zorder=9,
        )
        plt.plot(
            pos_x,
            dur_y,
            color="black",
            marker="None",
            ls="--",
            lw=lw_d,
            alpha=1.0,
            zorder=9,
        )
        plt.plot(
            tra_x,
            dur_y,
            color="black",
            marker="None",
            ls="-",
            lw=lw_d,
            alpha=1.0,
            label="transit",
            zorder=9,
        )
        xlim = np.max([dur_min, max_std]) * 1.03
    plt.xlim(-xlim, xlim)

    plt.yticks([])
    plt.xlabel("minutes from predicted transit time")
    plt.legend(
        loc="center left",
        bbox_to_anchor=(1.025, 0.5),
        fontsize=8,
        ncol=1,
        frameon=False,
        facecolor="white",
        framealpha=1,
    )

    plt.show()

    plt.close(fig)

    return


def calculate_epoch(t, tref, pref):

    epo = compute_epoch(tref, pref, t)

    return epo


def lstsq_fit(x, y, yerr):

    A = np.vstack((np.ones_like(x), x)).T
    C = np.diag(yerr * yerr)
    cov = np.linalg.inv(np.dot(A.T, np.linalg.solve(C, A)))
    b_ls, m_ls = np.dot(cov, np.dot(A.T, np.linalg.solve(C, y)))
    coeff_err = np.sqrt(np.diag(cov))

    return b_ls, m_ls, coeff_err


def linear_model(par, x):

    linmod = par[0] + x * par[1]

    return linmod


# ==============================================================================


def res_linear_model(par, x, y, ey=None):

    linmod = linear_model(par, x)
    if ey is not None:
        wres = (y - linmod) / ey
    else:
        wres = y - linmod

    return wres


# ==============================================================================


def chi2r_linear_model(par, x, y, ey=None):

    wres = res_linear_model(par, x, y, ey)
    nfit = np.shape(par)[0]
    ndata = np.shape(x)[0]
    dof = ndata - nfit
    chi2r = np.sum(np.power(wres, 2)) / np.float64(dof)

    return chi2r


def compute_lin_ephem(T0, eT0=None, epoin=None, modefit="wls"):

    nT0 = np.shape(T0)[0]

    TP_err = [None, None]

    if eT0 is None:
        errTT = np.ones((nT0)) / 86400.0
    else:
        errTT = np.array(eT0)

    if epoin is None:
        # Tref0 = T0[0]
        # Tref0 = np.percentile(T0, 50., interpolation='midpoint')
        Tref0 = T0[int(0.5 * nT0)]
        dT = [np.abs(T0[i + 1] - T0[i]) for i in range(nT0 - 1)]
        Pref0 = np.percentile(np.array(dT), 50.0, interpolation="midpoint")
        epo = calculate_epoch(T0, Tref0, Pref0)
    else:
        epo = epoin
        Tref0, Pref0, _ = lstsq_fit(np.array(epo), np.array(T0), errTT)

    if modefit in ["optimize", "minimize"]:
        # SCIPY.OPTIMIZE.MINIMIZE
        optres = sciopt.minimize(
            chi2r_linear_model,
            [Tref0, Pref0],
            method="nelder-mead",
            args=(epo, T0, errTT),
        )
        Tref, Pref = optres.x[0], optres.x[1]
        TP_err = [0.0, 0.0]
        epo = calculate_epoch(T0, Tref, Pref)

    elif modefit in ["sklearn", "linear_model"]:
        # SKLEARN.LINEAR_MODEL
        sk_mod = sklm.LinearRegression()
        sk_mod.fit(epo.reshape((-1, 1)), T0)
        Tref, Pref = np.asscalar(np.array(sk_mod.intercept_)), np.asscalar(
            np.array(sk_mod.coef_)
        )
        epo = calculate_epoch(T0, Tref, Pref)

    elif modefit == "odr":
        # SCIPY.ODR
        odr_lm = sodr.Model(linear_model)
        if eT0 is not None:
            odr_data = sodr.RealData(epo, T0, sy=eT0)
        else:
            odr_data = sodr.RealData(epo, T0)
        init_coeff = [Tref0, Pref0]
        odr_mod = sodr.ODR(odr_data, odr_lm, beta0=init_coeff)
        odr_out = odr_mod.run()
        Tref, Pref = odr_out.beta[0], odr_out.beta[1]
        TP_err = odr_out.sd_beta

    else:  # wls
        X = sm.add_constant(epo)
        if eT0 is not None:
            wls = sm.WLS(T0, X, weights=1.0 / (eT0 * eT0)).fit()
        else:
            wls = sm.WLS(T0, X).fit()
        Tref, Pref = wls.params[0], wls.params[1]
        TP_err = wls.bse

    return epo, Tref, Pref, TP_err


# Testing
def testing():
    tref, etref = 2458001.72138, 0.00016
    pref, epref = 14.893291, 0.000025

    t_test = Time("2020-04-18 13:30", scale="tdb", format="iso")

    n_repeat = 10000

    plot_transit_prediction(tref, etref, pref, epref, t_test, n_mc=n_repeat)

    return
