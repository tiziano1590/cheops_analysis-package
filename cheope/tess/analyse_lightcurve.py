import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from astropy.io import fits

from uncertainties import ufloat, UFloat

import cheope.pyconstants as cst
import cheope.pycheops_analysis as pyca
import cheope.linear_ephemeris as lep
from cheope.parameters import ReadFile

from pycheops.instrument import CHEOPS_ORBIT_MINUTES

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


class ReadFits:
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

    def load_fits_file(self, olog=None):

        info = {}

        printlog(
            "Reading fits file: {}".format(self.visit_args["file_fits"]), olog=olog
        )
        # load fits file and extract needed data and header keywords
        with fits.open(self.visit_args["file_fits"]) as hdul:
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

    def plot_lightcurve(self):
        matplotlib.use("TkAgg")

        example, info = self.load_fits_file()

        T_ref = self.planet_args["T_ref"]
        P = self.planet_args["P_user_data"]
        Wd = self.planet_args["W"] * P
        vdurh = self.visit_args["single_duration_hour"]
        if vdurh is None:
            vdurh = 1.5 * Wd.n * cst.day2hour + 3.0 * CHEOPS_ORBIT_MINUTES * cst.min2day
        vdur = vdurh / cst.day2hour
        hdur = 0.5 * vdur
        vdur_co = vdur * cst.day2min / CHEOPS_ORBIT_MINUTES

        btjd = info["BTJD"]

        t = example["TIME"] + btjd
        emin = np.rint((np.min(t) - T_ref.n) / P.n)
        x = T_ref.n + P.n * emin
        if x < np.min(t):
            emin += 1
        emax = np.rint((np.max(t) - T_ref.n) / P.n)
        x = T_ref.n + P.n * emax
        if x > np.max(t):
            emax -= 1
        epochs = np.arange(emin, emax + 1, 1)

        # printlog("t min: {:.5f}".format(np.min(t)))
        # printlog("t max: {:.5f}".format(np.max(t)))

        transits = []
        t0s = []
        for i_epo, epo in enumerate(epochs):

            bjd_lin = T_ref.n + P.n * epo

            sel = np.logical_and(t >= bjd_lin - hdur, t < bjd_lin + hdur)
            nsel = np.sum(sel)
            wsel = np.logical_and(
                t >= bjd_lin - (Wd.n * 0.5), t < bjd_lin + (Wd.n * 0.5)
            )
            nwsel = np.sum(wsel)
            if nsel > 0 and nwsel > 3:
                tra = {}
                tra["epoch"] = epo
                tra["data"] = {}
                for k, v in example.items():
                    tra["data"][k] = v[sel]
                bjdref = int(np.min(tra["data"]["TIME"]) + btjd)
                tra["bjdref"] = bjdref
                Tlin = bjd_lin - bjdref
                tra["T_0"] = Tlin
                tra["T_0_bounds"] = [Tlin - 0.5 * Wd.n, Tlin + 0.5 * Wd.n]
                tra["T_0_user_data"] = ufloat(Tlin, 0.5 * Wd.n)
                transits.append(tra)
            else:
                pass

            t0s.append(tra["bjdref"])

        t0s = np.array(t0s) - 2457000
        print(f"Aperture is: {self.visit_args['aperture']}")
        # print(t0s)

        if self.visit_args["aperture"].lower() == "sap":
            flux_lab = "SAP_FLUX"
        elif self.visit_args["aperture"].lower() == "pdc":
            flux_lab = "PDCSAP_FLUX"

        markers, caps, bars = plt.errorbar(
            example["TIME"],
            example[flux_lab],
            yerr=example[f"{flux_lab}_ERR"],
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
            ymax=10 * max(example[flux_lab]),
            color="firebrick",
            linewidth=0.5,
            linestyles="dashed",
        )

        for cap in caps:
            cap.set_markeredgewidth(0.3)

        [bar.set_alpha(0.5) for bar in bars]
        [cap.set_alpha(0.5) for cap in caps]
        plt.ylim(
            min(example[flux_lab]) - max(example[f"{flux_lab}_ERR"]),
            max(example[flux_lab] + max(example[f"{flux_lab}_ERR"])),
        )
        plt.xlabel("Time (BTJD - 2457000)")
        plt.ylabel("Flux")
        plt.show()
