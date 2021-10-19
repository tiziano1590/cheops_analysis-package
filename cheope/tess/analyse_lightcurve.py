import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from astropy.io import fits

import cheope.pyconstants as cst
import cheope.pycheops_analysis as pyca
import cheope.linear_ephemeris as lep
from cheope.parameters import ReadFile

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
        example = self.load_fits_file()

        markers, caps, bars = plt.errorbar(
            example[0]["TIME"],
            example[0]["SAP_FLUX"],
            yerr=example[0]["SAP_FLUX_ERR"],
            fmt="o",
            markersize=0.3,
            capsize=0.2,
            color="k",
            ecolor="gray",
            elinewidth=0.2,
        )

        for cap in caps:
            cap.set_markeredgewidth(0.3)

        [bar.set_alpha(0.5) for bar in bars]
        [cap.set_alpha(0.5) for cap in caps]
        plt.xlabel("Time (BTJD - 2457000)")
        plt.ylabel("Flux")
        plt.show()
