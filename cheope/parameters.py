import os
import yaml
import numpy as np

from uncertainties import ufloat, UFloat
from uncertainties import umath as um

from pycheops import StarProperties
from cheope.pycheops_analysis import printlog

import cheope.pyconstants as cst


class ReadFile:
    def __init__(self, input_file, multivisit=False):
        self.visit_args = {}
        self.star_args = {}
        self.planet_args = {}
        self.emcee_args = {}
        self.ultranest_args = {}
        self.read_file_status = []
        self.yaml_input = {}
        self.multivisit = multivisit

        self.bound_fac = 10

        self.visit_keys = [
            "main_folder",
            "visit_number",
            "file_key",
            "passband",
            "aperture",
            "dace",
            "shape",
            "seed",
            "glint",
            "clipping",
            "unroll",
            "nroll",
            "GP",
            "single_duration_hour",
        ]

        self.star_keys = [
            "star_name",
            "dace",
            "Rstar",
            "Mstar",
            "teff",
            "logg",
            "feh",
            "h_1",
            "h_2",
            "match_arcsec",
        ]

        self.planet_keys = [
            "P",
            "D",
            "inc",
            "aRs",
            "ecc",
            "w",
            "T_0",
            "Kms",
            "W",
            "b",
            "k",
            "Rp",
            "l_3"
        ]

        self.emcee_keys = [
            "nwalkers",
            "nprerun",
            "nsteps",
            "nburn",
            "nthin",
            "progress",
            "nthreads",
        ]

        self.ultranest_keys = [
            "live_points",
            "tol",
            "cluster_num_live_points",
            "resume",
            "adaptive_nsteps",
        ]

        if os.path.exists(input_file) and os.path.isfile(input_file):
            with open(input_file) as in_f:
                self.yaml_input = yaml.load(in_f, Loader=yaml.FullLoader)

            # merge list keys with a unique set of values
            self.visit_keys = list(set(self.visit_keys + list(self.yaml_input.keys())))
            self.star_keys = list(
                set(self.star_keys + list(self.yaml_input["star"].keys()))
            )
            self.planet_keys = list(
                set(self.planet_keys + list(self.yaml_input["planet"].keys()))
            )
            self.emcee_keys = list(
                set(self.emcee_keys + list(self.yaml_input["emcee"].keys()))
            )

            self.ultranest_keys = list(
                set(self.ultranest_keys + list(self.yaml_input["ultranest"].keys()))
            )

        else:
            self.read_file_status.append(f"NOT VALID INPUT FILE:\n{input_file}")

        self.visit_pars()
        self.star_pars()
        self.planet_pars()
        self.emcee_pars()
        self.ultranest_pars()

    def visit_pars(self):
        # -- visit_args
        for key in self.visit_keys:
            self.visit_args[key] = self.yaml_input.get(key, "None")
            self.read_file_status.append(self.err_msg(key, self.yaml_input))

        self.apply_visit_conditions()

    def star_pars(self):

        for key in self.star_keys:
            inval = self.yaml_input["star"].get(key)  # get the input value

            if (inval is None) or isinstance(inval, str) or isinstance(inval, bool):
                self.star_args[key] = inval
                self.star_args[key + "_fit"] = False
            elif isinstance(inval, list):
                self.star_args[key] = ufloat(inval[0], inval[1])
                self.star_args[key + "_fit"] = False
                self.star_args[key + "_bounds"] = [
                    inval[0] - self.bound_fac * inval[1],
                    inval[0] + self.bound_fac * inval[1],
                ]
                self.star_args[key + "_user_data"] = ufloat(inval[0], inval[1])
            elif isinstance(inval, dict):
                val = inval.get("value")
                fit = inval.get("fit")
                if val != None:
                    self.star_args[key] = ufloat(val[0], val[1]).n
                else:
                    self.star_args[key] = None
                self.star_args[key + "_fit"] = fit
                if (
                    inval.get("bounds") == None
                    and np.isfinite(val[1])
                    and abs(val[1]) > 0
                ):
                    self.star_args[key + "_bounds"] = [
                        val[0] - self.bound_fac * val[1],
                        val[0] + self.bound_fac * val[1],
                    ]
                elif inval.get("bounds") != None:
                    self.star_args[key + "_bounds"] = inval.get("bounds")
                else:
                    self.star_args[key + "_bounds"] = [-np.inf, np.inf]
                if val != None:
                    self.star_args[key + "_user_data"] = ufloat(val[0], val[1])
                else:
                    self.star_args[key + "_user_data"] = None
            else:
                self.read_file_status.append(
                    f"{key} should be either a list or a dictionary."
                )

            self.read_file_status.append(self.err_msg(key, self.yaml_input["star"]))

        self.apply_star_conditions()

    def planet_pars(self):

        for key in self.planet_keys:
            inval = self.yaml_input["planet"].get(key)  # get the input value
            if (inval is None) or isinstance(inval, str):
                self.planet_args[key] = inval
                self.planet_args[key + "_bounds"] = [-np.inf, np.inf]
                self.planet_args[key + "_fit"] = True
                self.planet_args[key + "_user_data"] = None
            elif isinstance(inval, list):
                self.planet_args[key] = ufloat(inval[0], inval[1])
                self.planet_args[key + "_fit"] = True
                self.planet_args[key + "_bounds"] = [
                    inval[0] - self.bound_fac * inval[1],
                    inval[0] + self.bound_fac * inval[1],
                ]
                self.planet_args[key + "_user_data"] = ufloat(inval[0], inval[1])
            elif isinstance(inval, dict):
                val = inval.get("value")
                if isinstance(val, list):
                    val = val[0]
                fit = inval.get("fit")
                priors = inval.get("priors")
                bounds = inval.get("bounds")
                self.planet_args[key] = val
                self.planet_args[key + "_fit"] = fit
                if bounds != None:
                    self.planet_args[key + "_bounds"] = bounds
                else:
                    self.planet_args[key + "_bounds"] = [-np.inf, np.inf]
                if priors != None:
                    self.planet_args[key + "_user_data"] = ufloat(priors[0], priors[1])
                else:
                    self.planet_args[key + "_user_data"] = None

            else:
                self.read_file_status.append(
                    f"{key} should be either a list or a dictionary."
                )

            self.read_file_status.append(self.err_msg(key, self.yaml_input["planet"]))

        if not self.multivisit:
            self.apply_planet_conditions()

        if self.multivisit:
            if self.planet_args["P_ref_bounds"][0] <= 0:
                printlog("Found P_ref lower bound <= 0. Setting to 1e-4 days")
                self.planet_args["P_ref_bounds"][0] = 1e-4

    def emcee_pars(self):

        self.emcee_args["nwalkers"] = 128
        self.emcee_args["nprerun"] = 512
        self.emcee_args["nsteps"] = 1280
        self.emcee_args["nburn"] = 256
        self.emcee_args["nthin"] = 1
        self.emcee_args["nthreads"] = 1
        self.emcee_args["progress"] = "overwrite"
        self.emcee_args["backend"] = None
        self.emcee_args["backend_type"] = "overwrite"

        for key in self.emcee_keys:
            self.emcee_args[key] = self.yaml_input["emcee"].get(
                key, self.emcee_args[key]
            )

    def ultranest_pars(self):
        self.ultranest_args["live_points"] = 500
        self.ultranest_args["tol"] = 0.5
        self.ultranest_args["cluster_num_live_points"] = 40
        self.ultranest_args["resume"] = False
        self.ultranest_args["adaptive_nsteps"] = False

        for key in self.ultranest_keys:
            self.ultranest_args[key] = self.yaml_input["ultranest"].get(
                key, self.ultranest_args[key]
            )

    def apply_visit_conditions(self):

        # main folder
        self.visit_args["main_folder"] = os.path.abspath(self.visit_args["main_folder"])

        # file_key
        self.visit_args["file_key"] = self.visit_args["file_key"].strip()

        # aperture
        apertures = ["DEFAULT", "OPTIMAL", "RINF", "RSUP", "SAP", "PDC"]
        self.visit_args["aperture"] = self.visit_args["aperture"].strip().upper()
        if not self.visit_args["aperture"] in apertures:
            self.visit_args["aperture"] = "DEFAULT"
            self.read_file_status.append("aperture set to default: DEFAULT")

        # shape
        shapes = ["fit", "fix"]
        self.visit_args["shape"] = self.visit_args["shape"].strip().lower()
        if not self.visit_args["shape"] in shapes:
            self.visit_args["shape"] = "fit"
            self.read_file_status.append("shape set to default: fit")

        # seed
        try:
            self.visit_args["seed"] = int(self.visit_args["seed"])
        except:
            self.read_file_status.append(
                "seed must be a positive integer. Set to default 42."
            )
            self.visit_args["seed"] = 42

        # glint_type
        glints = ["moon", "glint"]

        if self.visit_args["glint"] == "None" or not self.visit_args["glint"]["add"]:
            self.visit_args["glint"] = None
        else:
            self.visit_args["glint"]["type"] = (
                self.visit_args["glint"]["type"].strip().lower()
            )
            if not self.visit_args["glint"]["type"].lower() in glints:
                self.visit_args["glint"]["type"] = False

            self.visit_args["glint"]["moon"] = False
            if self.visit_args["glint"]["type"].lower() == "moon":
                self.visit_args["glint"]["moon"] = True

            if self.visit_args["glint"].get("scale") is not None:
                self.visit_args["glint"]["scale"] = tuple(
                    self.visit_args["glint"]["scale"]
                )
            else:
                self.visit_args["glint"]["scale"] = (0, 2)
            self.visit_args["glint"]["nspline"] = self.visit_args["glint"].get(
                "nspline", 48
            )
            self.visit_args["glint"]["binwidth"] = self.visit_args["glint"].get(
                "binwidth", 5
            )
            self.visit_args["glint"]["gapmax"] = self.visit_args["glint"].get(
                "gapmax", 5
            )

        # clip_outliers
        self.visit_args["clip_outliers"] = 5

        if self.visit_args["dace"] == "None":
            self.visit_args["dace"] = False

        try:
            co = round(self.yaml_input["clip_outliers"])
            if co < 1:
                co = 0
            self.visit_args["clip_outliers"] = co
        except:
            self.read_file_status.append(
                "clip_outliers must be a positive integer. Set to 5-sigma clip by default."
            )

    def apply_star_conditions(self):

        if self.star_args["match_arcsec"] == "None":
            self.star_args["match_arcsec"] = None

        star = StarProperties(
            self.star_args["star_name"],
            match_arcsec=self.star_args["match_arcsec"],
            teff=self.star_args["teff"],
            logg=self.star_args["logg"],
            metal=self.star_args["feh"],
            dace=self.visit_args["dace"],
        )

        if (self.star_args["h_1_fit"] == False) or (self.star_args["h_2_fit"] == False):
            if (
                self.yaml_input["star"].get("h_1") is None
                or self.yaml_input["star"].get("h_2") is None
            ):
                self.star_args["h_1"] = star.h_1.n
                self.star_args["h_2"] = star.h_2.n
                self.star_args["h_1_fit"] = False
                self.star_args["h_2_fit"] = False
                self.star_args["h_1_bounds"] = [0, 1]
                self.star_args["h_2_bounds"] = [0, 1]
                # priors_h1 = self.yaml_input["star"]["h_1"].get("priors")
                self.star_args["h_1_user_data"] = ufloat(star.h_1.n, star.h_1.s)
                self.star_args["h_2_user_data"] = ufloat(star.h_2.n, star.h_2.s)
        else:
            if (
                self.yaml_input["star"].get("h_1") is None
                or self.yaml_input["star"].get("h_2") is None
            ):
                self.star_args["h_1"] = star.h_1.n
                self.star_args["h_2"] = star.h_2.n
                self.star_args["h_1_fit"] = True
                self.star_args["h_2_fit"] = True
                self.star_args["h_1_bounds"] = [
                    # star.h_1.n - star.h_1.s,
                    # star.h_1.n + star.h_1.s,
                    0,
                    1,
                ]
                self.star_args["h_2_bounds"] = [
                    # star.h_2.n - star.h_2.s,
                    # star.h_2.n + star.h_2.s,
                    0,
                    1,
                ]
                self.star_args["h_1_user_data"] = ufloat(star.h_1.n, star.h_1.s)
                self.star_args["h_2_user_data"] = ufloat(star.h_2.n, star.h_2.s)

        # if self.yaml_input["star"].get("logrho") is None:
        #     try:
        #         self.star_args["logrho"] = star.logrho.n
        #     except AttributeError:
        #         self.star_args["logrho"] = star.logrho
        #     self.star_args["logrho_fit"] = True
        #     self.star_args["logrho_bounds"] = [-9, 6]
        #     self.star_args["logrho_user_data"] = star.logrho
        Mstar_input = self.yaml_input["star"].get("Mstar")
        Rstar_input = self.yaml_input["star"].get("Rstar")
        if (Mstar_input is not None) and (Rstar_input is not None):
            Ms = ufloat(Mstar_input[0], Mstar_input[1])
            Rs = ufloat(Rstar_input[0], Rstar_input[1])
            rho = Ms / (Rs ** 3)
            logrho = um.log10(rho)
            self.star_args["logrho"] = logrho.n
            self.star_args["logrho_user_data"] = logrho
        else:
            try:
                self.star_args["logrho"] = star.logrho.n
            except AttributeError:
                self.star_args["logrho"] = star.logrho
            self.star_args["logrho_user_data"] = star.logrho
        self.star_args["logrho_fit"] = True
        self.star_args["logrho_bounds"] = [-9, 6]

    def get_planet_value_or_user_data(self, key):

        if (self.planet_args["{}_user_data".format(key)] != None) and (
            isinstance(self.planet_args["{}_user_data".format(key)], UFloat)
        ):
            par = self.planet_args["{}_user_data".format(key)]
        else:
            par = self.planet_args["{}".format(key)]

        return par

    def apply_planet_conditions(self):

        planet_yaml = self.yaml_input["planet"]
        if "D" in planet_yaml:
            # if self.planet_args["D_user_data"] != None:
            #     # D = ufloat(
            #     #     self.planet_args["D_user_data"].n, self.planet_args["D_user_data"].s
            #     # )
            #     D = self.planet_args["D_user_data"]  # already in ufloat type!
            #     self.planet_args["D"] = D.n
            #     k = um.sqrt(D)
            # else:
            #     D = self.planet_args["D"]
            #     self.planet_args["D"] = D
            #     k = np.sqrt(D)
            D = self.planet_args["D"]
            # print(D)
            k = D**(1/2)
            
        elif "k" in planet_yaml:
            
            # if self.planet_args["k_user_data"] != None:
            #     # k = ufloat(
            #     #     self.planet_args["k_user_data"].n, self.planet_args["k_user_data"].s
            #     # )
            #     k = self.planet_args["k_user_data"]
            #     D = k ** 2
            #     self.planet_args["D"] = D.n
            # else:
            #     k = self.planet_args["k"]
            #     D = k ** 2
            #     self.planet_args["D"] = D

            k = self.planet_args["k"]
            D = k ** 2
            self.planet_args["D"] = D
            self.planet_args["D_fit"] = self.planet_args["k_fit"]
            if self.planet_args["k_user_data"] != None:
                kpriors = self.planet_args["k_user_data"]
                Dpriors = kpriors ** 2
            if self.planet_args["k_bounds"] != None:
                self.planet_args["D_bounds"] = [kb**2 for kb in self.planet_args["k_bounds"]]

        elif "Rp" in planet_yaml:

            # if self.planet_args["Rp_user_data"] != None:
            #     # Rp = (
            #     #     ufloat(
            #     #         self.planet_args["Rp_user_data"].n,
            #     #         self.planet_args["Rp_user_data"].s,
            #     #     )
            #     #     * cst.Rears
            #     # )
            #     Rp = self.planet_args["Rp_user_data"] * cst.Rears
            #     k = Rp / self.star_args["Rstar"]
            #     D = k ** 2
            #     self.planet_args["D"] = D.n
            # else:
            #     Rp = self.planet_args["Rp"] * cst.Rears
            #     k = Rp / self.star_args["Rstar"]
            #     D = k ** 2
            #     self.planet_args["D"] = D

            Rp = self.planet_args["Rp"]
            k = Rp / self.star_args["Rstar"]
            D = k ** 2
            self.planet_args["D"] = D
            self.planet_args["D_fit"] = self.planet_args["Rp_fit"]
            if self.planet_args["R_user_data"] != None:
                Rppriors = self.planet_args["R_user_data"]
                kpriors = Rppriors/self.star_args["Rstar"]
                Dpriors = kpriors ** 2
            if self.planet_args["Rp_bounds"] != None:
                self.planet_args["D_bounds"] = [(rr/self.star_args["Rstar"])**2 for rr in self.planet_args["Rp_bounds"]]

        else:
            self.read_file_status.append(
                "ERROR: missing needed planet keyword: D or k or Rp (Rearth)"
            )

        # self.planet_args["D"] = D.n
        # if self.planet_args["D_bounds"][0] < 0:
        #     self.planet_args["D_bounds"][0] = 0
        if self.planet_args["D_bounds"] == [-np.inf, np.inf]:
            self.planet_args["D_bounds"] = [
                0.01 * self.planet_args["D"],
                10.0 * self.planet_args["D"]
            ]
        # self.planet_args["D_fit"] = True
        # self.planet_args["D_user_data"] = D

        self.planet_args["k"] = k

        if "inc" in planet_yaml and "aRs" in planet_yaml and "b" in planet_yaml:
            # inc = ufloat(
            #     self.planet_args["inc_user_data"].n, self.planet_args["inc_user_data"].s
            # )
            # aRs = ufloat(
            #     self.planet_args["aRs_user_data"].n, self.planet_args["aRs_user_data"].s
            # )
            # b = ufloat(
            #     self.planet_args["b_user_data"].n, self.planet_args["b_user_data"].s
            # )
            inc = self.get_planet_value_or_user_data("inc")
            aRs = self.get_planet_value_or_user_data("aRs")
            b = self.get_planet_value_or_user_data("b")
        elif "inc" in planet_yaml and "aRs" in planet_yaml:
            # inc = ufloat(
            #     self.planet_args["inc_user_data"].n, self.planet_args["inc_user_data"].s
            # )
            # aRs = ufloat(
            #     self.planet_args["aRs_user_data"].n, self.planet_args["aRs_user_data"].s
            # )
            inc = self.get_planet_value_or_user_data("inc")
            aRs = self.get_planet_value_or_user_data("aRs")
            b = aRs * um.cos(inc * cst.deg2rad)
        elif "b" in planet_yaml and "aRs" in planet_yaml:
            # aRs = ufloat(
            #     self.planet_args["aRs_user_data"].n, self.planet_args["aRs_user_data"].s
            # )
            # b = ufloat(
            #     self.planet_args["b_user_data"].n, self.planet_args["b_user_data"].s
            # )
            aRs = self.get_planet_value_or_user_data("aRs")
            b = self.get_planet_value_or_user_data("b")
            inc = um.acos(b / aRs) * cst.rad2deg
        elif "b" in planet_yaml and "inc" in planet_yaml:
            # b = ufloat(
            #     self.planet_args["b_user_data"].n, self.planet_args["b_user_data"].s
            # )
            # inc = ufloat(
            #     self.planet_args["inc_user_data"].n, self.planet_args["inc_user_data"].s
            # )
            inc = self.get_planet_value_or_user_data("inc")
            b = self.get_planet_value_or_user_data("b")
            aRs = b / um.cos(inc * cst.deg2rad)
        else:
            self.read_file_status.append(
                "ERROR: missing needed one of these pairs/combinations: (inc, aRs) or (b, aRs) or (b, inc) or (inc, aRs, b)"
            )
            inc, aRs, b = 90.0, 1.0, 0.0
            # sys.exit()
        try:
            self.planet_args["inc"] = inc.n
        except AttributeError:
            self.planet_args["inc"] = inc
        try:
            self.planet_args["aRs"] = aRs.n
            self.planet_args["aRs_user_data"] = ufloat(aRs.n, aRs.s)
        except AttributeError:
            self.planet_args["aRs"] = aRs
            self.planet_args["aRs_user_data"] = ufloat(aRs, 0.1 * aRs)
        if self.planet_args["aRs_bounds"] == [-np.inf, np.inf]:
            self.planet_args["aRs_bounds"] = [1.0, 1e6]
        try:
            self.planet_args["b"] = b.n
        except AttributeError:
            self.planet_args["b"] = b

        # self.planet_args["aRs_bounds"] = [
        #     0.1 * self.planet_args["aRs"],
        #     10 * self.planet_args["aRs"],
        # ]

        # self.planet_args["b_fit"] = True
        if self.planet_args["b_bounds"] == [-np.inf, np.inf]:
            self.planet_args["b_bounds"] = [
                0.0,
                1.5,
            ]
        # self.planet_args["b_user_data"] = b

        self.planet_args["l_3"] = 0

        if "T14" in planet_yaml:
            W = (
                ufloat(planet_yaml["T14"][0], planet_yaml["T14"][1])
                / self.planet_args["P"]
            )
        else:
            W = um.sqrt((1 + k) ** 2 - b ** 2) / np.pi / aRs
        if isinstance(W, UFloat):
            self.planet_args["W"] = W.n
            self.planet_args["W_fit"] = True
            self.planet_args["W_bounds"] = [
                0.1 * W.n,
                10.0 * W.n,
            ]
            self.planet_args["W_user_data"] = W
        else:
            self.planet_args["W"] = W
            self.planet_args["W_fit"] = True
            self.planet_args["W_bounds"] = [
                0.1 * W,
                10.0 * W,
            ]
            self.planet_args["W_user_data"] = None

        ecc = ufloat(0.0, 0.0)
        if "ecc" in planet_yaml:
            # print('planet_yaml["ecc"]', planet_yaml['ecc'])
            if str(planet_yaml["ecc"]).lower() != "none":
                try:
                    ecc = ufloat(planet_yaml["ecc"][0], planet_yaml["ecc"][1])
                except:
                    self.read_file_status.append("wrong ecc format: setting to 0+/-0")
                    ecc = ufloat(0.0, 0.0)
        se = um.sqrt(ecc)
        w = ufloat(90.0, 0.0)
        if "w" in planet_yaml:
            # print('planet_yaml["w"]', planet_yaml['w'])
            if str(planet_yaml["w"]).lower() != "none":
                try:
                    w = ufloat(planet_yaml["w"][0], planet_yaml["w"][1])
                except:
                    self.read_file_status.append(
                        "wrong w format: setting to 90+/-0 deg"
                    )
                    w = ufloat(90.0, 0.0)
        w_r = w * cst.deg2rad
        f_c = se * um.cos(w_r)
        f_s = se * um.sin(w_r)
        self.planet_args["ecc"] = ecc.n
        self.planet_args["w"] = w
        self.planet_args["f_c"] = f_c.n
        self.planet_args["f_c_fit"] = False
        self.planet_args["f_c_bounds"] = [-np.inf, np.inf]
        self.planet_args["f_c_user_data"] = None

        self.planet_args["f_s"] = f_s.n
        self.planet_args["f_s_fit"] = False
        self.planet_args["f_s_bounds"] = [-np.inf, np.inf]
        self.planet_args["f_s_user_data"] = None

        self.planet_args["T_0_user_data"] = None

        # self.planet_args["P_bounds"] = [0, 500]

        # if type(self.planet_args["P"]) is not float:
        #     self.planet_args["P"] = self.planet_args["P"].n

        if self.visit_args["shape"] == "fix":
            self.planet_args["D_fit"] = False
            self.planet_args["W_fit"] = False
            self.planet_args["b_fit"] = False

        if self.planet_args["aRs_bounds"][0] < 1:
            printlog("Found aRs lower bound < 1. Setting to 1")
            self.planet_args["aRs_bounds"][0] = 1

        if self.planet_args["P_bounds"][0] <= 0:
            printlog("Found P lower bound <= 0. Setting to 1e-4 days")
            self.planet_args["P_bounds"][0] = 1e-4

        if self.planet_args["b_bounds"][0] < 0:
            printlog("Found b lower bound < 0. Setting to 0")
            self.planet_args["b_bounds"][0] = 0

    def category_args(self, par):
        if par in self.star_args.keys():
            return self.star_args
        elif par in self.planet_args.keys():
            return self.planet_args
        else:
            self.read_file_status.append(
                f"ERROR: {par} is not defined in neither the star or planet arguments"
            )

    def err_msg(self, keyword, dictionary):
        if dictionary.get(keyword) is None and keyword not in [
            "glint_type",
            "clipping",
            "h_1",
            "h_2",  # case: h_1/2 not as input, but guessed from StarProperties
            "D",  # case: using k=Rp/Rs instead of D, is not here the code returns error
        ]:
            return f"keyword {keyword} not in input file. Set to None."
        else:
            return ""
