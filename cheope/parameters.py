import os
import yaml
import numpy as np

from uncertainties import ufloat
from uncertainties import umath as um

from pycheops import StarProperties

import cheope.pyconstants as cst


class ReadFile:
    def __init__(self, input_file):
        self.visit_args = {}
        self.star_args = {}
        self.planet_args = {}
        self.emcee_args = {}
        self.read_file_status = []
        self.yaml_input = {}

        self.visit_keys = [
            "main_folder",
            "visit_number",
            "file_key",
            "aperture",
            "shape",
            "seed",
            "glint_type",
            "clipping",
        ]

        self.star_keys = [
            "star_name",
            "Rstar",
            "Mstar",
            "teff",
            "logg",
            "feh",
            "h_1",
            "h_2",
        ]

        self.planet_keys = ["P", "D", "inc", "aRs", "ecc", "w", "T_0", "Kms"]

        self.emcee_keys = [
            "nwalkers",
            "nprerun",
            "nsteps",
            "nburn",
            "nthin",
            "progress",
            "nthreads",
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

        else:
            self.read_file_status.append(f"NOT VALID INPUT FILE:\n{input_file}")

        self.visit_pars()
        self.star_pars()
        self.planet_pars()
        self.emcee_pars()

    def visit_pars(self):
        # -- visit_args
        for key in self.visit_keys:
            self.visit_args[key] = self.yaml_input.get(key, "None")
            self.read_file_status.append(self.err_msg(key, self.yaml_input))

        self.apply_visit_conditions()

    def star_pars(self):

        for key in self.star_keys:
            inval = self.yaml_input["star"].get(key)  # get the input value

            if (inval is None) or isinstance(inval, str):
                self.star_args[key] = inval
            elif isinstance(inval, list):
                self.star_args[key] = ufloat(inval[0], inval[1])
                self.star_args[key + "_fit"] = False
                self.star_args[key + "_bounds"] = [-np.inf, np.inf]
                self.star_args[key + "_user_data"] = ufloat(inval[0], inval[1])
            elif isinstance(inval, dict):
                val = inval.get("value")
                fit = inval.get("fit")
                self.star_args[key] = ufloat(val[0], val[1]).n
                self.star_args[key + "_fit"] = fit
                self.star_args[key + "_bounds"] = inval.get("bounds", [-np.inf, np.inf])
                self.star_args[key + "_user_data"] = ufloat(val[0], val[1])
                # if key == "h_1":
                #     print("HAHA")
                #     break
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
            elif isinstance(inval, list):
                self.planet_args[key] = ufloat(inval[0], inval[1])
                self.planet_args[key + "_fit"] = False
                self.planet_args[key + "_bounds"] = [-np.inf, np.inf]
                self.planet_args[key + "_user_data"] = ufloat(inval[0], inval[1])
            elif isinstance(inval, dict):
                val = inval.get("value")
                fit = inval.get("fit")
                self.planet_args[key] = val[0]
                self.planet_args[key + "_fit"] = fit
                self.planet_args[key + "_bounds"] = inval.get(
                    "bounds", [-np.inf, np.inf]
                )
                self.planet_args[key + "_user_data"] = ufloat(val[0], val[1])

            else:
                self.read_file_status.append(
                    f"{key} should be either a list or a dictionary."
                )

            self.read_file_status.append(self.err_msg(key, self.yaml_input["planet"]))

        self.apply_planet_conditions()

    def emcee_pars(self):

        self.emcee_args["nwalkers"] = 128
        self.emcee_args["nprerun"] = 512
        self.emcee_args["nsteps"] = 1280
        self.emcee_args["nburn"] = 256
        self.emcee_args["nthin"] = 1
        self.emcee_args["nthreads"] = 1
        self.emcee_args["progress"] = False

        for key in self.emcee_keys:
            self.emcee_args[key] = self.yaml_input["emcee"].get(
                key, self.emcee_args[key]
            )

    def apply_visit_conditions(self):

        # main folder
        self.visit_args["main_folder"] = os.path.abspath(self.visit_args["main_folder"])

        # file_key
        self.visit_args["file_key"] = self.visit_args["file_key"].strip()

        # aperture
        apertures = ["DEFAULT", "OPTIMAL", "RINF", "RSUP"]
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
        self.visit_args["glint_type"] = self.visit_args["glint_type"].strip().lower()
        if not self.visit_args["glint_type"] in glints:
            self.visit_args["glint_type"] = False

        # clip_outliers
        self.visit_args["clip_outliers"] = 5

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

        star = StarProperties(
            self.star_args["star_name"],
            match_arcsec=None,
            teff=self.star_args["teff"],
            logg=self.star_args["logg"],
            metal=self.star_args["feh"],
            dace=False,
        )

        if (
            self.yaml_input["star"].get("h_1") is None
            or self.yaml_input["star"].get("h_2") is None
        ):
            self.star_args["h_1"] = star.h_1.n
            self.star_args["h_2"] = star.h_2.n
            self.star_args["h_1_fit"] = True
            self.star_args["h_2_fit"] = True
            self.star_args["h_1_bounds"] = [0, 1]
            self.star_args["h_2_bounds"] = [0, 1]
            self.star_args["h_1_user_data"] = ufloat(star.h_1.n, 10 * star.h_1.s)
            self.star_args["h_2_user_data"] = ufloat(star.h_2.n, 10 * star.h_2.s)

        if self.yaml_input["star"].get("logrho") is None:
            self.star_args["logrho"] = star.logrho.n
            self.star_args["logrho_fit"] = True
            self.star_args["logrho_bounds"] = [-9, 6]
            self.star_args["logrho_user_data"] = star.logrho

    def apply_planet_conditions(self):

        planet_yaml = self.yaml_input["planet"]
        if "D" in planet_yaml:
            D = ufloat(planet_yaml["D"][0], planet_yaml["D"][1])
            k = um.sqrt(D)
        elif "k" in planet_yaml:
            k = ufloat(planet_yaml["k"][0], planet_yaml["k"][1])
            D = k ** 2
        elif "Rp" in planet_yaml:
            Rp = ufloat(planet_yaml["Rp"][0], planet_yaml["Rp"][1]) * cst.Rears
            k = Rp / self.star_args["Rstar"]
            D = k ** 2
        else:
            self.read_file_status.append(
                "ERROR: missing needed planet keyword: D or k or Rp (Rearth)"
            )

        self.planet_args["D"] = D.n
        self.planet_args["D_bounds"] = [
            0.5 * D.n,
            1.5 * D.n,
        ]
        self.planet_args["D_fit"] = True
        self.planet_args["D_user_data"] = D

        self.planet_args["k"] = k

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
            self.read_file_status.append(
                "ERROR: missing needed one of these pairs/combinations: (inc, aRs) or (b, aRs) or (b, inc) or (inc, aRs, b)"
            )
            inc, aRs, b = 90.0, 1.0, 0.0
            # sys.exit()
        self.planet_args["inc"] = inc.n
        self.planet_args["aRs"] = aRs.n
        self.planet_args["b"] = b.n
        self.planet_args["b_fit"] = True
        self.planet_args["b_bounds"] = [
            0.0,
            1.5,
        ]
        self.planet_args["b_user_data"] = b

        if "T14" in planet_yaml:
            W = (
                ufloat(planet_yaml["T14"][0], planet_yaml["T14"][1])
                / self.planet_args["P"]
            )
        else:
            W = um.sqrt((1 + k) ** 2 - b ** 2) / np.pi / aRs
        self.planet_args["W"] = W.n
        self.planet_args["W_fit"] = True
        self.planet_args["W_bounds"] = [
            0.5 * W.n,
            1.5 * W.n,
        ]
        self.planet_args["W_user_data"] = W

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

        self.planet_args["P"] = self.planet_args["P"].n

        if self.visit_args["shape"] == "fix":
            self.planet_args["D_fit"] = False
            self.planet_args["W_fit"] = False
            self.planet_args["b_fit"] = False

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
