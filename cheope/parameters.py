import os
import yaml
import numpy as np

from uncertainties import ufloat
from uncertainties import umath as um

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

        self.planet_keys = ["P", "D", "inc", "aRs", "ecc", "w", "T0", "Kms"]

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
            self.star_args[key] = self.yaml_input["star"].get(key)
            # TODO Restructure to get it flexible when the dictionary is not defined within the yml file
            if isinstance(self.star_args[key], list):
                value = self.star_args[key]
                self.star_args[key]["fit"] = False
                self.star_args[key]["value"] = value

            if not isinstance(self.star_args[key], str):
                self.star_args[key + "_fit"] = self.star_args[key].get(
                    "fit", value=False
                )
                self.star_args[key] = self.star_args[key].get("value")
                if isinstance(self.star_args[key], list):
                    self.star_args[key] = ufloat(
                        self.star_args[key][0], self.star_args[key][1]
                    )
                self.read_file_status.append(self.err_msg(key, self.yaml_input["star"]))

    def planet_pars(self):

        for key in self.planet_keys:
            self.planet_args[key] = self.yaml_input["planet"].get(key)
            self.planet_args[key + "_fit"] = self.planet_args[key].get(
                "fit", value=False
            )
            self.planet_args[key] = self.planet_args[key].get("value")
            if isinstance(self.planet_args[key], list):
                self.planet_args[key] = ufloat(
                    self.planet_args[key][0], self.planet_args[key][1]
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

        for key in self.emcee_args:
            self.emcee_args[key] = self.yaml_input["emcee"].get(
                key, value=self.emcee_args[key]
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

        self.planet_args["D"] = D
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
        self.planet_args["inc"] = inc
        self.planet_args["aRs"] = aRs
        self.planet_args["b"] = b

        if "T14" in planet_yaml:
            W = (
                ufloat(planet_yaml["T14"][0], planet_yaml["T14"][1])
                / self.planet_args["P"]
            )
        else:
            W = um.sqrt((1 + k) ** 2 - b ** 2) / np.pi / aRs
        self.planet_args["W"] = W

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
        self.planet_args["ecc"] = ecc
        self.planet_args["w"] = w
        self.planet_args["f_c"] = f_c
        self.planet_args["f_s"] = f_s

    def err_msg(self, keyword, dictionary):
        if dictionary.get(keyword) == None:
            return f"ERROR: needed keyword {keyword} not in input file. Set to None."
        else:
            return ""
