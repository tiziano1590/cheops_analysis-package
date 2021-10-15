from selenium import webdriver
from selenium.common.exceptions import (
    NoSuchElementException,
    ElementClickInterceptedException,
    StaleElementReferenceException,
)

from selenium.webdriver.common.keys import Keys
from cheope.parameters import ReadFile
import time
from cryptography.fernet import Fernet
import numpy as np
import os
import glob
import tarfile
import shutil
import functools

LIMIT_TIME = 60

## Define useful decorators
def _wait_browser(foo):
    def wait():
        i = 0
        while True and i < LIMIT_TIME:
            try:
                obj = foo()
                break
            except (
                NoSuchElementException,
                ElementClickInterceptedException,
                StaleElementReferenceException,
            ):
                time.sleep(2)
                i += 1
        return obj

    return wait


def _wait_browser_and_click(foo):
    def wait():
        i = 0
        while True and i < LIMIT_TIME:
            try:
                obj = foo()
                obj.click()
                break
            except (
                NoSuchElementException,
                ElementClickInterceptedException,
                StaleElementReferenceException,
            ):
                time.sleep(2)
                i += 1

    return wait


def _wait_browser_and_send_keys(foo):
    def wait(ref):
        i = 0
        while True and i < LIMIT_TIME:
            try:
                obj = foo(ref)
                obj.send_keys(ref)
                break
            except (
                NoSuchElementException,
                ElementClickInterceptedException,
                StaleElementReferenceException,
            ):
                time.sleep(2)
                i += 1

    return wait


class TESSSearch:
    def __init__(self, input_file):
        """
        Setup all the input parameters
        """
        self.input_file = input_file

        inpars = ReadFile(input_file)

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

        options = webdriver.FirefoxOptions()
        # options.headless = True
        profile = webdriver.FirefoxProfile()
        profile.set_preference("browser.download.folderList", 2)
        profile.set_preference("browser.download.manager.showWhenStarting", False)
        profile.set_preference(
            "browser.helperApps.neverAsk.saveToDisk",
            "text/plain;application/octet-stream;application/binary;text/csv;application/csv;application/excel;text/comma-separated-values;text/xml;application/xml;application/x-sh",
        )
        self.driver = webdriver.Firefox(
            executable_path=self.visit_args["firefox_driver_path"],
            firefox_profile=profile,
            options=options,
        )

    def get_observations(self):
        self.driver.get("https://heasarc.gsfc.nasa.gov/cgi-bin/tess/webtess/wtv.py")

        @_wait_browser_and_send_keys
        def search(ref):
            return self.driver.find_element_by_id("entry")

        search(str(self.visit_args["object_name"]) + Keys.ENTER)

        @_wait_browser
        def sector_selection():
            return self.driver.find_element_by_css_selector(
                ".entry-content > pre:nth-child(3)"
            )

        selection = sector_selection().text
        split = selection.split("\n")

        sectors = []
        for line in split:
            if "observed in camera" in line:
                sectors.append(int(line[7:9]))

        self.driver.get(
            "https://archive.stsci.edu/tess/bulk_downloads/bulk_downloads_ffi-tp-lc-dv.html"
        )

        time.sleep(5)

        cmd_dwn_tess_lcs = []
        for sector in sectors:
            try:
                # @_wait_browser_and_click
                def download_getfile():
                    return self.driver.find_element_by_link_text(
                        f"tesscurl_sector_{sector}_lc.sh"
                    )

                downfile = download_getfile()
                downfile.click()
                print(f"Downloading .sh sector {sector} file.")

                time.sleep(5)

                while os.path.exists(
                    os.path.join(self.visit_args["download_path"], "*.part")
                ):
                    time.sleep(1)

                with open(
                    os.path.join(self.visit_args["download_path"], "")
                    + f"tesscurl_sector_{sector}_lc.sh"
                ) as tesscurl:
                    all_lines = tesscurl.readlines()
                for line in all_lines:
                    if str(self.visit_args["object_name"]) in line:
                        cmd_dwn_tess_lcs.append(line.strip())

            except NoSuchElementException:
                pass

        print("All downloads finished!")

        os.chdir(os.path.join(self.visit_args["main_folder"], "") + "data/TESS_DATA")
        for cmd in cmd_dwn_tess_lcs:
            os.system(cmd)

        all_lcs = glob.glob("./*")
        keywords = [lc[2:-5] for lc in all_lcs]
        return keywords

    def substitute_file_key(self, keyword, visit_number):

        with open(self.input_file) as input_file:

            file_list = input_file.readlines()

        split_name = self.input_file.split("/")[:-1]
        new_file_name = (
            "/".join(split_name) + f"/V{visit_number}_CH_{keyword}_selenium.yml"
        )

        new_file = []
        for line in file_list:
            if "file_fits" in line:
                line = f"file_fits: {os.path.join(self.visit_args['main_folder'], '')}data/TESS_DATA/{keyword}.fits\n"

            new_file.append(line)

        with open(new_file_name, "w") as parfile:
            for line in new_file:
                parfile.write(line)

        return new_file_name
