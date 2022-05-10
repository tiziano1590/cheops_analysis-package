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
from zipfile import ZipFile

LIMIT_TIME = 60

## Define useful decorators
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


class DACESearch:
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
        options.headless = True
        profile = webdriver.FirefoxProfile()
        profile.set_preference(
            "browser.helperApps.neverAsk.saveToDisk",
            "text/plain, application/octet-stream, application/binary, text/csv, application/csv, application/excel, text/comma-separated-values, text/xml, application/xml",
        )
        self.driver = webdriver.Firefox(
            executable_path=self.visit_args["firefox_driver_path"],
            firefox_profile=profile,
            options=options,
        )

    def get_observations(self, download=False):

        if not os.path.exists(
            os.path.join(self.visit_args["pycheops_path"] + "/keys/private")
        ):
            os.makedirs(os.path.join(self.visit_args["pycheops_path"] + "/keys"))
            message = self.visit_args["password"]
            key = Fernet.generate_key()

            with open(
                os.path.join(self.visit_args["pycheops_path"] + "/keys/private"), "wb"
            ) as private:
                private.write(key)
            fernet = Fernet(key)

            # then use the Fernet class instance
            # to encrypt the string string must must
            # be encoded to byte string before encryption
            encMessage = fernet.encrypt(message.encode())
            with open(
                os.path.join(self.visit_args["pycheops_path"] + "/keys/public"), "wb"
            ) as public:
                public.write(encMessage)

            # print("original string: ", message)
            # print("encrypted string: ", encMessage)
        else:
            with open(
                os.path.join(self.visit_args["pycheops_path"] + "/keys/private"), "rb"
            ) as private:
                key = private.read()

            fernet = Fernet(key)
            # Instance the Fernet class with the key

            with open(
                os.path.join(self.visit_args["pycheops_path"] + "/keys/public"), "rb"
            ) as public:
                encMessage = public.read()

            # decrypt the encrypted string with the
            # Fernet instance of the key,
            # that was used for encrypting the string
            # encoded byte string is returned by decrypt method,
            # so decode it to string with decode methods
        if download:
            decMessage = fernet.decrypt(encMessage).decode()

            self.driver.get("https://dace.unige.ch/dashboard/index.html")

            @_wait_browser_and_click
            def signin():
                return self.driver.find_element_by_link_text("Sign in / Create account")

            signin()

            @_wait_browser_and_send_keys
            def userid(ref):
                return self.driver.find_element_by_id("loginUserField")

            userid(self.visit_args["login_dace"])

            @_wait_browser_and_send_keys
            def userpass(ref):
                return self.driver.find_element_by_id("loginPassField")

            userpass(decMessage)

            @_wait_browser_and_click
            def login():
                return self.driver.find_element_by_css_selector(
                    "button.popup_login_row"
                )

            login()

            @_wait_browser_and_click
            def cheops_polygon():
                return self.driver.find_element_by_id("smallCircle1")

            cheops_polygon()

            @_wait_browser_and_click
            def observations_data():
                return self.driver.find_element_by_css_selector(
                    ".col-lg-9 > div:nth-child(1) > div:nth-child(2) > div:nth-child(1) > table:nth-child(3) > tbody:nth-child(1) > tr:nth-child(1) > td:nth-child(1) > div:nth-child(1) > div:nth-child(1)"
                )

            observations_data()

            @_wait_browser_and_click
            def object_search():
                return self.driver.find_element_by_css_selector("span.fa-search")

            object_search()

            @_wait_browser_and_send_keys
            def object_equals(ref):
                return self.driver.find_element_by_xpath(
                    "/html/body/div[3]/div[2]/div[2]/div[1]/div/table/thead/tr[2]/th[3]/span/div/ul/li/form/div/input[1]"
                )

            object_equals(self.visit_args["object_name"] + Keys.ENTER)

            @_wait_browser_and_click
            def actions():
                return self.driver.find_element_by_xpath(
                    "/html/body/div[3]/div[2]/div[2]/div[1]/div/table/thead/tr[1]/th/div"
                )

            actions()

            @_wait_browser_and_click
            def select_all():
                return self.driver.find_element_by_xpath(
                    "/html/body/div[3]/div[2]/div[2]/div[1]/div/table/thead/tr[1]/th/div/ul/li[3]/ul/li[1]"
                )

            select_all()

            @_wait_browser_and_click
            def download_all():
                return self.driver.find_element_by_partial_link_text("Light curves")

            print("Downloading updated catalogue...")
            download_all()

            time.sleep(10)

            while os.path.exists(
                os.path.join(self.visit_args["download_path"], "*.part")
            ):
                print("Still downloading...")
                time.sleep(1)

            print("Download finished!")

        list_of_files = glob.glob(
            os.path.join(self.visit_args["download_path"], "*")
        )  # * means all if need specific format then *.csv
        latest_file = max(list_of_files, key=os.path.getctime)

        print(f"Last file is {latest_file}")

        path = os.path.join(self.visit_args["download_path"], "newly_extracted")

        if os.path.exists(path):
            shutil.rmtree(path)

        # os.makedirs(path)

        if latest_file.endswith("tar.gz"):
            tar = tarfile.open(latest_file, "r:gz")
            tar.extractall(path=path)
            tar.close()
        elif latest_file.endswith("tar"):
            tar = tarfile.open(latest_file, "r:")
            tar.extractall(path=path)
            tar.close()
        elif latest_file.endswith("zip"):
            zipfile = ZipFile(latest_file, "r")
            zipfile.extractall(path)
            zipfile.close()

        all_downloaded = glob.glob(
            os.path.join(self.visit_args["download_path"], "newly_extracted/*")
        )

        pycheops_path = os.path.join(self.visit_args["pycheops_path"], "")
        analysis_path = os.path.join(self.visit_args["main_folder"], "")

        keywords_list = []
        for folder in all_downloaded:
            keyword = folder.split("/")[-1]
            keywords_list.append(keyword)

        newly_downloaded = []

        for folder in all_downloaded:
            keyword = folder.split("/")[-1]
            if f"{pycheops_path}CH_{keyword}.tgz" not in glob.glob(
                os.path.join(self.visit_args["pycheops_path"], "*")
            ):
                newly_downloaded.append(folder)

        newly_downloaded = set(newly_downloaded)

        for folder in newly_downloaded:
            keyword = folder.split("/")[-1]

            tar = tarfile.open(
                f"{pycheops_path}/CH_{keyword}.tgz",
                "w:gz",
            )
            tar.add(folder, arcname=f"{keyword}")
            tar.close()

        if os.path.exists(path):
            shutil.rmtree(path)
        if os.path.exists(latest_file):
            os.remove(latest_file)

        return keywords_list

    def substitute_file_key(self, keyword, visit_number):

        with open(self.input_file) as input_file:

            file_list = input_file.readlines()

        split_name = self.input_file.split("/")[:-1]
        new_file_name = (
            "/".join(split_name)
            + f"/{self.visit_args['object_name']}_V{visit_number}_CH_{keyword}_selenium.yml"
        )

        print(new_file_name)

        new_file = []
        for line in file_list:
            if "file_key" in line:
                line = f"file_key: CH_{keyword}.tgz\n"
            if "visit_number" in line:
                line = f"visit_number: {visit_number}\n"

            new_file.append(line)

        with open(new_file_name, "w") as parfile:
            for line in new_file:
                parfile.write(line)

        return new_file_name
