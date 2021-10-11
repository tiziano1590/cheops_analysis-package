from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from cheope.parameters import ReadFile
import time
from cryptography.fernet import Fernet
import numpy as np
import os
import glob
import tarfile
import shutil


class DACESearch:
    def __init__(self, input_file):
        """
        Setup all the input parameters
        """
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
        profile.set_preference(
            "browser.helperApps.neverAsk.saveToDisk",
            "text/plain, application/octet-stream, application/binary, text/csv, application/csv, application/excel, text/comma-separated-values, text/xml, application/xml",
        )
        self.driver = webdriver.Firefox(
            executable_path=self.visit_args["firefox_driver_path"],
            firefox_profile=profile,
            options=options,
        )

    def get_observations(self):

        if not os.path.exists(self.visit_args["main_folder"] + "/private"):
            message = self.visit_args["password"]
            key = Fernet.generate_key()

            with open(self.visit_args["main_folder"] + "/private", "wb") as private:
                private.write(key)
            fernet = Fernet(key)

            # then use the Fernet class instance
            # to encrypt the string string must must
            # be encoded to byte string before encryption
            encMessage = fernet.encrypt(message.encode())
            with open(self.visit_args["main_folder"] + "/public", "wb") as public:
                public.write(encMessage)

            print("original string: ", message)
            print("encrypted string: ", encMessage)
        else:
            with open(self.visit_args["main_folder"] + "/private", "rb") as private:
                key = private.read()

            fernet = Fernet(key)
            # Instance the Fernet class with the key

            with open(self.visit_args["main_folder"] + "/public", "rb") as public:
                encMessage = public.read()

            # decrypt the encrypted string with the
            # Fernet instance of the key,
            # that was used for encrypting the string
            # encoded byte string is returned by decrypt method,
            # so decode it to string with decode methods
            decMessage = fernet.decrypt(encMessage).decode()

        self.driver.get("https://dace.unige.ch/dashboard/index.html")

        time.sleep(10)

        signin = self.driver.find_element_by_link_text("Sign in / Create account")
        signin.click()

        time.sleep(2)

        userid = self.driver.find_element_by_id("loginUserField")
        userid.send_keys("tiziano.zingales@unipd.it")

        userpass = self.driver.find_element_by_id("loginPassField")
        userpass.send_keys(decMessage)

        login = self.driver.find_element_by_css_selector("button.popup_login_row")
        login.click()

        time.sleep(2)

        cheops_polygon = self.driver.find_element_by_id("cheops_polygon")
        cheops_polygon.click()

        time.sleep(3)

        observations_data = self.driver.find_element_by_css_selector(
            ".col-lg-9 > div:nth-child(1) > div:nth-child(2) > div:nth-child(1) > table:nth-child(3) > tbody:nth-child(1) > tr:nth-child(1) > td:nth-child(1) > div:nth-child(1) > div:nth-child(1)"
        )
        observations_data.click()

        time.sleep(5)

        object_search = self.driver.find_element_by_css_selector("span.fa-search")
        object_search.click()

        time.sleep(2)

        object_equals = self.driver.find_element_by_xpath(
            "/html/body/div[3]/div[2]/div[2]/div[1]/div/table/thead/tr[2]/th[3]/span/div/ul/li/form/div/input[1]"
        )
        object_equals.send_keys(self.visit_args["object_name"] + Keys.ENTER)
        object_equals.click()

        time.sleep(2)

        actions = self.driver.find_element_by_xpath(
            "/html/body/div[3]/div[2]/div[2]/div[1]/div/table/thead/tr[1]/th/div"
        )
        actions.click()

        time.sleep(1)

        select_all = self.driver.find_element_by_xpath(
            "/html/body/div[3]/div[2]/div[2]/div[1]/div/table/thead/tr[1]/th/div/ul/li[3]/ul/li[1]"
        )
        select_all.click()

        time.sleep(1)

        # actions.click()

        # time.sleep(1)

        download_all = self.driver.find_element_by_link_text("Fullarray")
        download_all.click()

        time.sleep(10)

        list_of_files = glob.glob(
            os.path.join(self.visit_args["download_path"], "*")
        )  # * means all if need specific format then *.csv
        latest_file = max(list_of_files, key=os.path.getctime)
        print(latest_file)

        path = os.path.join(self.visit_args["download_path"], "newly_extracted")

        if os.path.exists(path):
            shutil.rmtree(path)

        if latest_file.endswith("tar.gz"):
            tar = tarfile.open(latest_file, "r:gz")
            tar.extractall(path=path)
            tar.close()
        elif latest_file.endswith("tar"):
            tar = tarfile.open(latest_file, "r:")
            tar.extractall(path=path)
            tar.close()

        all_downloaded = glob.glob(
            os.path.join(self.visit_args["download_path"], "newly_extracted/*")
        )
        print(all_downloaded)

        for folder in all_downloaded:
            keyword = folder.split("/")[-1]

            tar = tarfile.open(
                f"{os.path.join(self.visit_args['pycheops_path'], '')}/CH_{keyword}.tgz",
                "w:gz",
            )
            tar.add(folder, arcname=f"{keyword}")
            tar.close()

        if os.path.exists(path):
            shutil.rmtree(path)
