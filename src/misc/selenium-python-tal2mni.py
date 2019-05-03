import time
import os.path as op
from src.utils import utils

ROOT_FOL = utils.get_links_dir()
YALE_APP_URL = 'http://sprout022.sprout.yale.edu/mni2tal/mni2tal.html'


def tal2mni(points):
    try:
        from selenium import webdriver
    except:
        print('You should first install Selenium with Python!')
        print('https://selenium-python.readthedocs.io/installation.html')
        return False

    # Using Chrome to access web
    chrome_driver_fname = op.join(ROOT_FOL, 'chromedriver')
    if not op.isfile(chrome_driver_fname):
        print('Can\'t find chrome driver! It should be in {}'.format(chrome_driver_fname))
        print('ChromeDriver: https://sites.google.com/a/chromium.org/chromedriver/getting-started')
        return False

    driver = webdriver.Chrome(chrome_driver_fname)
    # Open the website
    driver.get(YALE_APP_URL)
    time.sleep(0.3)
    mni = []
    for xyz in points:
        for val, element_name in zip(xyz, ['talx', 'taly', 'talz']):
            set_val(driver, element_name, val)
        time.sleep(0.1)
        driver.find_element_by_id('talgo').click()
        time.sleep(0.1)
        mni.append([int(get_val(driver, element_name)) for element_name in ['mnix', 'mniy', 'mniz']])
    return mni


def get_val(driver, element_name):
    return driver.find_element_by_id(element_name).get_attribute('value')


def set_val(driver, element_name, val):
    driver.execute_script("document.getElementById('{}').value='{}'".format(element_name, str(val)))


if __name__ == '__main__':
    print(tal2mni([[2, 3, 9],[1, 4, 2], [6, 3, 7]]))
    # Should be [[3, 5, 7], [2, 7, -1], [7, 6, 5]]