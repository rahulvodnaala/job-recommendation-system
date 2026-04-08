import pandas as pd
from tqdm import tqdm
from time import sleep
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from bs4 import BeautifulSoup
from selenium.common.exceptions import ElementClickInterceptedException
from selenium.common.exceptions import NoSuchElementException
import json
import urllib
import time

driver = webdriver.Chrome()


def openbrowser(locid, key):
    driver.wait = WebDriverWait(driver, 5)
    driver.maximize_window()
    words = key.split()
    txt = ''
    for w in words:
        txt += (w + '+')
    driver.get(
        "https://www.glassdoor.co.in/Job/jobs.htm?suggestCount=0&suggestChosen=true&clickSource=searchBtn&typedKeyword={}"
        "&sc.keyword={}&locT=C&locId={}&jobType=fulltime&fromAge=1&radius=6&cityId=-1&minRating=0.0&industryId=-1"
        "&sgocId=-1&companyId=-1&employerSizes=0&applicationType=0&remoteWorkType=0".format(txt[:-1], txt[:-1], locid))
    return driver


def geturl(driver):
    url = set()
    while True:
        print(len(url))
        if len(url) >= 20:
            break
        soup1 = BeautifulSoup(driver.page_source, "lxml")
        main = soup1.find_all("li", {"class": "jl"})
        for m in main:
            url.add('https://www.glassdoor.co.in{}'.format(m.find('a')['href']))
        try:
            next_element = soup1.find("li", {"class": "next"})
            try:
                next_exist = next_element.find('a')
            except AttributeError:
                driver.quit()
                break
            except NoSuchElementException:
                driver.quit()
                break
            if next_exist:
                driver.find_element(By.CLASS_NAME, "next").click()
                time.sleep(2)
            else:
                driver.quit()
                break
        except ElementClickInterceptedException:
            pass
    return list(url)


x = openbrowser(locid=4477468, key='"Data Scientist"')
with open('url_data_scientist_loc_bangalore.json', 'w') as f:
    json.dump(geturl(driver), f, indent=4)
    print("file created")

with open('url_data_scientist_loc_bangalore.json', 'r') as f:
    url = json.load(f)

data = {}
i = 1
jd_df = pd.DataFrame()
driver = webdriver.Chrome()

for u in tqdm(url):
    driver.wait = WebDriverWait(driver, 2)
    driver.maximize_window()
    driver.get(u)
    soup = BeautifulSoup(driver.page_source, "lxml")
    try:
        position = driver.find_element(By.TAG_NAME, 'h2').text
        company = driver.find_element(By.XPATH, "//span[@class='strong ib']").text
        location = driver.find_element(By.XPATH, "//span[@class='subtle ib']").text
        jd_temp = driver.find_element(By.ID, "JobDescriptionContainer")
        jd = jd_temp.text
    except IndexError:
        print('IndexError: list index out of range')
    except NoSuchElementException:
        pass
    data[i] = {
        'url': u,
        'Position': position,
        'Company': company,
        'Location': location,
        'Job_Description': jd
    }
    i += 1

driver.quit()
jd_df = pd.DataFrame(data)
jd = jd_df.transpose()
jd = jd[['url', 'Position', 'Company', 'Location', 'Job_Description']]
jd.to_csv(r'C:\job_recommendation\data\jd_unstructured_data.csv')
print('file created')
