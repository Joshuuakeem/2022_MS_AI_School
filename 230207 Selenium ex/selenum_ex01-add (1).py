from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import os
import urllib.request

"""
키워드 입력 , chromedriver 실행
"""
options = webdriver.ChromeOptions()
options.add_experimental_option("detach", True)

keywords = "사과"
chromedriver_path = "./chromedriver"

# # windows
# chromedriver_path = "./chromedriver.exe"

driver = webdriver.Chrome(chromedriver_path, options=options)
driver.implicitly_wait(3)

#####
# 키워드 입력 selenium 실행
#####
driver.get("https://www.google.co.kr/imghp?h1=ko")

# input -> /html/body/div[1]/div[3]/form/div[1]/div[1]/div[1]/div/div[2]/input
# button -> /html/body/div[1]/div[3]/form/div[1]/div[1]/div[1]/button
keyword = driver.find_element_by_xpath(
    '/html/body/div[1]/div[3]/form/div[1]/div[1]/div[1]/div/div[2]/input')
keyword.send_keys(keywords)
keyword.send_keys(Keys.RETURN)
# driver.find_element_by_xpath(
#     '/html/body/div[1]/div[3]/form/div[1]/div[1]/div[1]/button').click()

# elem = driver.find_element_by_name("q")
# elem.send_keys(keyword)
# elem.send_keys(Keys.RETURN)

# <input class = "gLFyf" jsaction = "paste:puy29d;"
# maxlength = "2048" name = "q" type = "text" aria-autocomplete = "both"
# aria-haspopup = "false" autocapitalize = "off" autocomplete = "off"
# autocorrect = "off" autofocus = "" role = "combobox"
# spellcheck = "false" title = "검색" value = ""
# aria-label = "검색" data-ved = "0ahUKEwjK5OHq__L7AhVeQPUHHZemCioQ39UDCAM" >

####### 스크롤 ##########
print(keywords + '스크롤 중 .......')
elem = driver.find_element_by_tag_name('body')
for i in range(60):
    elem.send_keys(Keys.PAGE_DOWN)
    time.sleep(0.2)

try:
    # //*[@id="islmp"]/div/div/div/div[2]/div[1]/div[2]/div[2]/input
    driver.find_element_by_xpath(
        '//*[@id="islmp"]/div/div/div/div[2]/div[1]/div[2]/div[2]/input').click()
    for i in range(60):
        elem.send_keys(Keys.PAGE_DOWN)
        time.sleep(0.2)
except:
    pass
