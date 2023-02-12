### 데이터 수집하기위한 크롤링 코드 ###
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from multiprocessing import Pool
import pandas as pd
import os
import time
import urllib.request

# 키워드 가져오기
keys = pd.read_csv("./keyword.txt", encoding="utf-8", names=['keyword'])
keyword = []
[keyword.append(keys['keyword'][x]) for x in range(len(keys))]

print(keyword)


def create_folder(dir):
    # 이미지저장할 폴더 구성
    try:
        if not os.path.exists(dir):
            os.makedirs(dir)
    except OSError:
        print("Error creating folder", + dir)


def image_download(keyword):
    # image download 함수
    create_folder("./" + keyword + "/")

    # chromdriver 가져오기
    chromdriver = "./chromedriver."
    driver = webdriver.Chrome(chromdriver)
    driver.implicitly_wait(3)

    print("keyword: " + keyword)
    driver.get('https://www.google.co.kr/imghp?hl=ko')
    keywords = driver.find_element_by_xpath(
        '/html/body/div[1]/div[3]/form/div[1]/div[1]/div[1]/div/div[2]/input')
    keywords.send_keys(keyword)
    driver.find_element_by_xpath(
        '/html/body/div[1]/div[3]/form/div[1]/div[1]/div[1]/button').click()


# =============================================================================
# 실행
# =============================================================================
if __name__ == '__main__':
    pool = Pool(processes=5)  # 5개의 프로세스를 사용합니다.
    pool.map(image_download, keyword)
