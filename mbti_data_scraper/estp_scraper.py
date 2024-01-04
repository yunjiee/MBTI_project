import pandas as pd
import os
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

def crawl_estp_forum(url):
    driver = webdriver.Chrome()
    driver.implicitly_wait(60)
    driver.get(url)

    WebDriverWait(driver, 60).until(
        EC.presence_of_element_located((By.CLASS_NAME, 'structItem-title'))
    ) 

    posts_data = []

    try:
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        thread_links = soup.find_all('a', class_='thread-title--gtm')

        for i, thread_link in enumerate(thread_links):
            thread_url_relative = thread_link['href']
            thread_url = f"https://www.personalitycafe.com/{thread_url_relative}"

            try:
                driver.get(thread_url)
                WebDriverWait(driver, 30).until(
                    EC.presence_of_element_located((By.CLASS_NAME, 'message'))
                )
                thread_posts_soup = BeautifulSoup(driver.page_source, 'html.parser')
                post_content_element = thread_posts_soup.find('div', class_='bbWrapper')
                post_content = post_content_element.text.strip() if post_content_element else ''

                posts_data.append({
                    'Content': post_content
                })
            except TimeoutException:
                print(f"TimeoutException: Timed out on thread {i + 1}. Skipping...")
                pass

    except Exception as e:
        print(f"Error: {e}")

    finally:
        driver.quit()

    return posts_data

# 讀取存儲的網址
with open("mbti_urls/estp_urls.txt", "r") as file:
    urls = file.read().splitlines()

# 遍歷每個網址，爬取貼文
all_posts_data = []
for url in urls:
    print(f"Crawling posts from {url}")
    posts_data = crawl_estp_forum(url)
    all_posts_data.extend(posts_data)

# 將所有的資料轉換成 DataFrame
df_all_posts = pd.DataFrame(all_posts_data)

# 設定 CSV 檔案路徑
csv_file_path = "mbti_data/estp_posts_data.csv"

# 檢查檔案是否存在
if os.path.isfile(csv_file_path):
    # 如果檔案存在，則將資料附加到現有檔案
    df_all_posts.to_csv(csv_file_path, mode='a', header=False, index=False)
else:
    # 如果檔案不存在，則新建檔案並寫入資料
    df_all_posts.to_csv(csv_file_path, index=False)
