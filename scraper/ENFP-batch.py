import pandas as pd
import os
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

def crawl_enfp_forum_batch(url, start_page, end_page):
    driver = webdriver.Chrome()
    driver.implicitly_wait(60)
    driver.get(url)

    WebDriverWait(driver, 60).until(
        EC.presence_of_element_located((By.CLASS_NAME, 'structItem-title'))
    )

    posts_data = []

    for page_num in range(start_page, end_page + 1):
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
                print(f"TimeoutException: Timed out on thread {i + 1} of batch {start_page}-{end_page}. Skipping...")
                pass
            finally:
                pass

        try:
            # 滾動到頁面底部觸發動態加載
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

            # 等待 "Next >" 鈕可見性
            next_page_link = WebDriverWait(driver, 30).until(
                EC.presence_of_element_located((By.LINK_TEXT, 'Next >'))
            )

            next_page_link.click()
            WebDriverWait(driver, 60).until(
                EC.presence_of_element_located((By.CLASS_NAME, 'structItem-title'))
            )
        except TimeoutException:
            print(f"No more pages in batch {start_page}-{end_page}.")
            break

    driver.quit()

    return posts_data

# 定義每批次的頁數
batch_size = 5
total_pages = 10

enfp_forum_url = "https://www.personalitycafe.com/forums/enfp-forum-the-inspirers.19/"

# 分批次爬取
for batch_num in range(total_pages // batch_size):
    start_page = batch_num * batch_size + 1
    end_page = (batch_num + 1) * batch_size
    enfp_posts_data_batch = crawl_enfp_forum_batch(enfp_forum_url, start_page, end_page)
    
    # 將每批次的資料轉換成 DataFrame
    df_batch = pd.DataFrame(enfp_posts_data_batch)

    # 設定 CSV 檔案路徑
    csv_file_path = "data_personality/enfp_posts_data_batch.csv"

    # 檢查檔案是否存在
    if os.path.isfile(csv_file_path):
        # 如果檔案存在，則將資料附加到現有檔案
        df_batch.to_csv(csv_file_path, mode='a', header=False, index=False)
    else:
        # 如果檔案不存在，則新建檔案並寫入資料
        df_batch.to_csv(csv_file_path, index=False)
