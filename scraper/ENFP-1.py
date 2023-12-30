from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import pandas as pd

def crawl_enfp_forum(url, num_pages=2):
    driver = webdriver.Chrome()  # 請根據你使用的瀏覽器選擇對應的WebDriver
    driver.get(url)

    # 等待頁面載入完成
    WebDriverWait(driver, 30).until(
        EC.presence_of_element_located((By.CLASS_NAME, 'structItem-title'))
    )

    posts_data = []

    for page_num in range(1, num_pages + 1):
        # 使用 BeautifulSoup 解析頁面
        soup = BeautifulSoup(driver.page_source, 'html.parser')

        # 找到所有主題連結
        thread_links = soup.find_all('a', class_='thread-title--gtm')

        for i, thread_link in enumerate(thread_links):
            thread_url_relative = thread_link['href']
            thread_url = f"https://www.personalitycafe.com/{thread_url_relative}"

            # 進入主題連結，進入帖子頁面
            driver.get(thread_url)
            WebDriverWait(driver, 30).until(
                EC.presence_of_element_located((By.CLASS_NAME, 'message'))
            )
            thread_posts_soup = BeautifulSoup(driver.page_source, 'html.parser')

            # 找到第一個帖子的發文者和內容
            post_content_element = thread_posts_soup.find('div', class_='bbWrapper')

            post_content = post_content_element.text.strip() if post_content_element else ''

            posts_data.append({
                'Content': post_content
            })

        #轉到下一頁
        try:
            next_page_link = driver.find_element_by_link_text('Next >')
            next_page_link.click()
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, 'structItem-title'))
            )
        except:
            print("No more pages.")
            break

    # 關閉 WebDriver
    driver.quit()

    return posts_data

# 爬取 ENFP 類型的帖子資料
enfp_forum_url = "https://www.personalitycafe.com/forums/enfp-forum-the-inspirers.19/"
enfp_posts_data = crawl_enfp_forum(enfp_forum_url, num_pages=2)

# 將爬取到的資料轉換成 DataFrame
df = pd.DataFrame(enfp_posts_data)

# 將 DataFrame 寫入 CSV 檔案
csv_file_path = "data_personality/enfp_posts_data.csv"
df.to_csv(csv_file_path, index=False)
