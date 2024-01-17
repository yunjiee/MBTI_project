from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from bs4 import BeautifulSoup
import pandas as pd
import os

def crawl_single_page(base_url, existing_csv_path):
    driver = webdriver.Chrome()
    driver.implicitly_wait(60)

    processed_authors = set()  # 用於記錄已處理過的作者

    try:
        driver.get(base_url)
        WebDriverWait(driver, 60).until(
            EC.presence_of_element_located((By.CLASS_NAME, 'structItem-title'))
        )

        soup = BeautifulSoup(driver.page_source, 'html.parser')
        thread_links = soup.find_all('a', class_='thread-title--gtm')

        all_posts_data = []  # 存放所有貼文的資料

        for i, thread_link in enumerate(thread_links):
            thread_url_relative = thread_link['href']
            thread_url = f"https://www.personalitycafe.com/{thread_url_relative}"

            try:
                driver.get(thread_url)
                WebDriverWait(driver, 120).until(
                    EC.presence_of_element_located((By.CLASS_NAME, 'message'))
                )
                thread_posts_soup = BeautifulSoup(driver.page_source, 'html.parser')
                post_content_element = thread_posts_soup.find('div', class_='bbWrapper')
                post_content = post_content_element.text.strip() if post_content_element else ''

                # 獲取作者名稱
                author_element = thread_posts_soup.find('a', class_='MessageCard__user-info__name')
                author_name = author_element.text if author_element else ''

                # 獲取貼文標題
                thread_title_element = thread_posts_soup.find('h1', class_='MessageCard__thread-title')
                thread_title = thread_title_element.text.strip() if thread_title_element else ''

                print(f"Processing post from {author_name} ({thread_url})")

                # 檢查是否已經處理過該作者的貼文
                if author_name not in processed_authors:
                    processed_authors.add(author_name)

                    # 處理該作者的貼文內容
                    all_posts_data.append({
                        'Author': author_name,
                        'Thread_Title': thread_title,
                        'Content': post_content,
                        'Thread_URL': thread_url
                    })

            except TimeoutException:
                print(f"TimeoutException: Timed out on thread {i + 1}. Skipping...")
                pass

        # 將新資料轉換成 DataFrame
        df_new_posts = pd.DataFrame(all_posts_data)

        # 檢查現有 CSV 檔案是否存在，若不存在，就先建立一個新的git pull origin main
        if not os.path.exists(existing_csv_path):
            df_new_posts.to_csv(existing_csv_path, index=False)
            print(f"Created new file: {existing_csv_path}")
        else:
            # 讀取現有 CSV 檔案
            df_existing_posts = pd.read_csv(existing_csv_path)

            # 將新資料追加到現有 CSV 檔案
            df_combined = pd.concat([df_existing_posts, df_new_posts], ignore_index=True)

            # 將合併後的資料保存到 CSV 檔案
            df_combined.to_csv(existing_csv_path, index=False)
            print(f"Appended data to {existing_csv_path}")

    except Exception as e:
        print(f"Error: {e}")

    finally:
        # 關閉瀏覽器頁面
        driver.quit()

    return None

# 設定論壇基本網址
base_forum_url = "https://www.personalitycafe.com/forums/infj-forum-the-protectors.18/page-16?sorting=latest-activity"

# 現有的 CSV 檔案路徑
existing_csv_path = "data_personality/infj_data.csv"

# 爬取資料
crawl_single_page(base_forum_url, existing_csv_path)

#爬到16頁https://www.personalitycafe.com/threads/subtypes-of-infjs.1350854/