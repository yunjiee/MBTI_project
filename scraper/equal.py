# import pandas as pd
# from bs4 import BeautifulSoup
# from selenium import webdriver
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
# from selenium.common.exceptions import TimeoutException

# def crawl_enfp_forum(url, author_limit, total_pages):
#     driver = webdriver.Chrome()
#     driver.implicitly_wait(60)
#     driver.get(url)

#     WebDriverWait(driver, 60).until(
#         EC.presence_of_element_located((By.CLASS_NAME, 'structItem-title'))
#     )

#     posts_data = []
#     authors_set = set()

#     try:
#         for _ in range(total_pages):
#             soup = BeautifulSoup(driver.page_source, 'html.parser')
#             thread_links = soup.find_all('a', class_='thread-title--gtm')

#             for i, thread_link in enumerate(thread_links):
#                 thread_url_relative = thread_link['href']
#                 thread_url = f"https://www.personalitycafe.com/{thread_url_relative}"

#                 try:
#                     driver.get(thread_url)
#                     WebDriverWait(driver, 30).until(
#                         EC.presence_of_element_located((By.CLASS_NAME, 'message'))
#                     )
#                     thread_posts_soup = BeautifulSoup(driver.page_source, 'html.parser')
#                     post_content_element = thread_posts_soup.find('div', class_='bbWrapper')
#                     post_content = post_content_element.text.strip() if post_content_element else ''

#                     # 獲取作者名稱
#                     author_element = thread_posts_soup.find('a', class_='MessageCard__user-info__name')
#                     author_name = author_element.text if author_element else ''

#                     print(f"Processing post from {author_name} ({thread_url})")

#                     # 檢查是否已經爬取過該作者的貼文
#                     if author_name not in authors_set:
#                         authors_set.add(author_name)

#                         # 處理該作者的貼文內容
#                         posts_data.append({
#                             'Author': author_name,
#                             'Content': post_content,
#                             'Thread_URL': thread_url
#                         })

#                         # 如果已經達到指定作者數目，則退出迴圈
#                         if len(authors_set) >= author_limit:
#                             break
#                 except TimeoutException:
#                     print(f"TimeoutException: Timed out on thread {i + 1}. Skipping...")
#                     pass
#             # 點擊下一頁按鈕
#             try:
#                 next_page_button = WebDriverWait(driver, 10).until(
#                     EC.presence_of_element_located((By.CSS_SELECTOR, 'a.pageNav-jump--next'))
#                 )
#                 next_page_button.click()
#             except TimeoutException:
#                 print("TimeoutException: Timed out waiting for page to load. Exiting...")
#                 break

#             # 如果已經達到指定作者數目，則退出迴圈
#             if len(authors_set) >= author_limit:
#                 break

#     except Exception as e:
#         print(f"Error: {e}")

#     finally:
#         driver.quit()

#     return posts_data

# # 讀取存儲的網址
# with open("enfp_urls.txt", "r") as file:
#     all_urls = file.read().splitlines()

# # 設定要爬取的作者數目和總頁數
# author_limit = 15
# total_pages = 2

# # 爬取指定頁數的資料
# all_posts_data = crawl_enfp_forum(all_urls[0], author_limit, total_pages)

# # 將所有的資料轉換成 DataFrame
# df_all_posts = pd.DataFrame(all_posts_data)

# # 設定 CSV 檔案路徑
# csv_file_path = "enfp_posts_data.csv"

# # 將資料寫入 CSV 檔案
# df_all_posts.to_csv(csv_file_path, index=False)

import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

def crawl_enfp_forum(url, author_limit, total_pages):
    driver = webdriver.Chrome()
    driver.implicitly_wait(60)
    driver.get(url)

    WebDriverWait(driver, 60).until(
        EC.presence_of_element_located((By.CLASS_NAME, 'structItem-title'))
    )

    posts_data = []
    authors_set = set()

    try:
        for _ in range(total_pages):
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

                    # 獲取作者名稱
                    author_element = thread_posts_soup.find('a', class_='MessageCard__user-info__name')
                    author_name = author_element.text if author_element else ''

                    print(f"Processing post from {author_name} ({thread_url})")

                    # 檢查是否已經爬取過該作者的貼文
                    if author_name not in authors_set:
                        authors_set.add(author_name)

                        # 處理該作者的貼文內容
                        posts_data.append({
                            'Author': author_name,
                            'Content': post_content,
                            'Thread_URL': thread_url
                        })

                    # 如果已經達到指定作者數目，則退出迴圈
                    if len(authors_set) >= author_limit:
                        break
                except TimeoutException:
                    print(f"TimeoutException: Timed out on thread {i + 1}. Skipping...")
                    pass

            # 如果已經達到指定作者數目，則退出迴圈
            if len(authors_set) >= author_limit:
                break
            # 模擬滾動以加載更多帖子，這部分可視情況選擇是否需要
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

    except Exception as e:
        print(f"Error: {e}")

    finally:
        driver.quit()

    return posts_data

# 讀取存儲的網址
with open("enfp_urls.txt", "r") as file:
    all_urls = file.read().splitlines()

# 設定要爬取的作者數目和總頁數
author_limit = 80
total_pages = 2

# 爬取指定頁數的資料
all_posts_data = []
for url in all_urls[:total_pages]:
    current_posts_data = crawl_enfp_forum(url, author_limit, total_pages)
    all_posts_data.extend(current_posts_data)

# 將所有的資料轉換成 DataFrame
df_all_posts = pd.DataFrame(all_posts_data)

# 檢查爬取到的作者數量是否足夠
if len(df_all_posts['Author'].unique()) < author_limit:
    print(f"Warning: Only crawled {len(df_all_posts['Author'].unique())} unique authors, which is less than the required {author_limit}. The data may not be sufficient.")

# 設定 CSV 檔案路徑
csv_file_path = "enfp_posts_data1.csv"

# 將資料寫入 CSV 檔
df_all_posts.to_csv(csv_file_path, index=False)
