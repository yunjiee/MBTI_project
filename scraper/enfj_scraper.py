# import pandas as pd
# from bs4 import BeautifulSoup

# from selenium import webdriver
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
# from selenium.common.exceptions import TimeoutException

# # 定義在外面，保持相同的狀態
# authors_set = set()
# all_posts_data = []

# def crawl_enfj_forum(base_url, author_limit, output_path="enfj_posts_data.csv"):
#     driver = webdriver.Chrome()
#     driver.implicitly_wait(60)

#     try:
#         # 初始頁面
#         url = base_url
#         driver.get(url)
#         WebDriverWait(driver, 60).until(
#             EC.presence_of_element_located((By.CLASS_NAME, 'structItem-title'))
#         )

#         soup = BeautifulSoup(driver.page_source, 'html.parser')

#         # 找到qid為"page-nav-other-page"的元素
#         page_nav_element = soup.select_one('[qid="page-nav-other-page"]')

#         # 從中獲取頁數
#         total_pages = int(page_nav_element.text)
#         print(total_pages)

#         for page in range(1, total_pages + 1):
#             # 動態生成網址
#             if page == 1:
#                 url = base_url
#             else:
#                 url = f"{base_url}/page-{page}?sorting=latest-activity"

#             driver.get(url)
#             WebDriverWait(driver, 60).until(
#                 EC.presence_of_element_located((By.CLASS_NAME, 'structItem-title'))
#             )

#             soup = BeautifulSoup(driver.page_source, 'html.parser')
#             thread_links = soup.find_all('a', class_='thread-title--gtm')

#             for i, thread_link in enumerate(thread_links):
#                 thread_url_relative = thread_link['href']
#                 thread_url = f"https://www.personalitycafe.com/{thread_url_relative}"

#                 try:
#                     driver.get(thread_url)
#                     WebDriverWait(driver, 120).until(
#                         EC.presence_of_element_located((By.CLASS_NAME, 'message'))
#                     )
#                     thread_posts_soup = BeautifulSoup(driver.page_source, 'html.parser')
#                     post_content_element = thread_posts_soup.find('div', class_='bbWrapper')
#                     post_content = post_content_element.text.strip() if post_content_element else ''

#                     # 獲取作者名稱
#                     author_element = thread_posts_soup.find('a', class_='MessageCard__user-info__name')
#                     author_name = author_element.text if author_element else ''

#                     # 獲取貼文標題
#                     thread_title_element = thread_posts_soup.find('h1', class_='MessageCard__thread-title')
#                     thread_title = thread_title_element.text.strip() if thread_title_element else ''

#                     print(f"Processing post from {author_name} ({thread_url})")

#                     # 檢查是否已經爬取過該作者的貼文
#                     if author_name not in authors_set:
#                         authors_set.add(author_name)

#                         # 處理該作者的貼文內容
#                         all_posts_data.append({
#                             'Author': author_name,
#                             'Thread_Title': thread_title,  # 加入貼文標題
#                             'Content': post_content,
#                             'Thread_URL': thread_url
#                         })

#                     # 如果已經達到指定作者數目，則退出迴圈
#                     if len(authors_set) >= author_limit:
#                         break
#                 except TimeoutException:
#                     print(f"TimeoutException: Timed out on thread {i + 1}. Skipping...")
#                     pass

#             # 如果已經達到指定作者數目，則退出迴圈
#             if len(authors_set) >= author_limit:
#                 break
#             # 模擬滾動以加載更多帖子，這部分可視情況選擇是否需要
#             driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

#             # 每次迴圈結束都將資料保存到 CSV 檔案
#             df_posts = pd.DataFrame(all_posts_data)
#             df_posts.to_csv(output_path, index=False)
#             print(f"Saved data to {output_path}")

#     except Exception as e:
#         print(f"Error: {e}")

#     finally:
#         driver.quit()

#     return all_posts_data

# # 設定論壇基本網址
# base_forum_url = "https://www.personalitycafe.com/forums/enfj-forum-the-protectors.17/"

# # 設定要爬取的作者數目
# author_limit = 40

# # 設定輸出檔案的路徑
# output_csv_path = "./MBTI_project/data_personality/enfj_data.csv"

# # 爬取資料
# all_posts_data = crawl_enfj_forum(base_forum_url, author_limit, output_path=output_csv_path)

# # 將所有的資料轉換成 DataFrame
# df_all_posts = pd.DataFrame(all_posts_data)

# # 檢查爬取到的作者數量是否足夠
# if len(authors_set) < author_limit:
#     print(f"Warning: Only crawled {len(authors_set)} unique authors, which is less than the required {author_limit}. The data may not be sufficient.")


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
base_forum_url = "https://www.personalitycafe.com/forums/enfj-forum-the-givers.17/page-14?sorting=latest-activity"

# 現有的 CSV 檔案路徑
existing_csv_path = "data_personality/enfj_data1.csv"

# 爬取資料
crawl_single_page(base_forum_url, existing_csv_path)

#爬到14頁https://www.personalitycafe.com//threads/unhealthy-enfj-i-hate-my-personality-how-do-i-change-it.1066434/
