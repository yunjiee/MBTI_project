import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

# 定義在外面，保持相同的狀態
authors_set = set()
all_posts_data = []

def crawl_infj_forum(base_url, author_limit, output_path="infj_posts_data.csv"):
    driver = webdriver.Chrome()
    driver.implicitly_wait(60)

    try:
        # 初始頁面
        url = base_url
        driver.get(url)
        WebDriverWait(driver, 60).until(
            EC.presence_of_element_located((By.CLASS_NAME, 'structItem-title'))
        )

        soup = BeautifulSoup(driver.page_source, 'html.parser')

        # 找到qid為"page-nav-other-page"的元素
        page_nav_element = soup.select_one('[qid="page-nav-other-page"]')

        # 從中獲取頁數
        total_pages = int(page_nav_element.text)
        print(total_pages)

        for page in range(1, total_pages + 1):
            # 動態生成網址
            if page == 1:
                url = base_url
            else:
                url = f"{base_url}/page-{page}?sorting=latest-activity"

            driver.get(url)
            WebDriverWait(driver, 60).until(
                EC.presence_of_element_located((By.CLASS_NAME, 'structItem-title'))
            )

            soup = BeautifulSoup(driver.page_source, 'html.parser')
            thread_links = soup.find_all('a', class_='thread-title--gtm')

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

                    # 檢查是否已經爬取過該作者的貼文
                    if author_name not in authors_set:
                        authors_set.add(author_name)

                        # 處理該作者的貼文內容
                        all_posts_data.append({
                            'Author': author_name,
                            'Thread_Title': thread_title,  # 加入貼文標題
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

            # 每次迴圈結束都將資料保存到 CSV 檔案
            df_posts = pd.DataFrame(all_posts_data)
            df_posts.to_csv(output_path, index=False)
            print(f"Saved data to {output_path}")

    except Exception as e:
        print(f"Error: {e}")

    finally:
        driver.quit()

    return all_posts_data

# 設定論壇基本網址
base_forum_url = "https://www.personalitycafe.com/forums/infj-forum-the-protectors.18/"

# 設定要爬取的作者數目
author_limit = 500

# 設定輸出檔案的路徑
output_csv_path = "data_personality/infj_data.csv"

# 爬取資料
all_posts_data = crawl_infj_forum(base_forum_url, author_limit, output_path=output_csv_path)

# 將所有的資料轉換成 DataFrame
df_all_posts = pd.DataFrame(all_posts_data)

# 檢查爬取到的作者數量是否足夠
if len(authors_set) < author_limit:
    print(f"Warning: Only crawled {len(authors_set)} unique authors, which is less than the required {author_limit}. The data may not be sufficient.")
