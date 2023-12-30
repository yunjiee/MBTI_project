# from bs4 import BeautifulSoup
# from urllib.request import urlopen, Request
# from tqdm import tqdm
# from selenium import webdriver
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
# from bs4 import BeautifulSoup
# from time import sleep

# HEADERS = {
#     'user-agent': ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
#                    '(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'),
#     'referer': 'http://www.google.com'
# }

# f = open("ENFJ.txt", "a+")
# #change this

# urls=['https://www.personalitycafe.com/forums/enfj-forum-the-protectors.17/']
# #change this

# num_page = [100]
# #change this


# for i in tqdm(range(num_page)):
#     req = Request(url=urls[url_num] + str(i+1) + '.html', headers=HEADERS)
#     page = urlopen(req)
#     soup = BeautifulSoup(page, 'html.parser')

#     doc = ""
#     for p_tag in soup.find_all('blockquote','restore'):
#         para = p_tag.text.strip()
#         if len(para)==0:
#             continue
#         doc += para + '\n'

#     try:
#         f.write(doc)
#     except:
#         continue

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import os

def crawl_enfj_forum(url, num_pages=5):
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
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, 'message'))
            )
            thread_posts_soup = BeautifulSoup(driver.page_source, 'html.parser')

            # 找到第一個帖子的發文者和內容
            # post_user_element = thread_posts_soup.find('a', class_='MessageCard__user-info__name')
            post_content_element = thread_posts_soup.find('div', class_='bbWrapper')

            # post_username = post_user_element.text.strip() if post_user_element else ''
            post_content = post_content_element.text.strip() if post_content_element else ''

            posts_data.append({
                # 'Personality Type' : 'ENFJ',
                # 'Username': post_username,
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

# 爬取 ENFJ 類型的帖子資料
enfj_forum_url = "https://www.personalitycafe.com/forums/enfj-forum-the-protectors.17/"
enfj_posts_data = crawl_enfj_forum(enfj_forum_url, num_pages=5)

# 確保 data/archive 資料夾存在
archive_folder = './data'
os.makedirs(archive_folder, exist_ok=True)
file_path = os.path.join(archive_folder, 'ENFJ.txt')

#將資料寫入 txt 檔案
with open( file_path, 'w', encoding='utf-8' ) as txt_file:
    for post_data in enfj_posts_data:
        # txt_file.write(f"Personality Type: {post_data['Personality Type']}\n")
        # txt_file.write(f"Username: {post_data['Username']}\n")
        txt_file.write(f"Content: {post_data['Content']}\n")
        txt_file.write("\n")

# 輸出結果
for post_data in enfj_posts_data:
    # print(f"Username: {post_data['Username']}")
    print(f"Content: {post_data['Content']}")
    print("-----")
