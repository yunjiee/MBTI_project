# from selenium import webdriver
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
# from bs4 import BeautifulSoup

# def crawl_infj_forum(url, num_posts=5):
#     driver = webdriver.Chrome()  # 請根據你使用的瀏覽器選擇對應的WebDriver
#     driver.get(url)

#     # 等待頁面載入完成
#     WebDriverWait(driver, 10).until(
#         EC.presence_of_element_located((By.CLASS_NAME, 'structItem-title'))
#     )

#     # 使用 BeautifulSoup 解析頁面
#     soup = BeautifulSoup(driver.page_source, 'html.parser')

#     # 找到所有主題連結
#     thread_links = soup.find_all('a', class_='thread-title--gtm')

#     posts_data = []

#     for i, thread_link in enumerate(thread_links):
#         if i >= num_posts:
#             break

#         thread_url_relative = thread_link['href']
#         thread_url = f"https://www.personalitycafe.com/{thread_url_relative}"

#         # 進入主題連結，進入帖子頁面
#         driver.get(thread_url)
#         WebDriverWait(driver, 10).until(
#             EC.presence_of_element_located((By.CLASS_NAME, 'message'))
#         )
#         thread_posts_soup = BeautifulSoup(driver.page_source, 'html.parser')

#         # 找到帖子的發文者和內容
#         post_user_elements = thread_posts_soup.find_all('a', class_='MessageCard__user-info__name')
#         post_content_elements = thread_posts_soup.find_all('div', class_='bbWrapper')

#         post_usernames = [user.text.strip() for user in post_user_elements]
#         post_contents = [content.text.strip() for content in post_content_elements]

#         for post_username, post_content in zip(post_usernames, post_contents):
#             posts_data.append({
#                 'Username': post_username,
#                 'Content': post_content
#             })

#     # 關閉 WebDriver
#     driver.quit()

#     return posts_data

# # 爬取 INFJ 類型的前5篇帖子資料
# infj_forum_url = "https://www.personalitycafe.com/forums/infj-forum-the-protectors.18/"
# infj_posts_data = crawl_infj_forum(infj_forum_url, num_posts=5)

# # 輸出結果
# for post_data in infj_posts_data:
#     print(f"Username: {post_data['Username']}")
#     print(f"Content: {post_data['Content']}")
#     print("-----")

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup

def crawl_infj_forum(url, num_posts=5):
    driver = webdriver.Chrome()  # 請根據你使用的瀏覽器選擇對應的WebDriver
    driver.get(url)

    # 等待頁面載入完成
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CLASS_NAME, 'structItem-title'))
    )

    # 使用 BeautifulSoup 解析頁面
    soup = BeautifulSoup(driver.page_source, 'html.parser')

    # 找到所有主題連結
    thread_links = soup.find_all('a', class_='thread-title--gtm')

    posts_data = []

    for i, thread_link in enumerate(thread_links):
        if i >= num_posts:
            break

        thread_url_relative = thread_link['href']
        thread_url = f"https://www.personalitycafe.com/{thread_url_relative}"

        # 進入主題連結，進入帖子頁面
        driver.get(thread_url)
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, 'message'))
        )
        thread_posts_soup = BeautifulSoup(driver.page_source, 'html.parser')

        # 找到第一個帖子的發文者和內容
        post_user_element = thread_posts_soup.find('a', class_='MessageCard__user-info__name')
        post_content_element = thread_posts_soup.find('div', class_='bbWrapper')

        post_username = post_user_element.text.strip() if post_user_element else ''
        post_content = post_content_element.text.strip() if post_content_element else ''

        posts_data.append({
            'Username': post_username,
            'Content': post_content
        })

    # 關閉 WebDriver
    driver.quit()

    return posts_data

# 爬取 INFJ 類型的前5篇帖子資料
infj_forum_url = "https://www.personalitycafe.com/forums/infj-forum-the-protectors.18/"
infj_posts_data = crawl_infj_forum(infj_forum_url, num_posts=5)

# 輸出結果
for post_data in infj_posts_data:
    print(f"Username: {post_data['Username']}")
    print(f"Content: {post_data['Content']}")
    print("-----")

