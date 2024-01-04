from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

driver = webdriver.Chrome()
driver.implicitly_wait(10)
url = "https://www.personalitycafe.com/forums/infj-forum-the-protectors.18/"
driver.get(url)

urls = []

try:
    for _ in range(10):
        # 找到下一頁按鈕元素
        next_page_button = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'a.pageNav-jump--next'))
        )

        # 獲取當前頁面的網址並存儲
        current_url = driver.current_url
        urls.append(current_url)

        # 點擊下一頁按鈕
        next_page_button.click()

except Exception as e:
    print(f"Error: {e}")

finally:
    # 關閉瀏覽器
    driver.quit()

# 將爬取的網址存儲到文件中
with open("mbti_urls/infj_urls.txt", "w") as file:
    for url in urls:
        file.write(url + "\n")
