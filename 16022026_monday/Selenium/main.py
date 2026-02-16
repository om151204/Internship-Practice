from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import os

url = "https://www.scrapethissite.com/pages/forms/?page_num=1"

def create_driver():
    options = Options()
    options.add_argument("--start-maximized")
    driver = webdriver.Chrome(options=options)
    return driver

def open_page(driver):
    driver.get(url)

def extract_table(driver):
    rows = driver.find_elements(By.CSS_SELECTOR, "table.table tbody tr")
    data = []
    for row in rows:
        cols = row.find_elements(By.CSS_SELECTOR, "td")
        cols_text= [col.text.strip() for col in cols]
        data.append(cols_text)
    return data

def paginate(driver,page = 5):
    base_url = "https://www.scrapethissite.com/pages/forms/"
    all_data = []
    for page in range(1, page+1):
        driver.get(f"{base_url}?page_num{page}")
        table_data = extract_table(driver)
        all_data.extend(table_data)
    return all_data

if __name__ == "__main__":
    driver = create_driver()
    # open_page(driver)
    data = paginate(driver)
    print(data)