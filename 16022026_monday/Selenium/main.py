from selenium import webdriver
from selenium.webdriver.common.by import By
# from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import pandas as pd
# import os

url = "https://www.scrapethissite.com/pages/forms/?page_num=1"

def create_driver():
    """
    :return: None
    Basic driver configuration
    """
    options = Options()
    options.add_argument("--start-maximized")
    driver = webdriver.Chrome(options=options)
    return driver

def open_page(driver):
    """
    :param driver:
    Opens the page
    :return: None
    """
    driver.get(url)

def extract_table(driver):
    """
    :param driver:
    Function to extract table from page
    :return: None
    """
    rows = driver.find_elements(By.CSS_SELECTOR, "table.table tbody tr")
    new_data = []
    for row in rows:
        cols = row.find_elements(By.CSS_SELECTOR, "td")
        cols_text= [col.text.strip() for col in cols]
        new_data.append(cols_text)
    return new_data

def paginate(driver,page = 5):
    """
    :param driver:
    :param page: Page number till where we need to paginate
    Function to extract table from multiple pages
    :return:
    """
    base_url = "https://www.scrapethissite.com/pages/forms/"
    all_data = []
    cols = []
    for page in range(1, page+1):
        driver.get(f"{base_url}?page_num{page}")
        cols = [col.text.strip() for col in driver.find_elements(By.CSS_SELECTOR, "th")]
        table_data = extract_table(driver)
        all_data.extend(table_data)
    df = pd.DataFrame(all_data,columns=cols)
    return df.head()

if __name__ == "__main__":
    driver = create_driver()
    # open_page(driver)
    data = paginate(driver)
    print(data)