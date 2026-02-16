import requests
from bs4 import BeautifulSoup
import pandas as pd

url = "https://www.scrapethissite.com/pages/forms/"

def basic_config():
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url,headers=headers)
    soup = BeautifulSoup(response.text,"html.parser")
    print(response.status_code)
    return soup


def extracting_table_headers(soup):
    all_headers = []
    for th in soup.find_all("th"):
        all_headers.append(th.get_text(strip=True))
    print(all_headers)

def extracting_table_rows(soup):
    table = soup.select_one("table.table")
    rows = table.select("tr")[1:]
    header = table.select("tr th")
    all_headers = [h.get_text(strip=True) for h in header]
    print(all_headers)
    for row in rows:
        columns = row.select("td")
        data = [col.get_text(strip=True) for col in columns]
        print(data)

def extracting_table_in_dataframe(soup):
    table = soup.select_one("table.table")
    headers = [th.get_text(strip=True) for th in table.find_all("th")]
    rows = []
    for row in table.select("tr")[1:]:
        cols = [td.get_text(strip=True) for td in row.select("td")]
        rows.append(cols)
    df = pd.DataFrame(rows, columns=headers)
    print(df.to_string(index=False))

def pagination():
    base_url = "https://www.scrapethissite.com/pages/forms/"
    headers = {"User-Agent": "Mozilla/5.0"}
    all_rows = []
    col_names = []
    for page in range(1,6):
        url = f"{base_url}?page_num={page}"
        response = requests.get(url,headers=headers)
        soup = BeautifulSoup(response.text,"html.parser")
        table = soup.select_one("table.table")
        if table is None:
            continue
        if not col_names:
            col_names = [th.get_text(strip=True) for th in table.find_all("th")]
        rows = table.select("tr")[1:]
        for row in rows:
            cols = [td.get_text(strip=True) for td in row.select("td")]
            all_rows.append(cols)
    print(f"Total rows: {len(all_rows)}")
    print(f"Total columns: {len(col_names)}")
    df = pd.DataFrame(all_rows, columns=col_names)
    print(df)
    df.to_csv("table.csv")


if __name__ == "__main__":
    soup = basic_config()
    pagination()




