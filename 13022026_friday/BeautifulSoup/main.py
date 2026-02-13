import csv
from urllib.parse import urljoin
from bs4 import BeautifulSoup
import requests
from sqlalchemy.orm.base import CALLABLES_OK

# url = "https://www.crummy.com/software/BeautifulSoup/bs4/doc/"
url = "https://beautiful-soup-4.readthedocs.io/en/latest/"
second_url = "template.html"

def basic_fetching():
    response = requests.get(url)
    print(response.status_code)
    print(response.text[:500])   # The datatype of response is string

def parsing_using_beautifulsoup():
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    # print(soup.title)
    # print(soup.title.text,"\n\n")
    return soup

def parsing_using_beautifulsoup_second():
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    # print(soup.title)
    # print(soup.title.text,"\n\n")
    return soup

def finding_elements(soup):
    single_h3 = soup.find("h3")
    print(single_h3)
    all_h3 = soup.find_all("h3")
    print(all_h3)
    section_id = soup.find_all("section" ,id="quick-start")
    for texts in section_id:
        print(texts.text)

def extracting_attributes(soup):
    links = soup.find_all("a")
    for link in links:
        print(link.get("href"))


def store_data_in_list(soup):
    page_headings = []
    main_headings = soup.find_all("h1")
    sub_headings = soup.find_all("h2")
    for heading,subheading in zip(main_headings,sub_headings):
        page_headings.append({
            "main_heading": heading.text,
            "sub_heading": subheading.text
        })
    print(page_headings)

def store_to_csv(soup):
    # 1. Find all div tags with the class "highlight"
    codes = soup.select_one("div.highlight-default pre")

    # 2. Transform the Tag objects into a list of dictionaries containing just the text
    # We use .get_text() to extract only the code snippets
    data_to_save = [{"code snippets": tag.get_text().strip()} for tag in codes]

    # 3. Open file with newline="" to avoid blank rows
    with open("codes.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["code snippets"])
        writer.writeheader()

        # 4. Now writerows receives a list of dicts, which it expects
        writer.writerows(data_to_save)

def link_extractor(soup):
    links = soup.find_all("a")
    for link in links:
        href = link.get("href")
        full_href = urljoin(url, href)
        print(full_href)

def image_extractor(soup):
    images = soup.find_all("img")
    for image in images:
        print(image.get("src"))
        print(image.get("alt"))

def get_table(soup):
    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        for row in rows:
            cols = row.find_all(["td","th"])
            data = [ele.text.strip() for ele in cols]
            print(f"Table Row {data}")

def add_new_element(soup):
    ul_tag = soup.new_tag("ul")
    li_tag = soup.new_tag("li")
    li_tag.string = "Home"
    ul_tag.append(li_tag)
    li_tag = soup.new_tag("li")
    li_tag.string = "Work"
    ul_tag.append(li_tag)
    soup.html.body.insert(0,ul_tag)
    with open("template.html","w",encoding="utf-8") as f:
        f.write(str(soup))



if __name__ == "__main__":
    data = parsing_using_beautifulsoup()
    data2 = parsing_using_beautifulsoup_second()
    add_new_element(data2)



