import csv

from bs4 import BeautifulSoup
import requests

url = "https://www.crummy.com/software/BeautifulSoup/bs4/doc/"

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















if __name__ == "__main__":
    data = parsing_using_beautifulsoup()
    store_to_csv(data)


