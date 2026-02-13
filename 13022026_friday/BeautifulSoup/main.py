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










if __name__ == "__main__":
    data = parsing_using_beautifulsoup()
    store_data_in_list(data)


