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
    print(soup.title)
    print(soup.title.text)





