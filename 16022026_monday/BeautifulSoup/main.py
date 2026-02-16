from bs4 import BeautifulSoup
import requests

url = "https://www.crummy.com/software/BeautifulSoup/bs4/doc/"

response = requests.get(url)

soup = BeautifulSoup(response.text,"html.parser")
# print(soup.title.text) # Extracting the title

# Extracting all the h1,h2 and h3
headings = soup.find_all(["h1","h2","h3"])
# for heading in headings:
#    print(heading.get_text(strip=True))

# Extracting navigation links
links = soup.find_all("a")
# for link in links:
#     href = link.get("href")
#     text = link.get_text(strip=True)
#     if href:
#         print(text,"->",href)

# Extracting a specific section
# section  = soup.find_all("section",id="quick-start")
# for texts in section:
#     print(texts.text)

# # Extracting code blocks
# code_blocks = soup.select("div.highlight pre")
# for code in code_blocks:
#     print("-------------CODE-------------")
#     print(code.text)

# Scrap all internal documentation pages
# all_links = soup.find_all("a")
# internal_links = []
# for link in all_links:
#     href = link.get("href")
#
#     if href and href.startswith("#"):
#         internal_links.append(href)
#
# print(internal_links)

# # Saving the important documentation to text file
# with open("bs4_docs.txt","w",encoding="utf-8") as f:
#     for heading in soup.find_all(["h1","h2","h3"]):
#         f.write(heading.get_text(strip=True) + "\n\n")
#     for code in soup.select("div.highlight pre"):
#         f.write("CODE:\n")
#         f.write(code.get_text() + "\n\n")

# # Advance: Scarping section by section
# sections = soup.find_all("section")
# for section in sections:
#     section_id = section.get("id")
#     title = section.find(["h1","h2","h3"])
#     if title:
#         print("SECTION: ",title.get_text(strip=True))
#     print("-"*50)

# Advance selection
# print(f"Select by id {soup.select_one("#kinds-of-objects")}")
print(f"Selecting all code blocks {soup.select("div.highlight > pre")}")




