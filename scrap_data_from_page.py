import requests
from bs4 import BeautifulSoup

url = "https://hardskill.exchange/summit/agentic-ai-summit/"
html = requests.get(url).text

soup = BeautifulSoup(html, "html.parser")
text = soup.get_text(separator="\n")

with open("data_base.txt", "a", encoding="utf-8") as f:
    f.write("\n=== Source: Agentic AI Summit page ===\n")
    f.write(text)
    f.write("\n=== End Source ===\n")
