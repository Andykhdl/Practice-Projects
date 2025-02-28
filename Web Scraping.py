import requests # type: ignore
from bs4 import BeautifulSoup # type: ignore
import pandas as pd # type: ignore
import time

headers = {'user-agent':'Mozilla/5.0 \
            (Windows NT 10.0; Win64; x64) \
            AppleWebKit/537.36 (KHTML, like Gecko) \
            Chrome/84.0.4147.105 Safari/537.36'}

urls = [
    'https://groww.in/us-stocks/nke',
    'https://groww.in/us-stocks/ko', 
    'https://groww.in/us-stocks/aapl',
    'https://groww.in/us-stocks/ba', 
    'https://groww.in/us-stocks/msft', 
    'https://groww.in/stocks/m-india-ltd', 
    'https://groww.in/us-stocks/axp', 
    'https://groww.in/us-stocks/amgn', 
    'https://groww.in/us-stocks/csco', 
    'https://groww.in/us-stocks/gs', 
    'https://groww.in/us-stocks/ibm', 
    'https://groww.in/us-stocks/intc', 
    'https://groww.in/us-stocks/jpm', 
    'https://groww.in/us-stocks/mcd',
    'https://groww.in/us-stocks/crm', 
    'https://groww.in/us-stocks/vz', 
    'https://groww.in/us-stocks/v', 
    'https://groww.in/us-stocks/wmt',  
    'https://groww.in/us-stocks/dis'
    ]

all= []

for url in urls:
    page = requests.get(url,headers=headers)

    try:
        soup = BeautifulSoup(page.text, 'html.parser')
        company = soup.find('h1', {'class': 'usph14Head displaySmall'}).text # type: ignore
        price = soup.find('span', {'class': 'uht141Pri contentPrimary displayBase'})       
        change = soup.find('div', {'class': 'uht141Day bodyBaseHeavy contentNegative'}).text # type: ignore
        volume_tag=soup.find('table', {'class': 'tb10Table borderPrimary width100 usp100NoBorder usp100Table'})
        if volume_tag:
            rows = volume_tag.find_all('tr')  # type: ignore # Find all rows
            if len(rows) > 1:  # Check if second row exists
                cells = rows[1].find_all('td')  # Find all cells in the second row
                if len(cells) > 2:  # Check if the third cell exists
                    volume = cells[2].text.strip() 
        x=[company,price,change,volume] # type: ignore
        all.append(x)
        
    except AttributeError:
      print("Change the Element id")
    # Wait for a short time to avoid rate limiting
    time.sleep(1)

column_names = ["Company", "Price", "Change","Volume"]
df = pd.DataFrame(columns = column_names)

for i in all:
  index=0
  df.loc[index] = i
  df.index = df.index + 1
df=df.reset_index(drop=True)

print(df)
