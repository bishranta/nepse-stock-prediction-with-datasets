import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import schedule
import time

def scrape_and_write_to_csv():
    url = "https://www.sharesansar.com/live-trading"
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'html.parser')

    rows = soup.find('table', {'id': 'headFixed'}).find('tbody').find_all('tr')
    data = []

    for row in rows:
        dic = {}
        dic['SN'] = row.find_all('td')[0].text.replace(',', '').replace(' ', '').replace('\n', '')
        dic['Symbol'] = row.find_all('td')[1].text.replace(',', '').replace(' ', '').replace('\n', '')
        dic['LTP'] = row.find_all('td')[2].text.replace(',', '').replace(' ', '').replace('\n', '')
        dic['%Change'] = row.find_all('td')[4].text.replace(',', '').replace(' ', '').replace('\n', '')
        dic['Open'] = row.find_all('td')[5].text.replace(',', '').replace(' ', '').replace('\n', '')
        dic['High'] = row.find_all('td')[6].text.replace(',', '').replace(' ', '').replace('\n', '')
        dic['Low'] = row.find_all('td')[7].text.replace(',', '').replace(' ', '').replace('\n', '')
        dic['Volume'] = row.find_all('td')[8].text.replace(',', '').replace(' ', '').replace('\n', '')
        dic['P.Close'] = row.find_all('td')[9].text.replace(',', '').replace(' ', '').replace('\n', '')
        dic['Difference'] = row.find_all('td')[3].text.replace(',', '').replace(' ', '').replace('\n', '')
        data.append(dic)

    current_date = datetime.now().strftime('%Y-%m-%d')

    df = pd.DataFrame(data)
    # df.to_excel(f'{current_date}.xlsx', index=False)
    df.to_csv(f'{current_date}.csv', index=False)

    print(f"Data written to file '{current_date}'")

scrape_and_write_to_csv()
# schedule.every().sunday.at("15:30").do(scrape_and_write_to_csv)
# schedule.every().monday.at("15:30").do(scrape_and_write_to_csv)
# schedule.every().tuesday.at("15:30").do(scrape_and_write_to_csv)
# schedule.every().wednesday.at("15:30").do(scrape_and_write_to_csv)
# schedule.every().thursday.at("15:30").do(scrape_and_write_to_csv)
#
# while True:
#     schedule.run_pending()
#     time.sleep(1)
