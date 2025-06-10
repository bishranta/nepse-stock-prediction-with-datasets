import requests
from bs4 import BeautifulSoup
import csv
from datetime import datetime
import schedule
import time

def scrape_and_write_to_csv():
    url = "https://www.sharesansar.com/live-trading"
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'html.parser')
    table = soup.find('table')

    current_date = datetime.now().strftime('%Y-%m-%d')
    filename = f"{current_date}.csv"

    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)

        for row in table.find_all('tr'):
            row_data = [data.strip().replace('"', '').replace(' ', '') for data in row.text.split()]
            csvwriter.writerow(row_data)

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
