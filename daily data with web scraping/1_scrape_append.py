import requests
from bs4 import BeautifulSoup
import csv
import os
from datetime import datetime

def scrape_and_write_to_csv():
    url = 'https://www.sharesansar.com/live-trading'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Extract the date from the <span> tag
    date_span = soup.find('span', {'id': 'dDate'})
    if date_span:
        date_text = date_span.text.strip()  # Extract and strip text
        current_date = date_text.split()[0]

    table = soup.find('table')
    if table:
        rows = table.find_all('tr')[1:]  # Skip the header row

        for row in rows:
            try:
                cells = row.find_all('td')
                if len(cells) > 6:  # Ensure there are enough cells for each field
                    # Extract relevant data matching the CSV structure
                    stock_symbol = cells[1].text.strip()
                    data = [
                        stock_symbol,  # Symbol
                        current_date,  # Date
                        cells[5].text.strip(),  # Open
                        cells[6].text.strip(),  # High
                        cells[7].text.strip(),  # Low
                        cells[2].text.strip(),  # Close
                        cells[4].text.strip() + ' %',  # Percent Change
                        cells[8].text.strip()  # Volume
                    ]

                    csv_filename = f"{stock_symbol}.csv"
                    file_mode = 'a+' if os.path.exists(csv_filename) else 'w'

                    # Check if the data for the current_date already exists in the file
                    if os.path.exists(csv_filename):
                        with open(csv_filename, 'r') as file:
                            last_line = file.readlines()[-1].strip()  # Get the last line
                            if last_line.split(',')[1] == current_date:  # Compare the date
                                print(f"Data for {current_date} already exists in '{csv_filename}'. Skipping.")
                                continue

                    with open(csv_filename, mode=file_mode, newline='') as file:
                        # Check the cursor position and ensure it starts on a new line
                        if file_mode == 'a+':  # Only check for append mode
                            file.seek(0, os.SEEK_END)  # Move the cursor to the end of the file
                            # If the file is not empty, move the cursor back to check the last line
                            if file.tell() > 0:
                                file.seek(file.tell() - 1, os.SEEK_SET)  # Move back 1 character
                                while file.tell() > 0 and file.read(1) != '\n':  # Find the start of the last line
                                    file.seek(file.tell() - 2, os.SEEK_SET)

                                last_line = file.readline()  # Read the last line

                                if last_line.strip():  # If the last line is not empty
                                    file.write('\n')  # Write a newline before adding new data

                        writer = csv.writer(file)
                        if file_mode == 'w':
                            # Write header if file is new
                            writer.writerow(["Symbol", "Date", "Open", "High", "Low", "Close", "Percent Change", "Volume"])
                            print(f"CSV file '{csv_filename}' created with header.")
                        writer.writerow(data)
                        print(f"Data appended to '{csv_filename}'.")

            except Exception as e:
                # Log the error and skip the problematic row
                print(f"Error processing row: {row}. Skipping to next row. Error: {e}")

scrape_and_write_to_csv()
