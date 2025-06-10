from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import csv

# Get the symbol from the user
symbol = input("Enter the company symbol: ")

# Initialize the WebDriver
driver = webdriver.Chrome()

try:
    # Navigate to the URL with the entered symbol
    driver.get(f"https://merolagani.com/CompanyDetail.aspx?symbol={symbol}#0")

    # Wait for the fourth tab to be clickable and click it
    tab_xpath = '//*[@id="navHistory"]'
    fourth_tab = WebDriverWait(driver, 5).until(
        EC.element_to_be_clickable((By.XPATH, tab_xpath))
    )
    fourth_tab.click()

    # Wait for the table to be present
    table_xpath = '//*[@id="ctl00_ContentPlaceHolder1_CompanyDetail1_divDataPrice"]/div[2]/table/tbody'
    next_page_xpath = '//*[@id="ctl00_ContentPlaceHolder1_CompanyDetail1_divDataPrice"]/div[1]/div[2]/a[6]'

    with open(f"{symbol}.csv", "w", newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

        while True:
            # Wait for the table to be present
            table_body = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.XPATH, table_xpath))
            )

            # Extract table rows
            table_rows = table_body.find_elements(By.TAG_NAME, "tr")

            # Iterate over rows and extract cell data
            for row in table_rows:
                table_data = row.find_elements(By.CSS_SELECTOR, "th, td")
                row_data = [data.text for data in table_data]
                writer.writerow(row_data)

            print("Data written")

            try:
                # Check for the presence of the "Next" button and click it
                next_page_button = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.XPATH, next_page_xpath))
                )
                next_page_button.click()
            except TimeoutException:
                # No more pages to navigate
                break

except (TimeoutException, NoSuchElementException):
    print(f"Error: The entered symbol '{symbol}' does not match any webpage or the webpage took too long to load.")
finally:
    # Close the browser
    driver.quit()
