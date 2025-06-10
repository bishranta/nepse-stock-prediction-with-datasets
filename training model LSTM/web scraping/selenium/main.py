from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import csv

symbol = input("Enter the company symbol: ")
driver = webdriver.Chrome()
try:
    driver.get(f"https://merolagani.com/CompanyDetail.aspx?symbol={symbol}#0")
    tab_xpath = '//*[@id="navHistory"]'
    fourth_tab = WebDriverWait(driver, 5).until(
        EC.element_to_be_clickable((By.XPATH, tab_xpath))
    )
    fourth_tab.click()

    table_xpath = '//*[@id="ctl00_ContentPlaceHolder1_CompanyDetail1_divDataPrice"]/div[2]/table/tbody'
    table_body = WebDriverWait(driver, 5).until(
        EC.presence_of_element_located((By.XPATH, table_xpath))
    )

    table_rows = table_body.find_elements(By.TAG_NAME, "tr")

    with open(f"{symbol}.csv", "w", newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        for row in table_rows:
            table_data = row.find_elements(By.CSS_SELECTOR, "th, td")
            row_data = [data.text for data in table_data]
            writer.writerow(row_data)

except (TimeoutException, NoSuchElementException):
    print(f"Error: The entered symbol '{symbol}' does not match any webpage or the webpage took too long to load.")
finally:
    driver.quit()
