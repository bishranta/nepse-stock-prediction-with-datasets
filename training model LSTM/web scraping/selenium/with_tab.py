from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
import csv
import time
import pandas as pd

symbol = "NABIL"
# symbol = input('Enter symbol:')
driver = webdriver.Chrome()
driver.get(f"https://merolagani.com/CompanyDetail.aspx?symbol={symbol}#0")

tab_xpath = '//*[@id="navHistory"]'
fourth_tab = driver.find_element(By.XPATH, tab_xpath)
fourth_tab.click()
time.sleep(3)

table_xpath = '//*[@id="ctl00_ContentPlaceHolder1_CompanyDetail1_divDataPrice"]/div[2]/table/tbody'
table_body = driver.find_element(By.XPATH, table_xpath)
table_rows = table_body.find_elements(By.TAG_NAME, "tr")
with open("DATA.csv", "w", newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    for row in table_rows:
        table_data = row.find_elements(By.CSS_SELECTOR, "th, td")
        row_data = [data.text for data in table_data]
        writer.writerow(row_data)

time.sleep(5)
driver.quit()