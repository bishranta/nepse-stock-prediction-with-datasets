from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select
import time
import os

# URL of the website
url = "https://nepsealpha.com/nepse-data"

# Setup WebDriver
driver = webdriver.Chrome()  # Use the appropriate driver (e.g., chromedriver)
driver.get(url)

# Wait for the page to load
wait = WebDriverWait(driver, 10)

# Maximize the browser window
driver.maximize_window()

# Directory to save CSV files
save_dir = "nepse_data"
os.makedirs(save_dir, exist_ok=True)

# Step 1: Set the maximum date range
try:
    # Wait for the date fields to be visible
    start_date = wait.until(EC.element_to_be_clickable((By.XPATH, '//*[@id="nepse_app_content"]/div[3]/div/div/div/form/div/div[2]/div[2]/input')))
    end_date = driver.find_element(By.XPATH, '//*[@id="nepse_app_content"]/div[3]/div/div/div/form/div/div[2]/div[3]/input')

    # Input the date range (adjust dates as needed)
    start_date.clear()
    start_date.send_keys("20-01-2020")  # Adjust to the earliest available date
    end_date.clear()
    end_date.send_keys("07-01-2025")  # Adjust to today's date

except Exception as e:
    print("Error setting the date range:", e)
    driver.quit()

# Step 2: Get all available symbols
try:
    # Locate the dropdown for symbols
    symbol_dropdown = Select(driver.find_element(By.XPATH, '//*[@id="nepse_app_content"]/div[3]/div/div/div/form/div/div[2]/div[4]/select'))

    # Fetch all symbol options
    all_symbols = [option.get_attribute("value") for option in symbol_dropdown.options if option.get_attribute("value")]
    print(f"Found {len(all_symbols)} symbols.")

    # Step 3: Loop through symbols and download data
    for symbol in all_symbols:
        print(f"Processing symbol: {symbol}")
        try:
            # Select the symbol
            symbol_dropdown.select_by_value(symbol)

            # Click the Filter button
            filter_button = driver.find_element(By.XPATH, '//*[@id="nepse_app_content"]/div[3]/div/div/div/form/div/div[2]/div[5]/button')
            filter_button.click()

            # Wait for the data to load (adjust timeout if necessary)
            wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="result-table"]')))
            time.sleep(5)  # Add a slight delay to ensure data loads fully

            # Click the Download CSV button
            download_button = wait.until(
                EC.element_to_be_clickable((By.XPATH, '//*[@id="result-table_wrapper"]/div[1]/button[4]'))
            )
            download_button.click()

            # Wait to ensure the file is downloaded
            print(f"Downloading data for {symbol}. Waiting for file to download...")
            time.sleep(5)  # Adjust this delay based on download time

        except Exception as e:
            print(f"Error processing symbol {symbol}: {e}")
            continue

except Exception as e:
    print("Error processing symbols:", e)

# Step 4: Close the browser
driver.quit()
print("Script completed.")
