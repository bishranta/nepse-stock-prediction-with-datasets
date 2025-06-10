import requests

# Define the API endpoint
url = "https://nepsealpha.com/nepse-data"

# Define headers
headers = {
    "accept": "*/*",
    "accept-language": "en-US,en;q=0.5",
    "cache-control": "no-cache",
    "content-type": "application/x-www-form-urlencoded; charset=UTF-8",
    "pragma": "no-cache",
    "priority": "u=1, i",
    "sec-ch-ua": "\"Brave\";v=\"131\", \"Chromium\";v=\"131\", \"Not_A Brand\";v=\"24\"",
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": "\"Windows\"",
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-origin",
    "sec-gpc": "1",
    "x-requested-with": "XMLHttpRequest",
}

# Define form data (body)
data = {
    "symbol": "SBID89",  # Replace with desired symbol
    "specific_date": "2025-01-07",  # Adjust as needed
    "start_date": "2020-01-07",  # Adjust as needed
    "end_date": "2025-01-07",  # Adjust as needed
    "filter_type": "date-range",
    "time_frame": "daily",
    "_token": "qVEqMEaEdP37xrQDEpn5bNxbnniqSGoIxDiYVqga",  # Ensure this is correct
}

# Send the POST request
response = requests.post(url, headers=headers, data=data)

# Check if the request was successful
if response.status_code == 200:
    print("Data fetched successfully!")

    # Save the response to a file (assuming it's CSV or JSON)
    with open("nepse_data.csv", "wb") as file:
        file.write(response.content)
    print("Data saved to nepse_data.csv")

else:
    print(f"Failed to fetch data. Status code: {response.status_code}")
    print("Response:", response.text)
