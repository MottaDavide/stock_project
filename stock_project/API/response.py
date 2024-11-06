import requests

# Define the base URL of your API
base_url = "http://127.0.0.1:8000/stock"

# Define the stock_code you want to query
stock_code = str(input("Write here the Stock Code (e.g. 10002):"))  # Example stock code

# Send a GET request to the API endpoint
response = requests.get(f"{base_url}/{stock_code}")

# Check if the request was successful
if response.status_code == 200:
    # Parse and print the response JSON (the dictionary)
    data = response.json()
    item = data['item']
    bias = data['bias']
    rmse = data['rmse']
    message = data['message']
    print(f"Selected Stock: {item}")
    print(f"{message} with bias {bias} and RMSE {rmse:.2f}")
else:
    print(f"Error: {response.status_code}, {response.text}")