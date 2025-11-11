import requests
 
# Replace with your actual API key and endpoint

API_KEY = 'your_api_key'

BASE_URL = 'https://api.lseg.com/tick-history/v1/'
 
# Function to retrieve tick data

def get_tick_data(instrument, start_date, end_date):

    headers = {

        'Authorization': f'Bearer {API_KEY}',

        'Content-Type': 'application/json'

    }

    # Construct the URL with parameters

    url = f"{BASE_URL}ticks?instrument={instrument}&startDate={start_date}&endDate={end_date}"

    # Make the GET request

    response = requests.get(url, headers=headers)

    # Check if the request was successful

    if response.status_code == 200:

        return response.json()  # Return the JSON response

    else:

        print(f"Error: {response.status_code} - {response.text}")

        return None
 
# Example usage

instrument = 'AAPL'  # Example instrument

start_date = '2023-01-01'

end_date = '2023-01-31'
 
tick_data = get_tick_data(instrument, start_date, end_date)

print(tick_data)

 
import yfinance as yf
 
# Specify the stock symbol

symbol = 'AAPL'
 
# Get the stock data

stock = yf.Ticker(symbol)
 
# Get options expiration dates

expiration_dates = stock.options

print("Expiration Dates:", expiration_dates)
 
# Get options data for a specific expiration date

options_data = stock.option_chain(expiration_dates[0])  # Get data for the first expiration date

print("Calls:\n", options_data.calls)

print("Puts:\n", options_data.puts)

 