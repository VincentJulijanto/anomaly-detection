import os
import requests
import pandas as pd
from datetime import datetime

def fetch_crypto_prices():
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        "vs_currency": "usd",
        "order": "market_cap_desc",
        "per_page": 10,
        "page": 1,
        "sparkline": "false"
    }
    try:
        print(f"Current working directory: {os.getcwd()}")  # Debug: Current directory
        response = requests.get(url, params=params)
        if response.status_code == 200:
            print("Successfully fetched cryptocurrency data.")  # Debug: API success
            data = response.json()
            # Convert to DataFrame
            df = pd.DataFrame(data)
            print("Converting data to DataFrame...")  # Debug: DataFrame creation
            df = df[["id", "symbol", "current_price", "price_change_percentage_24h", "market_cap"]]
            df["timestamp"] = datetime.now()
            # Check if 'data/' directory exists
            if not os.path.exists("data"):
                print("Creating 'data' directory...")
                os.makedirs("data")
            # Save to CSV
            file_path = "data/crypto_data.csv"
            print(f"Saving file to {file_path}...")  # Debug: File saving
            df.to_csv(file_path, index=False)
            print(f"File saved to {file_path}.")  # Debug: Success message
        else:
            print(f"Error fetching cryptocurrency data: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    fetch_crypto_prices()
