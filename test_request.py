import requests

# Define the API endpoint
url = "http://127.0.0.1:8000/predict"  # Ensure this matches your FastAPI server URL

# Sample data for testing
data = {
    "transaction_value": 1000.0,
    "frequency": 5
}

# Make the POST request
response = requests.post(url, json=data)

# Print the result
print("Status Code:", response.status_code)
try:
    print("Response JSON:", response.json())
except requests.exceptions.JSONDecodeError:
    print("Error: Response is not in JSON format")
