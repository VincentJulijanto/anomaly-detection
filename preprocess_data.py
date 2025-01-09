import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from geopy.distance import geodesic

# Step 1: Load Blockchain and Behavioral Data
blockchain_data = pd.read_csv('./data/blockchain_data.csv')
credit_data = pd.read_csv('./data/creditcard.csv')

# Step 2: Handle Missing Columns
# Add a dummy `user_id` column if it's missing
if 'user_id' not in blockchain_data.columns:
    blockchain_data['user_id'] = range(len(blockchain_data))
if 'user_id' not in credit_data.columns:
    credit_data['user_id'] = range(len(credit_data))

# Add dummy `latitude` and `longitude` columns if missing
if 'latitude' not in blockchain_data.columns:
    blockchain_data['latitude'] = 0.0  # Default value
if 'longitude' not in blockchain_data.columns:
    blockchain_data['longitude'] = 0.0  # Default value

# Step 3: Feature Engineering
# Transaction Frequency: Count the number of transactions per user
blockchain_data['frequency'] = blockchain_data.groupby('user_id')['transaction_value'].transform('count')

# Calculate location deviation using geopy
blockchain_data['prev_latitude'] = blockchain_data['latitude'].shift(1)
blockchain_data['prev_longitude'] = blockchain_data['longitude'].shift(1)
blockchain_data['location_deviation'] = blockchain_data.apply(
    lambda row: geodesic((row['latitude'], row['longitude']),
                         (row['prev_latitude'], row['prev_longitude'])).km
    if not pd.isnull(row['prev_latitude']) and not pd.isnull(row['prev_longitude'])
    else 0,
    axis=1
)

# Normalize numeric features
scaler = StandardScaler()
numeric_features = ['transaction_value', 'frequency', 'latitude', 'longitude', 'location_deviation']
blockchain_data[numeric_features] = scaler.fit_transform(blockchain_data[numeric_features])

# Step 4: Merge Blockchain and Credit Data
print("Blockchain Data Columns:", blockchain_data.columns)
print("Credit Data Columns:", credit_data.columns)
combined_data = pd.merge(blockchain_data, credit_data, on='user_id', how='inner')
print("Blockchain and behavioral data combined successfully.")

# Step 5: Save Preprocessed Data
combined_data.to_csv('./data/combined_data.csv', index=False)
print("Preprocessed data saved to ./data/combined_data.csv.")
