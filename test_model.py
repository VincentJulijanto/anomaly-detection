import numpy as np
import pandas as pd
import tensorflow as tf

# Step 1: Load the Updated Model
model = tf.keras.models.load_model('./models/autoencoder_model.keras')
print("Updated model loaded successfully.")

# Step 2: Load Test Data
test_data = pd.read_csv('./data/combined_data.csv')
test_features = test_data.drop(columns=['user_id', 'wallet_id'], errors='ignore').values

# Step 3: Reconstruct Data and Calculate Reconstruction Error
reconstructed = model.predict(test_features)
mse = np.mean(np.power(test_features - reconstructed, 2), axis=1)
print(f"Reconstruction errors calculated: {mse[:5]}")  # Print first 5 errors

# Step 4: Detect Anomalies
threshold = 753671588.9395875  # Replace with your updated threshold
anomalies = mse > threshold
test_data['is_anomaly'] = anomalies
print(f"Anomalies detected: {test_data['is_anomaly'].sum()}")

# Step 5: Save Results
test_data.to_csv('./data/anomaly_results.csv', index=False)
print("Anomaly results saved to ./data/anomaly_results.csv.")
