import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Step 1: Load Preprocessed Data
data = pd.read_csv('./data/combined_data.csv')

# Select relevant features for training
features = data[["transaction_value", "frequency", "latitude", "longitude", "location_deviation"]]
X = features.values

# Step 2: Split Data
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

# Step 3: Build Autoencoder Model
input_dim = X_train.shape[1]
encoding_dim = 3  # Number of dimensions in the bottleneck layer

# Define the autoencoder architecture
input_layer = tf.keras.layers.Input(shape=(input_dim,))
encoder = tf.keras.layers.Dense(encoding_dim, activation="relu")(input_layer)
decoder = tf.keras.layers.Dense(input_dim, activation="sigmoid")(encoder)

autoencoder = tf.keras.models.Model(inputs=input_layer, outputs=decoder)
autoencoder.compile(optimizer='adam', loss='mse')
print("Autoencoder model built successfully.")

# Step 4: Train the Model
history = autoencoder.fit(
    X_train, X_train,
    epochs=50,
    batch_size=32,
    shuffle=True,
    validation_data=(X_test, X_test)
)
print("Model training complete.")

# Step 5: Save the Model
autoencoder.save('./models/autoencoder_model.keras')
print("Model saved to ./models/autoencoder_model.keras")

# Step 6: Calculate and Save Anomaly Threshold
reconstructed = autoencoder.predict(X_test)
mse = np.mean(np.power(X_test - reconstructed, 2), axis=1)
threshold = np.percentile(mse, 95)  # Set threshold at the 95th percentile of reconstruction errors
print(f"Anomaly threshold: {threshold}")

# Save the threshold for future use
with open('./models/threshold.txt', 'w') as f:
    f.write(str(threshold))
