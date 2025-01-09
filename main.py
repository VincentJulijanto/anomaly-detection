from fastapi import FastAPI, HTTPException
import tensorflow as tf
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# Load the trained model
try:
    autoencoder = tf.keras.models.load_model("models/autoencoder_model.keras")
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    exit()

# Load the anomaly threshold
try:
    with open('./models/threshold.txt', 'r') as f:
        threshold = float(f.read())
    logger.info(f"Anomaly threshold loaded: {threshold}")
except Exception as e:
    logger.error(f"Error loading threshold: {e}")
    exit()

# Initialize FastAPI
app = FastAPI()

@app.post("/predict")
async def predict(data: dict):
    """
    Predict whether a transaction is an anomaly.
    Expects input in the format:
    {
        "transaction_value": <value>,
        "frequency": <value>,
        "latitude": <value>,
        "longitude": <value>,
        "location_deviation": <value>
    }
    """
    try:
        # Validate input data
        required_keys = ["transaction_value", "frequency", "latitude", "longitude", "location_deviation"]
        if not all(key in data for key in required_keys):
            logger.warning(f"Invalid input data: {data}")
            raise HTTPException(status_code=400, detail="Missing one or more required keys.")

        # Normalize input data
        transaction_value = (data["transaction_value"] - 10) / (5000 - 10)
        frequency = (data["frequency"] - 1) / (10 - 1)
        latitude = (data["latitude"] - (-90)) / (90 - (-90))
        longitude = (data["longitude"] - (-180)) / (180 - (-180))
        location_deviation = data["location_deviation"]  # Assume already scaled

        # Combine features into a single array
        normalized_data = np.array([[transaction_value, frequency, latitude, longitude, location_deviation]])
        logger.info(f"Normalized data: {normalized_data}")

        # Predict reconstruction error
        reconstruction = autoencoder.predict(normalized_data)
        error = np.mean(np.power(normalized_data - reconstruction, 2))
        logger.info(f"Reconstruction error: {error}")

        # Detect anomaly
        is_anomaly = int(error > threshold)
        logger.info(f"Prediction made: Input={data}, Anomaly={is_anomaly}, Error={error}")

        # Return result
        return {"anomaly": is_anomaly, "error": error}

    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return {"detail": "An error occurred during prediction. Please check server logs."}
