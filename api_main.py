from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
import logging
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Transaction(BaseModel):
    transaction_value: float
    frequency: int
    latitude: float
    longitude: float
    location_deviation: float


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
try:
    autoencoder = tf.keras.models.load_model("models/autoencoder_model.keras")
    with open('./models/threshold.txt', 'r') as f:
        threshold = float(f.read())
except Exception as e:
    logger.error(f"Error loading model/threshold: {e}")
    # For demo, create simple model
    input_dim = 5
    encoding_dim = 3
    input_layer = tf.keras.layers.Input(shape=(input_dim,))
    encoder = tf.keras.layers.Dense(
        encoding_dim, activation="relu")(input_layer)
    decoder = tf.keras.layers.Dense(input_dim, activation="sigmoid")(encoder)
    autoencoder = tf.keras.models.Model(inputs=input_layer, outputs=decoder)
    autoencoder.compile(optimizer='adam', loss='mse')
    threshold = 0.1


@app.post("/predict")
async def predict(transaction: Transaction):
    try:
        # Normalize
        norm_data = {
            "transaction_value": (transaction.transaction_value - 10) / (5000 - 10),
            "frequency": (transaction.frequency - 1) / (10 - 1),
            "latitude": (transaction.latitude - (-90)) / (90 - (-90)),
            "longitude": (transaction.longitude - (-180)) / (180 - (-180)),
            "location_deviation": transaction.location_deviation
        }

        input_data = np.array([[
            norm_data["transaction_value"],
            norm_data["frequency"],
            norm_data["latitude"],
            norm_data["longitude"],
            norm_data["location_deviation"]
        ]])

        # Predict
        reconstruction = autoencoder.predict(input_data, verbose=0)
        error = np.mean(np.power(input_data - reconstruction, 2))
        is_anomaly = int(error > threshold)

        return {
            "anomaly": is_anomaly,
            "error": float(error),
            "threshold": float(threshold)
        }

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/test")
async def test():
    sample_tx = Transaction(
        transaction_value=4000.0,
        frequency=5,
        latitude=1.3521,
        longitude=103.8198,
        location_deviation=0.1
    )
    return await predict(sample_tx)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
