from tensorflow.keras.models import load_model

# Load the model in HDF5 format
autoencoder = load_model("models/autoencoder_model.h5")

# Re-save the model in the new .keras format
autoencoder.save("models/autoencoder_model.keras")
print("Model successfully saved in .keras format.")
