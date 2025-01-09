import pandas as pd
import numpy as np
from faker import Faker

fake = Faker()

def generate_blockchain_data(n=1000):
    data = []
    for _ in range(n):
        data.append({
            "wallet_id": fake.uuid4(),
            "transaction_value": np.random.uniform(10, 5000),  # Random transaction value between 10 and 5000
            "frequency": np.random.randint(1, 10),            # Frequency of transactions (1 to 10)
            "geolocation": str(fake.latitude()) + "," + str(fake.longitude())  # Convert to strings
        })
    return pd.DataFrame(data)

if __name__ == "__main__":
    blockchain_data = generate_blockchain_data()
    blockchain_data.to_csv("data/blockchain_data.csv", index=False)
    print("Synthetic blockchain data saved to data/blockchain_data.csv")
