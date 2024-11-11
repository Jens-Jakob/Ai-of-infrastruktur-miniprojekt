from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn
import joblib
import numpy as np

app = FastAPI()

def read_root():
    return {"message": "Welcome to the Pizza Cluster API"}

class InputData(BaseModel):
    order_hour: float
    order_dayofweek: float
    quantity: float
    unit_price: float
    total_price: float

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=2):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 10),
            nn.ReLU(True),
            nn.Linear(10, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 10),
            nn.ReLU(True),
            nn.Linear(10, input_dim),
            nn.Tanh()
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent


input_dim = 5
latent_dim = 2
model = Autoencoder(input_dim, latent_dim)
model.load_state_dict(torch.load('models/autoencoder.pth'))
model.eval()

kmeans = joblib.load('models/kmeans.pkl')
scaler = joblib.load('models/scaler.pkl')

@app.post('/predict')
def predict_cluster(data: InputData):
    input_array = np.array([[data.order_hour, data.order_dayofweek, data.quantity, data.unit_price, data.total_price]])
    scaled_input = scaler.transform(input_array)
    input_tensor = torch.tensor(scaled_input, dtype=torch.float32)
    with torch.no_grad():
        _, latent_features = model(input_tensor)
    latent_features_np = latent_features.numpy()
    cluster = kmeans.predict(latent_features_np)
    return {'cluster': int(cluster[0])}

# docker run -p 8000:8000 pizza-cluster-app2