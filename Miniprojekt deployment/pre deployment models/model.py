import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib


torch.manual_seed(42)
np.random.seed(42)

data = pd.read_csv('/Users/jens-jakobskotingerslev/PycharmProjects/AI infrastruktur/Miniprojekt/garlic_per_day_data_cleaned.csv')
data['order_datetime'] = pd.to_datetime(data['order_date'] + ' ' + data['order_time'])
data['order_hour'] = data['order_datetime'].dt.hour
data['order_dayofweek'] = data['order_datetime'].dt.dayofweek
data['order_month'] = data['order_datetime'].dt.month

features = data[['order_hour', 'order_dayofweek', 'quantity', 'unit_price', 'total_price']]
features = features.dropna()

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
features_tensor = torch.tensor(scaled_features, dtype=torch.float32)

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

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
num_epochs = 30
batch_size = 256
dataset = torch.utils.data.TensorDataset(features_tensor)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


for epoch in range(num_epochs):
    epoch_loss = 0.0
    for data_batch in dataloader:
        inputs = data_batch[0]
        optimizer.zero_grad()
        outputs, _ = model(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * inputs.size(0)
    epoch_loss /= len(dataloader.dataset)
    if (epoch+1) % 10 == 0 or epoch == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

with torch.no_grad():
    _, latent_features = model(features_tensor)

latent_features_np = latent_features.numpy()
num_clusters = 2
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
cluster_assignments = kmeans.fit_predict(latent_features_np)
cluster_centers = kmeans.cluster_centers_

torch.save(model.state_dict(), 'app/models/autoencoder.pth')
joblib.dump(kmeans, 'app/models/kmeans.pkl')
joblib.dump(scaler, 'app/models/scaler.pkl')

score = silhouette_score(latent_features_np, cluster_assignments)
print(f'Silhouette Score for {num_clusters} clusters: {score:.4f}')

