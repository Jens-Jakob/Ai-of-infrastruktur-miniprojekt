import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import wandb
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score


wandb.init(project='K-means', name='run4')

torch.manual_seed(42)
np.random.seed(42)

data = pd.read_csv('/Miniprojekt/garlic_per_day_data_cleaned.csv')
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


wandb.watch(model, log='all')

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
    wandb.log({'epoch': epoch + 1, 'loss': epoch_loss})
    if (epoch+1) % 10 == 0 or epoch == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

with torch.no_grad():
    _, latent_features = model(features_tensor)

latent_features_np = latent_features.numpy()
num_clusters = 2
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
cluster_assignments = kmeans.fit_predict(latent_features_np)
cluster_centers = kmeans.cluster_centers_


plt.figure(figsize=(10, 6))
for i in range(num_clusters):
    cluster_data = latent_features_np[cluster_assignments == i]
    plt.scatter(cluster_data[:, 0], cluster_data[:, 1], alpha=0.5, label=f'Cluster {i}')
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], s=200, c='black', marker='X', label='Centroids')
plt.xlabel('Latent Feature 1')
plt.ylabel('Latent Feature 2')
plt.title('Clustering of Pizza Orders using Autoencoder and K-Means')
plt.legend()
#plt.savefig('clustering.png')
plt.show()
wandb.log({'Clustering Plot': wandb.Image('clustering.png')})


score = silhouette_score(latent_features_np, cluster_assignments)
print(f'Silhouette Score for {num_clusters} clusters: {score:.4f}')
wandb.log({'Silhouette Score': score})


data = data.reset_index(drop=True)
data['Cluster'] = cluster_assignments

cluster_summary = data.groupby('Cluster')[
    ['order_hour', 'order_dayofweek', 'quantity', 'unit_price', 'total_price']].agg(
    ['mean', 'median', 'std', 'min', 'max', 'count'])
print(cluster_summary)

wandb.log({'Cluster Summary': cluster_summary})

#Visualization of clusters
numerical_features = ['order_hour', 'order_dayofweek', 'quantity', 'unit_price', 'total_price']

for feature in numerical_features:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Cluster', y=feature, data=data)
    plt.title(f'{feature.capitalize()} Distribution by Cluster')
    plt.xlabel('Cluster')
    plt.ylabel(feature.capitalize())
    #plt.savefig(f'{feature}_distribution.png')
    plt.show()

    wandb.log({f'{feature.capitalize()} Distribution': wandb.Image(f'{feature}_distribution.png')})

if 'pizza_category' in data.columns:
    category_counts = data.groupby(['Cluster', 'pizza_category']).size().unstack(fill_value=0)
    print(category_counts)
    category_counts.plot(kind='bar', stacked=True, figsize=(10, 6))
    plt.title('Pizza Category Distribution by Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Count')
    plt.legend(title='Pizza Category')
    #plt.savefig('pizza_category_distribution.png')
    plt.show()
    wandb.log({'Pizza Category Distribution': wandb.Image('pizza_category_distribution.png')})

if 'pizza_size' in data.columns:
    size_counts = data.groupby(['Cluster', 'pizza_size']).size().unstack(fill_value=0)
    print(size_counts)
    size_counts.plot(kind='bar', stacked=True, figsize=(10, 6))
    plt.title('Pizza Size Distribution by Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Count')
    plt.legend(title='Pizza Size')
    #plt.savefig('pizza_size_distribution.png')
    plt.show()
    wandb.log({'Pizza Size Distribution': wandb.Image('pizza_size_distribution.png')})

for cluster in data['Cluster'].unique():
    cluster_data = data[data['Cluster'] == cluster]
    pivot_table = cluster_data.pivot_table(values='order_id', index='order_dayofweek', columns='order_hour',
                                           aggfunc='count', fill_value=0)
    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot_table, cmap='YlGnBu', annot=True, fmt='d')
    plt.title(f'Order Count Heatmap for Cluster {cluster}')
    plt.xlabel('Order Hour')
    plt.ylabel('Day of Week')
    #plt.savefig(f'heatmap_cluster_{cluster}.png')
    plt.show()
    wandb.log({f'Heatmap Cluster {cluster}': wandb.Image(f'heatmap_cluster_{cluster}.png')})

wandb.finish()
