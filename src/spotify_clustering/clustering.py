# spotify_clustering/clustering.py

import pandas as pd
from sklearn.cluster import KMeans
import os

# Load processed data
def load_processed(file_path):
    """
    Load processed CSV data for clustering.
    """
    df = pd.read_csv(file_path)
    return df

# Apply K-Means clustering
def apply_kmeans(df, n_clusters=5, random_state=42):
    """
    Perform K-Means clustering on the dataset.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    clusters = kmeans.fit_predict(df)  # Compute cluster assignments
    df['cluster'] = clusters           # Add cluster label to dataframe
    return df, kmeans

# Save clustered data
def save_clustered(df, file_name, folder='data/processed'):
    """
    Save the clustered DataFrame to CSV.
    """
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, file_name)
    df.to_csv(path, index=False)
    print(f"Saved clustered data: {path}")

# Full clustering pipeline
def clustering_pipeline(file_path, n_clusters=5):
    df = load_processed(file_path)
    df_clustered, kmeans_model = apply_kmeans(df, n_clusters)
    save_clustered(df_clustered, f'clustered_{n_clusters}.csv')
    return df_clustered, kmeans_model
