# spotify_clustering/visualization.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set(style="whitegrid")  # Make plots look clean

# Load clustered data
def load_clustered(file_path):
    """
    Load CSV data that already contains cluster labels.
    """
    df = pd.read_csv(file_path)
    return df

# Plot clusters
def plot_clusters(df, x='PC1', y='PC2', cluster_col='cluster', save_fig=False, folder='plots'):
    """
    Plot clusters in 2D using PCA components.
    """
    plt.figure(figsize=(10, 7))
    palette = sns.color_palette("tab10", n_colors=df[cluster_col].nunique())
    
    sns.scatterplot(
        data=df, 
        x=x, 
        y=y, 
        hue=cluster_col, 
        palette=palette, 
        legend='full',
        alpha=0.7
    )
    
    plt.title('K-Means Clustering Visualization')
    plt.xlabel(x)
    plt.ylabel(y)
    plt.legend(title='Cluster')
    
    if save_fig:
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, 'clusters.png')
        plt.savefig(path, bbox_inches='tight')
        print(f"Saved plot: {path}")
    
    plt.show()

# Full visualization pipeline
def visualize_clusters(file_path, x='PC1', y='PC2', cluster_col='cluster', save_fig=False):
    df = load_clustered(file_path)
    plot_clusters(df, x, y, cluster_col, save_fig)
