# src/main.py

from spotify_clustering import preprocess, clustering, visualization

def main():
    # Step 1: Preprocess the data
    raw_file = 'data/raw/spotify.csv'
    print("Starting preprocessing...")
    df_raw, df_clean, df_normalized, df_pca = preprocess.preprocess_pipeline(raw_file)
    print("Preprocessing done!\n")
    
    # Step 2: Apply K-Means clustering
    pca_file = 'data/processed/spotify_pca.csv'
    n_clusters = 5
    print(f"Clustering with {n_clusters} clusters...")
    df_clustered, kmeans_model = clustering.clustering_pipeline(pca_file, n_clusters=n_clusters)
    print("Clustering done!\n")
    
    # Step 3: Visualize the clusters
    clustered_file = f'data/processed/clustered_{n_clusters}.csv'
    print("Visualizing clusters...")
    visualization.visualize_clusters(clustered_file, x='PC1', y='PC2', save_fig=True)
    print("Visualization done!")

if __name__ == "__main__":
    main()
