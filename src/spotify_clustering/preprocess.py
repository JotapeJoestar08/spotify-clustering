# spotify_clustering/preprocess.py

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

# Load the dataset
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# Clean the dataset
def clean_data(df):
    """
    Remove duplicates, missing values, and unnecessary columns.
    """
    df = df.drop_duplicates()  # Remove duplicate rows
    df = df.dropna()           # Remove rows with missing values
    
    # Select only relevant features for clustering
    features = [
        'popularity', 'danceability', 'energy', 'loudness',
        'speechiness', 'acousticness', 'instrumentalness',
        'liveness', 'valence', 'tempo'
    ]
    df_features = df[features]
    return df_features

# Normalize the dataset
def normalize_data(df):
    """
    Scale features to have mean=0 and variance=1.
    """
    scaler = StandardScaler()   # Standardization
    df_scaled = scaler.fit_transform(df)
    df_scaled = pd.DataFrame(df_scaled, columns=df.columns)
    return df_scaled

# Apply PCA
def apply_pca(df, n_components=2):
    """
    Reduce dimensionality using PCA.
    """
    pca = PCA(n_components=n_components)
    df_pca = pca.fit_transform(df)
    df_pca = pd.DataFrame(df_pca, columns=[f'PC{i+1}' for i in range(n_components)])
    return df_pca

# Save processed data
def save_processed(df, file_name, folder='data/processed'):
    """
    Save the processed DataFrame to CSV.
    """
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, file_name)
    df.to_csv(path, index=False)
    print(f"Saved: {path}")

# Full preprocessing pipeline
def preprocess_pipeline(file_path):
    df_raw = load_data(file_path)
    df_clean = clean_data(df_raw)
    save_processed(df_clean, 'spotify_clean.csv')

    df_normalized = normalize_data(df_clean)
    save_processed(df_normalized, 'spotify_normalized.csv')

    df_pca = apply_pca(df_normalized)
    save_processed(df_pca, 'spotify_pca.csv')

    return df_raw, df_clean, df_normalized, df_pca
