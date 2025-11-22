# Spotify Clustering Project

This project implements clustering on Spotify tracks using the K-Means algorithm. The goal is to analyze patterns and relationships between tracks based on their features such as danceability, energy, and tempo. The project is organized to keep data processing, clustering, and visualization cleanly separated.

## Project Structure

spotify-clustering/
│
├── data/
│ ├── raw/ # Original CSV dataset
│ └── processed/ # Cleaned, normalized, and PCA-applied datasets
│
├── src/
│ ├── main.py # Main script to run the project
│ └── spotify_clustering/
│ ├──── init.py
│ ├──── preprocess.py # Data cleaning, normalization, and PCA
│ ├──── visualization.py # Data visualization
│ └──── clustering.py # K-Means clustering
│
└── venv/ # Python virtual environment


## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/tu-usuario/spotify-clustering.git
cd spotify-clustering

### 2. Create virtual environment 
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

### 3. Install dependencies
pip install -r requirements.txt

### 4. Add your dataset
Place your Spotify CSV file in data/raw/.

### 5. Run the project
python src/main.py

Features
- Data preprocessing: cleaning, normalization, PCA reduction
- K-Means clustering
-Interactive visualizations of clusters
-Clean and professional project structure
-Dependencies

All Python dependencies are listed in requirements.txt.