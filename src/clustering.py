import pandas as pd
from sklearn.cluster import KMeans

def run_clustering():
    df = pd.read_csv("data/matches.csv")

    wins = df['winner'].value_counts().reset_index()
    wins.columns = ['team', 'wins']

    kmeans = KMeans(n_clusters=3)
    wins['cluster'] = kmeans.fit_predict(wins[['wins']])

    print("\n--- Team Clustering ---")
    print(wins)