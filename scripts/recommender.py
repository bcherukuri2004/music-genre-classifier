"""
KNN and cosine similarity based track recommender over the extracted
feature space. Given a track's feature vector, finds the most similar
tracks in the dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path(__file__).parent.parent / "data" / "features_final.csv"


class TrackRecommender:
    def __init__(self, features_path=DATA_PATH):
        self.df = pd.read_csv(features_path)
        self.feature_cols = [c for c in self.df.columns if c not in ("track_id", "genre")]

        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.df[self.feature_cols].values)

        self.knn = NearestNeighbors(n_neighbors=11, metric="euclidean")
        self.knn.fit(self.X_scaled)

    def recommend_by_track_id(self, track_id, n=5, method="knn"):
        idx = self.df.index[self.df["track_id"] == track_id]
        if len(idx) == 0:
            raise ValueError(f"track_id {track_id} not found in dataset")
        idx = idx[0]
        return self._recommend_from_vector(self.X_scaled[idx], n=n, method=method, exclude_idx=idx)

    def recommend_by_features(self, feature_dict, n=5, method="knn"):
        vec = np.array([[feature_dict[c] for c in self.feature_cols]])
        vec_scaled = self.scaler.transform(vec)[0]
        return self._recommend_from_vector(vec_scaled, n=n, method=method, exclude_idx=None)

    def _recommend_from_vector(self, vector, n=5, method="knn", exclude_idx=None):
        if method == "knn":
            distances, indices = self.knn.kneighbors([vector], n_neighbors=n + 1)
            indices = indices[0]
            distances = distances[0]
            results = []
            for dist, i in zip(distances, indices):
                if exclude_idx is not None and i == exclude_idx:
                    continue
                results.append({
                    "track_id": self.df.iloc[i]["track_id"],
                    "genre": self.df.iloc[i]["genre"],
                    "distance": float(dist),
                })
            return results[:n]

        elif method == "cosine":
            sims = cosine_similarity([vector], self.X_scaled)[0]
            if exclude_idx is not None:
                sims[exclude_idx] = -1
            top_indices = np.argsort(sims)[::-1][:n]
            results = []
            for i in top_indices:
                results.append({
                    "track_id": self.df.iloc[i]["track_id"],
                    "genre": self.df.iloc[i]["genre"],
                    "similarity": float(sims[i]),
                })
            return results

        else:
            raise ValueError("method must be 'knn' or 'cosine'")
