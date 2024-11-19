# KNN-based recommender implementation

from .base_recommender import BaseRecommender
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np

class KNNRecommender(BaseRecommender):
    def __init__(self, data, k=5):
        super().__init__(data)
        self.k = k
        self.model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=self.k)

    def fit(self):
        """Fit the KNN model."""
        ratings_matrix = self.data.pivot(index='user_id', columns='item_id', values='rating').fillna(0)
        self.model.fit(ratings_matrix)
        self.ratings_matrix = ratings_matrix

    def recommend(self, user_id, num_recommendations=5):
        """Generate recommendations for a given user."""
        user_index = self.ratings_matrix.index.get_loc(user_id)
        distances, indices = self.model.kneighbors(
            self.ratings_matrix.iloc[user_index, :].values.reshape(1, -1), n_neighbors=self.k + 1
        )
        recommended_items = []
        for i in range(1, len(distances.flatten())):
            recommended_items.append(self.ratings_matrix.index[indices.flatten()[i]])
        return recommended_items[:num_recommendations]
