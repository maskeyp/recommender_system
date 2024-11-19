# SVD-based recommender implementation
from .base_recommender import BaseRecommender
from scipy.sparse.linalg import svds
import pandas as pd
import numpy as np

class SVDRecommender(BaseRecommender):
    def __init__(self, data, num_factors=10):
        super().__init__(data)
        self.num_factors = num_factors
        self.user_factors = None
        self.item_factors = None

    def fit(self):
        """Fit the SVD model."""
        ratings_matrix = self.data.pivot(index='user_id', columns='item_id', values='rating').fillna(0)
        mean_rating = ratings_matrix.values.mean()
        ratings_demeaned = ratings_matrix.values - mean_rating

        u, sigma, vt = svds(ratings_demeaned, k=self.num_factors)
        self.user_factors = u
        self.item_factors = vt.T
        self.mean_rating = mean_rating

    def recommend(self, user_id, num_recommendations=5):
        """Generate recommendations for a given user."""
        user_index = self.data.pivot(index='user_id', columns='item_id', values='rating').index.get_loc(user_id)
        predicted_ratings = self.user_factors[user_index, :].dot(self.item_factors.T) + self.mean_rating
        top_items = np.argsort(predicted_ratings)[::-1][:num_recommendations]
        return top_items
