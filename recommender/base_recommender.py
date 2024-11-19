 # Base class for all recommenders
# 
import pandas as pd

class BaseRecommender:
    def __init__(self, data):
        self.data = data

    def preprocess_data(self):
        """Common data preprocessing steps."""
        pass
