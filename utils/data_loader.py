
import pandas as pd

def load_data():
    """Load movies and credits datasets from the data folder."""
    movies = pd.read_csv('data/movie.csv')
    credits = pd.read_csv('data/credits.csv')
    return movies, credits
