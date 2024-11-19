# Main Entry point of the project 
from utils.data_loader import load_data
from recommender.knn_recommender import KNNRecommender
from recommender.svd_recommender import SVDRecommender

def main():
    data = load_data('data/movie.csv')

    # Switch between KNN and SVD
    recommender = SVDRecommender(data)  # Change to KNNRecommender(data) for KNN
    recommender.fit()
    
    # Example recommendation for user_id = 1
    recommendations = recommender.recommend(user_id=1)
    print("Recommended items:", recommendations)

if __name__ == '__main__':
    main()
