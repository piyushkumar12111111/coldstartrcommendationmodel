import logging
from data_loader import load_users_data, load_movies_data
from recommender import HybridRecommender

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    try:
        # File paths
        users_file = '../data/users.csv'
        movies_file = '../data/movies.csv'

        # Load data
        users_df = load_users_data(users_file)
        movies_df = load_movies_data(movies_file)

        logger.info(f"Loaded {len(movies_df)} movies and {len(users_df)} users")

        # Initialize recommender
        recommender = HybridRecommender(movies_df, users_df)
        
        # Test recommendations for a specific user and movie
        test_user_id = 1
        test_movie = 'Movie E'
        
        # Get personalized recommendations
        user_recommendations = recommender.get_recommendations_for_user(test_user_id)
        print(f"\nTop 3 recommendations for User {test_user_id}:")
        if user_recommendations:
            for idx, (movie, score) in enumerate(user_recommendations, 1):
                print(f"{idx}. {movie} (relevance score: {score:.3f})")
        
        # Get similar movie recommendations
        similar_movies = recommender.get_similar_movies(test_movie)
        print(f"\nSimilar movies to '{test_movie}':")
        if similar_movies:
            for idx, (movie, score) in enumerate(similar_movies, 1):
                print(f"{idx}. {movie} (similarity score: {score:.3f})")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()