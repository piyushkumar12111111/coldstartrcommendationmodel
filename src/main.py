import logging
from data_loader import load_users_data, load_movies_data
from recommender import UserRecommender

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
        recommender = UserRecommender(movies_df, users_df)
        
        # Get recommendations for user 1
        test_user_id = 2
        recommendations = recommender.get_recommendations_for_user(test_user_id)
        
        if recommendations:
            print(f"\nTop movie recommendations for User {test_user_id}:")
            for idx, (movie, score, genres) in enumerate(recommendations, 1):
                print(f"{idx}. {movie}")
                print(f"   Genres: {genres}")
                print(f"   Relevance Score: {score:.3f}\n")
        else:
            print(f"No recommendations found for User {test_user_id}")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()