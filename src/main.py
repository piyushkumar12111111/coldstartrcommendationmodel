import logging
from data_loader import load_users_data, load_movies_data
from recommender import MLRecommender

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

        # Initialize and train recommender
        recommender = MLRecommender(movies_df, users_df)
        recommender.train()
        
        # Get recommendations for user 1
        test_user_id = 1
        recommendations = recommender.get_recommendations_for_user(test_user_id)
        
        if recommendations:
            print(f"\nTop movie recommendations for User {test_user_id}:")
            for idx, (movie, score, explanation) in enumerate(recommendations, 1):
                print(f"\n{idx}. {movie}")
                print(f"   Confidence Score: {score:.3f}")
                print(f"   Recommendation Basis: {explanation}")
        else:
            print(f"No recommendations found for User {test_user_id}")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()