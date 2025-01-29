import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import logging

logger = logging.getLogger(__name__)

class HybridRecommender:
    def __init__(self, movies_df, users_df):
        """Initialize the hybrid recommender system."""
        try:
            self.movies_df = movies_df.copy()
            self.users_df = users_df.copy()
            
            # Process movie features
            self._process_movie_features()
            
            # Process user features
            self._process_user_features()
            
            logger.info("Recommender system initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing recommender: {str(e)}")
            raise

    def _process_movie_features(self):
        """Process and vectorize movie features."""
        # Combine features with weights
        self.movies_df['features'] = (
            self.movies_df['cast'].fillna('') + ' ' + 
            self.movies_df['genre'].fillna('') * 2 + ' ' + 
            self.movies_df['metadata'].fillna('')
        )
        
        # Initialize and fit TF-IDF vectorizer for movies
        self.movie_tfidf = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_features=5000,
            strip_accents='unicode'
        )
        
        self.movie_matrix = self.movie_tfidf.fit_transform(self.movies_df['features'])
        self.movie_sim_matrix = cosine_similarity(self.movie_matrix, self.movie_matrix)

    def _process_user_features(self):
        """Process and normalize user features."""
        # Create user feature matrix
        self.user_features = self.users_df[['age_normalized', 'gender_encoded']].values
        
        # Calculate user similarity matrix
        self.user_sim_matrix = cosine_similarity(self.user_features)
        
        # Create user preferences based on demographics
        self._create_user_preferences()

    def _create_user_preferences(self):
        """Create synthetic user preferences based on demographics."""
        # Simple rule-based preference generation
        self.user_preferences = {}
        
        for _, user in self.users_df.iterrows():
            user_id = user['user_id']
            age = user['age']
            gender = user['gender']
            
            # Initialize preference scores for each genre
            genre_scores = {
                'Action': 0.5,
                'Comedy': 0.5,
                'Drama': 0.5
            }
            
            # Adjust preferences based on demographics
            if age < 25:
                genre_scores['Action'] += 0.2
                genre_scores['Comedy'] += 0.1
            elif age > 35:
                genre_scores['Drama'] += 0.2
                
            if gender == 'M':
                genre_scores['Action'] += 0.1
            else:
                genre_scores['Drama'] += 0.1
                
            self.user_preferences[user_id] = genre_scores

    def get_similar_movies(self, title, num_recommendations=3):
        """Get similar movies based on content."""
        try:
            if title not in self.movies_df['title'].values:
                logger.warning(f"Movie '{title}' not found in the database")
                return None

            idx = self.movies_df[self.movies_df['title'] == title].index[0]
            sim_scores = list(enumerate(self.movie_sim_matrix[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:num_recommendations+1]
            
            recommendations = [
                (self.movies_df['title'].iloc[i[0]], i[1]) 
                for i in sim_scores
            ]
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting similar movies: {str(e)}")
            return None

    def get_recommendations_for_user(self, user_id, num_recommendations=3):
        """Get personalized movie recommendations for a user."""
        try:
            if user_id not in self.users_df['user_id'].values:
                logger.warning(f"User {user_id} not found in the database")
                return None

            # Get user preferences
            user_prefs = self.user_preferences.get(user_id, {})
            
            # Calculate movie scores based on user preferences
            movie_scores = []
            for idx, movie in self.movies_df.iterrows():
                score = 0
                genre = movie['genre']
                
                # Base score from genre preferences
                score += user_prefs.get(genre, 0.5)
                
                # Add score from content-based similarity
                similar_movies = self.movie_sim_matrix[idx]
                score += np.mean(similar_movies) * 0.3
                
                movie_scores.append((movie['title'], score))
            
            # Sort and return top recommendations
            recommendations = sorted(movie_scores, key=lambda x: x[1], reverse=True)
            return recommendations[:num_recommendations]
            
        except Exception as e:
            logger.error(f"Error getting user recommendations: {str(e)}")
            return None