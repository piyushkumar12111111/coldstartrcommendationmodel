import numpy as np
from sklearn.preprocessing import MinMaxScaler
import logging

logger = logging.getLogger(__name__)

class UserRecommender:
    def __init__(self, movies_df, users_df):
        """Initialize the user-based recommender system."""
        try:
            self.movies_df = movies_df.copy()
            self.users_df = users_df.copy()
            
            # Process user features and create user profiles
            self._create_user_profiles()
            
            logger.info("Recommender system initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing recommender: {str(e)}")
            raise

    def _create_user_profiles(self):
        """Create user profiles based on demographics and preferences."""
        self.user_profiles = {}
        
        for _, user in self.users_df.iterrows():
            user_id = user['user_id']
            age = user['age']
            gender = user['gender']
            
            # Create base profile
            profile = {
                'age': age,
                'gender': gender,
                'genre_preferences': self._calculate_genre_preferences(age, gender),
                'content_preferences': self._calculate_content_preferences(age, gender)
            }
            
            self.user_profiles[user_id] = profile

    def _calculate_genre_preferences(self, age, gender):
        """Calculate genre preferences based on user demographics."""
        preferences = {
            'Action': 0.5,
            'Comedy': 0.5,
            'Drama': 0.5
        }
        
        # Age-based adjustments
        if age < 25:
            preferences['Action'] += 0.3
            preferences['Comedy'] += 0.2
            preferences['Drama'] -= 0.1
        elif 25 <= age <= 35:
            preferences['Comedy'] += 0.2
            preferences['Drama'] += 0.1
        else:
            preferences['Drama'] += 0.3
            preferences['Action'] -= 0.1
        
        # Gender-based adjustments
        if gender == 'M':
            preferences['Action'] += 0.2
            preferences['Comedy'] += 0.1
        else:
            preferences['Drama'] += 0.2
            preferences['Comedy'] += 0.2
        
        return preferences

    def _calculate_content_preferences(self, age, gender):
        """Calculate content preferences based on user demographics."""
        preferences = {
            'fast-paced': 0.5,
            'emotional': 0.5,
            'light-hearted': 0.5,
            'thrilling': 0.5
        }
         
        # Age-based adjustments
        if age < 25:
            preferences['fast-paced'] += 0.3
            preferences['thrilling'] += 0.2
        elif age > 35:
            preferences['emotional'] += 0.3
            preferences['light-hearted'] += 0.2
        
        # Gender-based adjustments
        if gender == 'M':
            preferences['thrilling'] += 0.2
        else:
            preferences['emotional'] += 0.2
            preferences['light-hearted'] += 0.1
        
        return preferences

    def _calculate_movie_score(self, movie, user_profile):
        """Calculate movie score based on user profile."""
        score = 0
        
        # Genre matching
        genre = movie['genre']
        genre_prefs = user_profile['genre_preferences']
        if genre in genre_prefs:
            score += genre_prefs[genre] * 0.6  # Genre weight
        
        # Content matching
        metadata = movie['metadata'].lower()
        content_prefs = user_profile['content_preferences']
        
        for keyword, weight in content_prefs.items():
            if keyword in metadata:
                score += weight * 0.4  # Metadata weight
        
        return score

    def get_recommendations_for_user(self, user_id, num_recommendations=3):
        """Get personalized movie recommendations for a user."""
        try:
            if user_id not in self.user_profiles:
                logger.warning(f"User {user_id} not found in the database")
                return None

            user_profile = self.user_profiles[user_id]
            
            # Calculate scores for all movies
            movie_scores = []
            for _, movie in self.movies_df.iterrows():
                score = self._calculate_movie_score(movie, user_profile)
                movie_scores.append((
                    movie['title'],
                    score,
                    movie['genre']
                ))
            
            # Sort by score and return top recommendations
            recommendations = sorted(movie_scores, key=lambda x: x[1], reverse=True)
            return recommendations[:num_recommendations]
            
        except Exception as e:
            logger.error(f"Error getting user recommendations: {str(e)}")
            return None