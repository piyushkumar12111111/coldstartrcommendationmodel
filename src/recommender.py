import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from textblob import TextBlob
import logging
import joblib
from collections import defaultdict

logger = logging.getLogger(__name__)

class MLRecommender:
    def __init__(self, movies_df, users_df):
        """Initialize ML-based recommender system."""
        try:
            self.movies_df = movies_df.copy()
            self.users_df = users_df.copy()
            
            # Initialize vectorizers
            self.text_vectorizer = TfidfVectorizer(
                stop_words='english',
                ngram_range=(1, 3),
                max_features=1000,
                strip_accents='unicode'
            )
            
            # Initialize dimensionality reduction
            self.svd = TruncatedSVD(n_components=50, random_state=42)
            
            # Initialize classifier
            self.classifier = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )
            
            # Initialize feature stores
            self.movie_features = None
            self.user_features = None
            self.genre_embeddings = {}
            
            logger.info("Recommender system initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing recommender: {str(e)}")
            raise

    def _extract_text_features(self, text_series):
        """Extract TF-IDF features from text."""
        features = self.text_vectorizer.fit_transform(text_series)
        return self.svd.fit_transform(features)

    def _extract_sentiment_features(self, text):
        """Extract sentiment features from text using TextBlob."""
        blob = TextBlob(text)
        return {
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity
        }

    def _create_movie_features(self):
        """Create comprehensive movie feature vectors."""
        # Combine all text features
        self.movies_df['combined_text'] = (
            self.movies_df['cast'] + ' ' + 
            self.movies_df['genre'] + ' ' + 
            self.movies_df['metadata']
        )
        
        # Extract text features
        text_features = self._extract_text_features(self.movies_df['combined_text'])
        
        # Extract sentiment features
        sentiment_features = np.array([
            list(self._extract_sentiment_features(text).values())
            for text in self.movies_df['combined_text']
        ])
        
        # Combine features
        self.movie_features = np.hstack([text_features, sentiment_features])
        
        # Create genre embeddings
        for genre in self.movies_df['genre'].unique():
            genre_movies = text_features[self.movies_df['genre'] == genre]
            if len(genre_movies) > 0:
                self.genre_embeddings[genre] = np.mean(genre_movies, axis=0)

    def _create_user_features(self):
        """Create user feature vectors."""
        # Combine demographic features
        self.user_features = np.column_stack([
            self.users_df['age_normalized'],
            self.users_df['gender_encoded']
        ])

    def train(self):
        """Train the recommendation model."""
        try:
            logger.info("Starting model training...")
            
            # Create feature vectors
            self._create_movie_features()
            self._create_user_features()
            
            # Train classifier on synthetic preferences
            # This simulates user-movie interactions based on demographics
            X_train = []
            y_train = []
            
            for idx, user in self.users_df.iterrows():
                user_age = user['age']
                user_gender = user['gender']
                
                for movie_idx, movie in self.movies_df.iterrows():
                    features = np.concatenate([
                        self.user_features[idx],
                        self.movie_features[movie_idx]
                    ])
                    
                    # Generate synthetic preference (1 = like, 0 = dislike)
                    # This is a placeholder for real user interaction data
                    preference = self._generate_synthetic_preference(
                        user_age, user_gender, 
                        movie['genre'], movie['metadata']
                    )
                    
                    X_train.append(features)
                    y_train.append(preference)
            
            # Train the classifier
            self.classifier.fit(X_train, y_train)
            
            logger.info("Model training completed successfully")
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise

    def _generate_synthetic_preference(self, age, gender, genre, metadata):
        """Generate synthetic user preference based on demographics and content."""
        score = 0.5  # Base score
        
        # Add random variation
        score += np.random.normal(0, 0.1)
        
        # Adjust based on text sentiment
        sentiment = TextBlob(metadata).sentiment.polarity
        score += sentiment * 0.2
        
        return 1 if score > 0.5 else 0

    def get_recommendations_for_user(self, user_id, num_recommendations=3):
        """Get personalized movie recommendations for a user."""
        try:
            if user_id not in self.users_df['user_id'].values:
                logger.warning(f"User {user_id} not found in the database")
                return None

            user_idx = self.users_df[self.users_df['user_id'] == user_id].index[0]
            user_features = self.user_features[user_idx]
            
            # Calculate preference scores for all movies
            movie_scores = []
            for idx, movie in self.movies_df.iterrows():
                # Combine user and movie features
                features = np.concatenate([
                    user_features,
                    self.movie_features[idx]
                ])
                
                # Get prediction probability
                score = self.classifier.predict_proba([features])[0][1]
                
                # Generate explanation
                explanation = self._generate_recommendation_explanation(
                    movie, score, user_idx
                )
                
                movie_scores.append((
                    movie['title'],
                    score,
                    explanation
                ))
            
            # Sort by score and return top recommendations
            recommendations = sorted(movie_scores, key=lambda x: x[1], reverse=True)
            return recommendations[:num_recommendations]
            
        except Exception as e:
            logger.error(f"Error getting user recommendations: {str(e)}")
            return None

    def _generate_recommendation_explanation(self, movie, score, user_idx):
        """Generate human-readable explanation for recommendation."""
        user = self.users_df.iloc[user_idx]
        explanations = []
        
        # Genre-based explanation
        genre = movie['genre']
        if genre in self.genre_embeddings:
            genre_similarity = cosine_similarity(
                [self.genre_embeddings[genre]], 
                [self.movie_features[movie.name][:-2]]  # Exclude sentiment features
            )[0][0]
            if genre_similarity > 0.5:
                explanations.append(f"Strong match with {genre} preferences")
        
        # Sentiment-based explanation
        sentiment = TextBlob(movie['metadata']).sentiment
        if sentiment.polarity > 0:
            explanations.append("Positive content sentiment")
        
        # Age-based explanation
        age = user['age']
        if 15 <= age <= 25 and 'fast-paced' in movie['metadata'].lower():
            explanations.append("Matches age group preferences")
        
        return " | ".join(explanations) if explanations else "Based on overall profile match"
