import pandas as pd
import logging

logger = logging.getLogger(__name__)

def load_users_data(file_path):
    """Load and validate users data."""
    try:
        df = pd.read_csv(file_path)
        required_columns = ['user_id', 'age', 'gender']
        
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"Missing required columns. Expected: {required_columns}")
        
        # Convert gender to numeric
        df['gender_encoded'] = df['gender'].map({'M': 0, 'F': 1})
        
        # Normalize age
        df['age_normalized'] = (df['age'] - df['age'].mean()) / df['age'].std()
            
        return df
    except Exception as e:
        logger.error(f"Error loading users data: {str(e)}")
        raise

def load_movies_data(file_path):
    """Load and validate movies data."""
    try:
        df = pd.read_csv(file_path)
        required_columns = ['movie_id', 'title', 'cast', 'genre', 'metadata']
        
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"Missing required columns. Expected: {required_columns}")
            
        return df
    except Exception as e:
        logger.error(f"Error loading movies data: {str(e)}")
        raise
