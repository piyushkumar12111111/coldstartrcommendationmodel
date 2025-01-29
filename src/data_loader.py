import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import logging

logger = logging.getLogger(__name__)

def load_users_data(file_path):
    """Load and preprocess users data."""
    try:
        df = pd.read_csv(file_path)
        required_columns = ['user_id', 'age', 'gender']
        
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"Missing required columns. Expected: {required_columns}")
        
        # Encode categorical variables
        label_encoder = LabelEncoder()
        df['gender_encoded'] = label_encoder.fit_transform(df['gender'])
        
        # Normalize numerical features
        scaler = StandardScaler()
        df['age_normalized'] = scaler.fit_transform(df[['age']])
            
        return df
    except Exception as e:
        logger.error(f"Error loading users data: {str(e)}")
        raise

def load_movies_data(file_path):
    """Load and preprocess movies data."""
    try:
        df = pd.read_csv(file_path)
        required_columns = ['movie_id', 'title', 'cast', 'genre', 'metadata']
        
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"Missing required columns. Expected: {required_columns}")
        
        # Clean text fields
        text_columns = ['cast', 'genre', 'metadata']
        for col in text_columns:
            df[col] = df[col].fillna('').str.lower()
            
        return df
    except Exception as e:
        logger.error(f"Error loading movies data: {str(e)}")
        raise