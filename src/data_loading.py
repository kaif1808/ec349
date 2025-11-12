import pandas as pd
import os
from typing import List


def load_business_data(filepath: str) -> pd.DataFrame:
    """
    Load business data from CSV file with appropriate dtypes.

    Args:
        filepath: Path to the business CSV file

    Returns:
        DataFrame containing business data

    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If required columns are missing
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Business data file not found: {filepath}")

    # Define dtypes for business data
    dtypes = {
        'business_id': str,
        'name': str,
        'address': str,
        'city': str,
        'state': str,
        'postal_code': str,
        'latitude': float,
        'longitude': float,
        'stars': float,
        'review_count': int,
        'is_open': int
    }

    # Load the data
    df = pd.read_csv(filepath, dtype=dtypes, low_memory=False)

    # Check for required columns
    required_columns = ['business_id', 'name', 'stars', 'review_count']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in business data: {missing_columns}")

    return df


def load_review_data(filepath: str) -> pd.DataFrame:
    """
    Load review data from CSV file with appropriate dtypes.

    Args:
        filepath: Path to the review CSV file

    Returns:
        DataFrame containing review data

    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If required columns are missing
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Review data file not found: {filepath}")

    # Define dtypes for review data
    dtypes = {
        'review_id': str,
        'user_id': str,
        'business_id': str,
        'stars': int,
        'useful': int,
        'funny': int,
        'cool': int,
        'text': str,
        'date': str
    }

    # Load the data
    df = pd.read_csv(filepath, dtype=dtypes, low_memory=False)

    # Check for required columns
    required_columns = ['review_id', 'user_id', 'business_id', 'stars', 'text', 'date']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in review data: {missing_columns}")

    return df


def load_user_data(filepath: str) -> pd.DataFrame:
    """
    Load user data from CSV file with appropriate dtypes.

    Args:
        filepath: Path to the user CSV file

    Returns:
        DataFrame containing user data

    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If required columns are missing
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"User data file not found: {filepath}")

    # Define dtypes for user data
    dtypes = {
        'user_id': str,
        'name': str,
        'review_count': int,
        'yelping_since': str,
        'useful': int,
        'funny': int,
        'cool': int,
        'elite': str,
        'friends': str,
        'fans': int,
        'average_stars': float,
        'compliment_hot': int,
        'compliment_more': int,
        'compliment_profile': int,
        'compliment_cute': int,
        'compliment_list': int,
        'compliment_note': int,
        'compliment_plain': int,
        'compliment_cool': int,
        'compliment_funny': int,
        'compliment_writer': int,
        'compliment_photos': int
    }

    # Load the data
    df = pd.read_csv(filepath, dtype=dtypes, low_memory=False)

    # Check for required columns
    required_columns = ['user_id', 'name', 'review_count', 'yelping_since', 'average_stars']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in user data: {missing_columns}")

    return df