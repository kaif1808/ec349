import pandas as pd
import os
from typing import Tuple

from src.config import INPUT_FILES, OUTPUT_FILES
from src.data_loading import load_business_data, load_review_data, load_user_data


def rename_columns(user_df: pd.DataFrame, business_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Rename columns in user and business DataFrames to avoid naming conflicts.

    Args:
        user_df: DataFrame containing user data
        business_df: DataFrame containing business data

    Returns:
        Tuple of (renamed_user_df, renamed_business_df)
    """
    # Rename user columns
    user_renames = {
        'useful': 'total_useful',
        'funny': 'total_funny',
        'cool': 'total_cool',
        'review_count': 'user_review_count',
        'name': 'user_name',
        'average_stars': 'user_average_stars'
    }

    # Rename business columns
    business_renames = {
        'stars': 'business_average_stars',
        'review_count': 'business_review_count',
        'name': 'business_name'
    }

    # Apply renames
    renamed_user_df = user_df.rename(columns=user_renames)
    renamed_business_df = business_df.rename(columns=business_renames)

    return renamed_user_df, renamed_business_df


def convert_date_columns(review_df: pd.DataFrame, user_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert date columns to datetime dtype.

    Args:
        review_df: DataFrame containing review data
        user_df: DataFrame containing user data

    Returns:
        Tuple of (converted_review_df, converted_user_df)
    """
    # Convert 'date' column in review_df to datetime
    converted_review_df = review_df.copy()
    converted_review_df['date'] = pd.to_datetime(converted_review_df['date'])

    # Convert 'yelping_since' column in user_df to datetime
    converted_user_df = user_df.copy()
    converted_user_df['yelping_since'] = pd.to_datetime(converted_user_df['yelping_since'])

    return converted_review_df, converted_user_df
def merge_datasets(review_df: pd.DataFrame, user_df: pd.DataFrame, business_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge review, user, and business DataFrames using inner joins.

    Args:
        review_df: DataFrame containing review data
        user_df: DataFrame containing user data
        business_df: DataFrame containing business data

    Returns:
        Merged DataFrame with all three sources combined
    """
    # Inner join review -> user on 'user_id'
    merged = review_df.merge(user_df, on='user_id', how='inner')
    # Then result -> business on 'business_id'
    merged = merged.merge(business_df, on='business_id', how='inner')
    return merged


def clean_merged_data(merged_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean merged DataFrame by removing rows with missing values in critical columns.

    Args:
        merged_df: Merged DataFrame from merge_datasets

    Returns:
        Cleaned DataFrame with no missing values in critical columns
    """
    # Drop rows with missing values in specified columns
    cleaned = merged_df.dropna(subset=['stars', 'text', 'business_average_stars', 'user_average_stars', 'user_review_count'])
    return cleaned


def preprocess_pipeline() -> pd.DataFrame:
    """
    Complete preprocessing pipeline: load, rename, convert dates, merge, clean, and save.

    Returns:
        Final preprocessed DataFrame
    """
    # Load all three datasets
    review_df = load_review_data(INPUT_FILES["review"])
    user_df = load_user_data(INPUT_FILES["user"])
    business_df = load_business_data(INPUT_FILES["business"])

    # Rename columns
    user_df, business_df = rename_columns(user_df, business_df)

    # Convert date columns
    review_df, user_df = convert_date_columns(review_df, user_df)

    # Merge datasets
    merged_df = merge_datasets(review_df, user_df, business_df)

    # Clean merged data
    cleaned_df = clean_merged_data(merged_df)

    # Ensure output directory exists
    output_dir = os.path.dirname(OUTPUT_FILES["merged_data"])
    os.makedirs(output_dir, exist_ok=True)

    # Save to CSV
    cleaned_df.to_csv(OUTPUT_FILES["merged_data"], index=False)

    return cleaned_df