import pandas as pd
from src.utils import count_elite_statuses, check_elite_status
from src import config


def engineer_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer time-based features from the DataFrame.

    Calculates time_yelping as the difference between date and yelping_since
    in weeks, and extracts date_year from the date column.

    Args:
        df: Input DataFrame with 'date' and 'yelping_since' columns

    Returns:
        DataFrame with added 'time_yelping' and 'date_year' columns
    """
    df = df.copy()

    # Convert date columns to datetime if they are strings
    df['date'] = pd.to_datetime(df['date'])
    df['yelping_since'] = pd.to_datetime(df['yelping_since'])

    # Calculate time_yelping in weeks
    df['time_yelping'] = (df['date'] - df['yelping_since']).dt.total_seconds() / (7 * 24 * 3600)

    # Extract year from date
    df['date_year'] = df['date'].dt.year

    return df


def engineer_elite_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer elite status features from the DataFrame.

    Creates 'total_elite_statuses' by counting elite years up to the review year,
    and 'elite_status' by checking if the user was elite in the review year or previous year.

    Args:
        df: Input DataFrame with 'elite' and 'date_year' columns

    Returns:
        DataFrame with added 'total_elite_statuses' and 'elite_status' columns
    """
    df = df.copy()

    # Create total_elite_statuses using count_elite_statuses
    df['total_elite_statuses'] = df.apply(
        lambda row: count_elite_statuses(row['elite'], row['date_year']),
        axis=1
    )

    # Create elite_status using check_elite_status
    df['elite_status'] = df.apply(
        lambda row: check_elite_status(row['elite'], row['date_year']),
        axis=1
    )

    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in the DataFrame.

    Fills 'time_yelping' with the median value, and 'total_elite_statuses'
    and 'elite_status' with 0.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with missing values handled
    """
    df = df.copy()

    df['time_yelping'] = df['time_yelping'].fillna(df['time_yelping'].median())
    df['total_elite_statuses'] = df['total_elite_statuses'].fillna(0)
    df['elite_status'] = df['elite_status'].fillna(0)

    return df


def feature_engineering_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Complete feature engineering pipeline.

    Applies time feature engineering, elite feature engineering, handles missing values,
    saves the processed data to CSV, and returns the DataFrame.

    Args:
        df: Input DataFrame

    Returns:
        Processed DataFrame with engineered features
    """
    df = engineer_time_features(df)
    df = engineer_elite_features(df)
    df = handle_missing_values(df)

    df.to_csv(config.FEATURED_DATA_PATH, index=False)

    return df