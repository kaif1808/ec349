import pandas as pd
import json
import os
import logging
from typing import List, Tuple
from itertools import combinations
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from src import config

logger = logging.getLogger(__name__)


def prepare_feature_data(df: pd.DataFrame, candidate_features: List[str]) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare data for feature selection.

    Args:
        df: Input DataFrame containing all features and target
        candidate_features: List of feature column names to consider

    Returns:
        Tuple of (X, y) where X is features DataFrame and y is target Series
    """
    # Select candidate features plus target column
    selected_cols = candidate_features + ['stars']
    subset_df = df[selected_cols].copy()

    # Remove rows with missing values in selected columns
    subset_df = subset_df.dropna()

    # Separate features and target
    X = subset_df[candidate_features]
    y = subset_df['stars']

    return X, y


def run_best_subset_selection(X: pd.DataFrame, y: pd.Series) -> List[str]:
    """
    Run best subset feature selection using exhaustive search over different numbers of features.

    Args:
        X: Feature DataFrame
        y: Target Series

    Returns:
        List of selected feature names
    """
    # Split data into train and validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    best_overall_mse = float('inf')
    best_k = None
    best_features = None

    max_k = min(10, len(X.columns))

    for k in tqdm(range(1, max_k + 1), desc="Evaluating k"):
        # Get all combinations of k features
        feature_combos = list(combinations(X.columns, k))

        best_mse_for_k = float('inf')
        best_combo_for_k = None

        for combo in tqdm(feature_combos, desc=f"Evaluating combinations for k={k}"):
            combo = list(combo)
            # Select features
            X_train_combo = X_train[combo]
            X_val_combo = X_val[combo]

            # Fit model
            model = RandomForestRegressor(n_estimators=100, random_state=1, n_jobs=-1)
            model.fit(X_train_combo, y_train)

            # Predict and compute MSE
            y_pred = model.predict(X_val_combo)
            mse = mean_squared_error(y_val, y_pred)

            if mse < best_mse_for_k:
                best_mse_for_k = mse
                best_combo_for_k = combo

        # Compare across k
        if best_mse_for_k < best_overall_mse:
            best_overall_mse = best_mse_for_k
            best_k = k
            best_features = best_combo_for_k

    logger.info(f"Best k: {best_k}")
    logger.info(f"Best feature set: {best_features}")
    logger.info(f"Best MSE: {best_overall_mse}")

    return best_features


def feature_selection_pipeline(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Complete feature selection pipeline.

    Args:
        df: Input DataFrame with all features and target

    Returns:
        Tuple of (final DataFrame with optimal features + target, list of optimal features)
    """
    # Get candidate features from config
    candidate_features = config.CANDIDATE_FEATURES

    # Prepare feature data
    X, y = prepare_feature_data(df, candidate_features)

    # Run best subset selection
    optimal_features = run_best_subset_selection(X, y)

    # Save optimal features to JSON
    optimal_features_path = os.path.join(config.OUTPUT_DIR, "optimal_features.json")
    with open(optimal_features_path, 'w') as f:
        json.dump(optimal_features, f, indent=2)

    # Create final dataset with optimal features + target
    final_cols = optimal_features + ['stars']
    final_df = df[final_cols].copy()

    # Remove any remaining missing values
    final_df = final_df.dropna()

    # Save final dataset
    final_data_path = config.OUTPUT_FILES["final_model_data"]
    os.makedirs(os.path.dirname(final_data_path), exist_ok=True)
    final_df.to_csv(final_data_path, index=False)

    # Return final DataFrame and feature list
    return final_df, optimal_features