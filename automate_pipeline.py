#!/usr/bin/env python3
"""
Automated Machine Learning Pipeline for Yelp Rating Prediction

This script automates the complete ML pipeline including:
1. Data Loading & Preprocessing
2. Feature Engineering
3. Sentiment Analysis
4. Feature Selection
5. Model Training
6. Output Generation/Inference

The pipeline is modular, using functions from the src/ modules, with comprehensive
logging, progress tracking, and error handling.
"""

import logging
import sys
from typing import Any, Callable
from tqdm import tqdm

# Import pipeline modules
from src.preprocessing import preprocess_pipeline
from src.features import feature_engineering_pipeline
from src.sentiment import sentiment_analysis_pipeline
from src.feature_selection import feature_selection_pipeline
from src.train import training_pipeline
from src.model import YelpRatingPredictor
from src import config

# Import inference dependencies
import torch
import pickle
import pandas as pd
import json
import os

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('pipeline.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)


def run_stage(stage_name: str, stage_func: Callable[[], Any]) -> Any:
    """
    Wrapper function to run a pipeline stage with error handling and logging.

    Args:
        stage_name: Name of the pipeline stage for logging
        stage_func: Function to execute for this stage

    Returns:
        Result of the stage function if successful

    Raises:
        SystemExit: If a critical error occurs (FileNotFoundError or general Exception)
    """
    logger.info(f"Starting stage: {stage_name}")
    try:
        result = stage_func()
        logger.info(f"Completed stage: {stage_name}")
        return result
    except FileNotFoundError as e:
        logger.error(f"File not found in stage '{stage_name}': {e}")
        logger.error("Please ensure all required data files are present in the data/ directory.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error in stage '{stage_name}': {e}")
        logger.error("Pipeline execution failed. Check logs for details.")
        sys.exit(1)


def data_loading_preprocessing_stage() -> pd.DataFrame:
    """
    Stage 1: Data Loading & Preprocessing

    Loads raw Yelp datasets, performs preprocessing including column renaming,
    date conversion, dataset merging, and data cleaning. Saves the merged data.

    Returns:
        Preprocessed DataFrame ready for feature engineering
    """
    logger.info("Loading and preprocessing raw data...")
    df = preprocess_pipeline()
    logger.info(f"Preprocessed data shape: {df.shape}")
    return df


def feature_engineering_stage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stage 2: Feature Engineering

    Creates time-based and elite status features, handles missing values,
    and saves the featured data.

    Args:
        df: Preprocessed DataFrame from stage 1

    Returns:
        DataFrame with engineered features
    """
    logger.info("Engineering features...")
    df_featured = feature_engineering_pipeline(df)
    logger.info(f"Featured data shape: {df_featured.shape}")
    return df_featured


def sentiment_analysis_stage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stage 3: Sentiment Analysis

    Performs sentiment analysis on review texts using a pre-trained transformer model,
    normalizes sentiment scores, and adds sentiment features to the DataFrame.

    Args:
        df: Featured DataFrame from stage 2

    Returns:
        DataFrame with sentiment analysis features
    """
    logger.info("Performing sentiment analysis...")
    df_sentiment = sentiment_analysis_pipeline(df)
    logger.info(f"Sentiment data shape: {df_sentiment.shape}")
    return df_sentiment


def feature_selection_stage(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """
    Stage 4: Feature Selection

    Uses best subset selection to identify optimal features for model training,
    saves the selected features and final dataset.

    Args:
        df: DataFrame with all features from stage 3

    Returns:
        Tuple of (final DataFrame with optimal features, list of optimal features)
    """
    logger.info("Performing feature selection...")
    final_df, optimal_features = feature_selection_pipeline(df)
    logger.info(f"Selected {len(optimal_features)} optimal features: {optimal_features}")
    logger.info(f"Final model data shape: {final_df.shape}")
    return final_df, optimal_features


def model_training_stage() -> dict:
    """
    Stage 5: Model Training

    Trains a neural network model using the selected features, evaluates on test data,
    and saves the trained model, scaler, and metrics.

    Returns:
        Dictionary with training results including metrics and file paths
    """
    logger.info("Training the model...")
    results = training_pipeline()
    logger.info(f"Training completed. Metrics: {results['metrics']}")
    return results


def output_generation_inference_stage() -> None:
    """
    Stage 6: Output Generation/Inference

    Loads the trained model and generates example predictions to demonstrate
    the inference pipeline. Saves prediction results to outputs/predictions.json.
    """
    logger.info("Generating outputs and running inference demo...")

    # File paths
    model_path = 'models/best_model.pt'
    scaler_path = 'models/scaler.pkl'
    features_path = 'data/processed/optimal_features.json'
    output_path = 'outputs/predictions.json'

    # Check if required files exist
    for path in [model_path, scaler_path, features_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required file not found: {path}")

    # Load model, scaler, and features
    with open(features_path, 'r') as f:
        optimal_features = json.load(f)

    input_size = len(optimal_features)
    model = YelpRatingPredictor(input_size=input_size)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    # Create example data for inference
    example_data = {
        'user_average_stars': [4.2, 3.8, 4.5],
        'business_average_stars': [4.0, 3.5, 4.8],
        'time_yelping': [52.3, 24.1, 156.7],
        'elite_status': [1, 0, 1],
        'normalized_sentiment_score': [0.8, -0.3, 0.9]
    }

    # Filter to optimal features
    filtered_data = {k: v for k, v in example_data.items() if k in optimal_features}
    df = pd.DataFrame(filtered_data)

    # Make predictions
    scaled_data = scaler.transform(df.values)
    input_tensor = torch.FloatTensor(scaled_data)

    with torch.no_grad():
        predictions = model(input_tensor).flatten().numpy()

    # Save predictions
    os.makedirs('outputs', exist_ok=True)
    prediction_results = {
        'predictions': predictions.tolist(),
        'input_features': optimal_features,
        'example_inputs': df.to_dict('records')
    }

    with open(output_path, 'w') as f:
        json.dump(prediction_results, f, indent=2)

    logger.info(f"Inference completed. Predictions saved to {output_path}")
    logger.info(f"Example predictions: {predictions}")


def main():
    """
    Main function to execute the complete ML pipeline.

    Runs all pipeline stages sequentially with progress tracking and error handling.
    """
    logger.info("Starting Automated ML Pipeline for Yelp Rating Prediction")

    # Execute pipeline stages sequentially with progress bar
    stages = [
        ("Data Loading & Preprocessing", data_loading_preprocessing_stage),
        ("Feature Engineering", None),  # Will be called with df
        ("Sentiment Analysis", None),   # Will be called with df_featured
        ("Feature Selection", None),    # Will be called with df_sentiment
        ("Model Training", model_training_stage),
        ("Output Generation/Inference", output_generation_inference_stage)
    ]

    # Initialize data variables
    df = None
    df_featured = None
    df_sentiment = None
    final_df = None
    optimal_features = None
    training_results = None

    with tqdm(total=len(stages), desc="Pipeline Progress") as pbar:
        # Stage 1: Data Loading & Preprocessing
        df = run_stage(stages[0][0], stages[0][1])
        pbar.update(1)

        # Stage 2: Feature Engineering
        df_featured = run_stage(stages[1][0], lambda: feature_engineering_stage(df))
        pbar.update(1)

        # Stage 3: Sentiment Analysis
        df_sentiment = run_stage(stages[2][0], lambda: sentiment_analysis_stage(df_featured))
        pbar.update(1)

        # Stage 4: Feature Selection
        final_df, optimal_features = run_stage(stages[3][0], lambda: feature_selection_stage(df_sentiment))
        pbar.update(1)

        # Stage 5: Model Training
        training_results = run_stage(stages[4][0], stages[4][1])
        pbar.update(1)

        # Stage 6: Output Generation/Inference
        run_stage(stages[5][0], stages[5][1])
        pbar.update(1)

    logger.info("Pipeline execution completed successfully!")
    logger.info("Check the following outputs:")
    logger.info("- Processed data: data/processed/")
    logger.info("- Trained model: models/")
    logger.info("- Metrics and predictions: outputs/")


if __name__ == "__main__":
    main()