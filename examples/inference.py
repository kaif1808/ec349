#!/usr/bin/env python3
"""
Inference Example: Load Trained Model and Make Predictions

This script demonstrates how to load a trained Yelp rating prediction model and make predictions
on new data. It shows the complete inference pipeline including data preprocessing and prediction.

Requirements:
- Trained model file: models/best_model.pt
- Scaler file: models/scaler.pkl
- Optimal features file: data/processed/optimal_features.json
"""

import torch
import pickle
import pandas as pd
import json
import os
import logging
from src.model import YelpRatingPredictor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_model_and_scaler(model_path: str, scaler_path: str, features_path: str):
    """
    Load the trained model, scaler, and feature information.

    Args:
        model_path: Path to the saved model state dict
        scaler_path: Path to the saved scaler pickle file
        features_path: Path to the optimal features JSON file

    Returns:
        Tuple of (model, scaler, features)
    """
    # Load optimal features
    with open(features_path, 'r') as f:
        optimal_features = json.load(f)

    # Initialize model with correct input size
    input_size = len(optimal_features)
    model = YelpRatingPredictor(input_size=input_size)

    # Load trained weights
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    # Load scaler
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    logger.info(f"Model loaded with {input_size} input features: {optimal_features}")

    return model, scaler, optimal_features

def create_example_data(features: list) -> pd.DataFrame:
    """
    Create example data for inference demonstration.

    Args:
        features: List of feature names

    Returns:
        DataFrame with example data
    """
    # Create example data with realistic values
    example_data = {
        'user_average_stars': [4.2, 3.8, 4.5],
        'business_average_stars': [4.0, 3.5, 4.8],
        'time_yelping': [52.3, 24.1, 156.7],  # weeks
        'elite_status': [1, 0, 1],  # 1 if elite, 0 otherwise
        'normalized_sentiment_score': [0.8, -0.3, 0.9]  # -1 to 1 scale
    }

    # Filter to only include features that were selected
    filtered_data = {k: v for k, v in example_data.items() if k in features}

    df = pd.DataFrame(filtered_data)
    logger.info(f"Created example data with {len(df)} samples and {len(features)} features")

    return df

def make_predictions(model, scaler, data: pd.DataFrame) -> torch.Tensor:
    """
    Make predictions on the input data.

    Args:
        model: Trained PyTorch model
        scaler: Fitted MinMaxScaler
        data: Input DataFrame with features

    Returns:
        Tensor of predictions
    """
    # Scale the data
    scaled_data = scaler.transform(data.values)

    # Convert to tensor
    input_tensor = torch.FloatTensor(scaled_data)

    # Make predictions
    with torch.no_grad():
        predictions = model(input_tensor)

    return predictions.flatten()

def main():
    """
    Main inference demonstration.
    """
    logger.info("Starting Yelp Rating Prediction Inference Demo")

    # File paths
    model_path = 'models/best_model.pt'
    scaler_path = 'models/scaler.pkl'
    features_path = 'data/processed/optimal_features.json'

    # Check if required files exist
    for path in [model_path, scaler_path, features_path]:
        if not os.path.exists(path):
            logger.error(f"Required file not found: {path}")
            logger.error("Please run examples/quick_start.py first to train the model")
            return

    # Load model, scaler, and features
    logger.info("Loading trained model and preprocessing artifacts")
    model, scaler, features = load_model_and_scaler(model_path, scaler_path, features_path)

    # Create example data
    logger.info("Creating example data for prediction")
    example_df = create_example_data(features)

    # Display example data
    print("\nExample Input Data:")
    print(example_df)

    # Make predictions
    logger.info("Making predictions")
    predictions = make_predictions(model, scaler, example_df)

    # Display results
    print("\nPrediction Results:")
    print("-" * 50)
    for i, pred in enumerate(predictions.numpy(), 1):
        print(".2f")
        print(f"  Actual range: 1-5 stars")
        print(f"  Confidence: {'High' if abs(pred - round(pred)) < 0.3 else 'Medium'}")
        print()

    logger.info("Inference demo completed successfully!")

if __name__ == "__main__":
    main()