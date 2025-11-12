#!/usr/bin/env python3
"""
Quick Start Example: Complete Pipeline for Yelp Rating Prediction

This script demonstrates how to run the complete pipeline for training a Yelp rating prediction model,
including data loading, preprocessing, feature engineering, sentiment analysis, feature selection, and training.

Requirements:
- All required data files in data/ directory
- Installed dependencies from requirements.txt
"""

import logging
from src.utils import set_seed, verify_gpu_support
from src.preprocessing import preprocess_pipeline
from src.features import feature_engineering_pipeline
from src.sentiment import sentiment_analysis_pipeline
from src.feature_selection import feature_selection_pipeline
from src.train import training_pipeline

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """
    Run the complete pipeline from data loading to model training.
    """
    logger.info("Starting Yelp Rating Prediction Pipeline")

    # Step 1: Setup and verification
    logger.info("Step 1: Setting up environment")
    set_seed(1)  # Set random seed for reproducibility
    gpu_available = verify_gpu_support()  # Check GPU support

    if gpu_available:
        logger.info("GPU support detected - using MPS/CUDA acceleration")
    else:
        logger.info("No GPU support detected - using CPU")

    # Step 2: Data preprocessing
    logger.info("Step 2: Running data preprocessing pipeline")
    merged_data = preprocess_pipeline()
    logger.info(f"Preprocessing complete. Dataset shape: {merged_data.shape}")

    # Step 3: Feature engineering
    logger.info("Step 3: Running feature engineering pipeline")
    featured_data = feature_engineering_pipeline(merged_data)
    logger.info(f"Feature engineering complete. Dataset shape: {featured_data.shape}")

    # Step 4: Sentiment analysis
    logger.info("Step 4: Running sentiment analysis pipeline")
    sentiment_data = sentiment_analysis_pipeline(featured_data)
    logger.info(f"Sentiment analysis complete. Dataset shape: {sentiment_data.shape}")

    # Step 5: Feature selection
    logger.info("Step 5: Running feature selection pipeline")
    final_data, optimal_features = feature_selection_pipeline(sentiment_data)
    logger.info(f"Feature selection complete. Optimal features: {optimal_features}")
    logger.info(f"Final dataset shape: {final_data.shape}")

    # Step 6: Model training
    logger.info("Step 6: Running model training pipeline")
    training_results = training_pipeline()
    logger.info("Training complete!")
    logger.info(f"Model saved to: {training_results['model_path']}")
    logger.info(f"Scaler saved to: {training_results['scaler_path']}")
    logger.info(f"Metrics saved to: {training_results['metrics_path']}")
    logger.info(f"Test Metrics: MSE={training_results['metrics']['mse']:.4f}, "
                f"MAE={training_results['metrics']['mae']:.4f}, "
                f"R²={training_results['metrics']['r2']:.4f}")

    logger.info("Pipeline execution completed successfully!")
    logger.info("You can now use the trained model for inference with examples/inference.py")

if __name__ == "__main__":
    main()