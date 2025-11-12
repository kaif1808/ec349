import os
import torch

# File paths for input CSV files
DATA_DIR = "data"
INPUT_FILES = {
    "business": os.path.join(DATA_DIR, "yelp_business_data.csv"),
    "review": os.path.join(DATA_DIR, "yelp_review.csv"),
    "user": os.path.join(DATA_DIR, "yelp_user.csv"),
    "checkin": os.path.join(DATA_DIR, "yelp_checkin_data.csv"),
    "tip": os.path.join(DATA_DIR, "yelp_tip_data.csv")
}

# Output paths for processed data
OUTPUT_DIR = os.path.join(DATA_DIR, "processed")
OUTPUT_FILES = {
    "merged_data": os.path.join(OUTPUT_DIR, "merged_data.csv"),
    "featured_data": os.path.join(OUTPUT_DIR, "featured_data.csv"),
    "sentiment_data": os.path.join(OUTPUT_DIR, "sentiment_data.csv"),
    "final_model_data": os.path.join(OUTPUT_DIR, "final_model_data.csv")
}

FEATURED_DATA_PATH = OUTPUT_FILES["featured_data"]

# Model hyperparameters
LEARNING_RATE = 0.0001
BATCH_SIZE = 64
MAX_EPOCHS = 40

# Feature lists
CANDIDATE_FEATURES = [
    "user_average_stars",
    "business_average_stars",
    "user_review_count",
    "business_review_count",
    "time_yelping",
    "date_year",
    "total_elite_statuses",
    "elite_status",
    "normalized_sentiment_score"
]

EXPECTED_OPTIMAL_FEATURES = [
    "user_average_stars",
    "business_average_stars",
    "time_yelping",
    "elite_status",
    "normalized_sentiment_score"
]

# Random seed
SEED = 1

# Sentiment settings
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
MAX_TOKENS = 512
SENTIMENT_BATCH_SIZE = 64

# Device detection helper function
def get_device():
    """
    Detect the available device for PyTorch computations.

    Returns:
        str: 'mps' if MPS is available, 'cuda' if CUDA is available, otherwise 'cpu'
    """
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"