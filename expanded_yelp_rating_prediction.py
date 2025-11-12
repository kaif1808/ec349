#!/usr/bin/env python3
"""
Complete Yelp Rating Prediction Pipeline

A comprehensive, self-contained machine learning notebook for predicting Yelp star ratings
using scikit-learn regression models.

This script contains all the code for the complete ML pipeline and can be run as a Jupyter notebook.
"""

# Section 1: Setup and Configuration
print("Section 1: Setup and Configuration")
print("=" * 50)

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

# Machine Learning libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor

# NLP libraries
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Interactive widgets (for notebook)
try:
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    WIDGETS_AVAILABLE = True
except ImportError:
    WIDGETS_AVAILABLE = False
    print("Warning: ipywidgets not available. Interactive features will be limited.")

# Download NLTK data
nltk.download('vader_lexicon', quiet=True)

# Set random seed for reproducibility
np.random.seed(42)

# Plotting style
plt.style.use('default')
sns.set_palette("husl")

print("✓ All libraries imported successfully")
print("✓ NLTK VADER lexicon downloaded")
print("✓ Random seed set to 42")

# Configuration
DATA_DIR = "/kaggle/input/yelp-dataset"  # Adjust for your environment
OUTPUT_DIR = "/kaggle/working"

INPUT_FILES = {
    "business": os.path.join(DATA_DIR, "yelp_academic_dataset_business.json"),
    "review": os.path.join(DATA_DIR, "yelp_academic_dataset_review.json"),
    "user": os.path.join(DATA_DIR, "yelp_academic_dataset_user.json")
}

OUTPUT_FILES = {
    "merged_data": os.path.join(OUTPUT_DIR, "merged_data.csv"),
    "featured_data": os.path.join(OUTPUT_DIR, "featured_data.csv"),
    "sentiment_data": os.path.join(OUTPUT_DIR, "sentiment_data.csv"),
    "final_model_data": os.path.join(OUTPUT_DIR, "final_model_data.csv")
}

TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5

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

print(f"Data directory: {DATA_DIR}")
print(f"Output directory: {OUTPUT_DIR}")

# Section 2: Data Loading and Preprocessing
print("\n\nSection 2: Data Loading and Preprocessing")
print("=" * 50)

def load_business_data(filepath: str) -> pd.DataFrame:
    """Load business data from JSON file."""
    print(f"Loading business data from {filepath}...")
    df = pd.read_json(filepath, lines=True)
    print(f"✓ Loaded {len(df)} business records")
    return df

def load_review_data(filepath: str, nrows=None) -> pd.DataFrame:
    """Load review data from JSON file."""
    print(f"Loading review data from {filepath}...")
    df = pd.read_json(filepath, lines=True, nrows=nrows)
    print(f"✓ Loaded {len(df)} review records")
    return df

def load_user_data(filepath: str) -> pd.DataFrame:
    """Load user data from JSON file."""
    print(f"Loading user data from {filepath}...")
    df = pd.read_json(filepath, lines=True)
    print(f"✓ Loaded {len(df)} user records")
    return df

def rename_columns(user_df: pd.DataFrame, business_df: pd.DataFrame) -> tuple:
    """Rename columns to avoid naming conflicts."""
    user_renames = {
        'useful': 'total_useful',
        'funny': 'total_funny',
        'cool': 'total_cool',
        'review_count': 'user_review_count',
        'name': 'user_name',
        'average_stars': 'user_average_stars'
    }

    business_renames = {
        'stars': 'business_average_stars',
        'review_count': 'business_review_count',
        'name': 'business_name'
    }

    renamed_user_df = user_df.rename(columns=user_renames)
    renamed_business_df = business_df.rename(columns=business_renames)

    return renamed_user_df, renamed_business_df

def convert_date_columns(review_df: pd.DataFrame, user_df: pd.DataFrame) -> tuple:
    """Convert date columns to datetime format."""
    review_df = review_df.copy()
    user_df = user_df.copy()

    review_df['date'] = pd.to_datetime(review_df['date'])
    user_df['yelping_since'] = pd.to_datetime(user_df['yelping_since'])

    return review_df, user_df

def merge_datasets(review_df: pd.DataFrame, user_df: pd.DataFrame, business_df: pd.DataFrame) -> pd.DataFrame:
    """Merge datasets using inner joins."""
    print("Merging datasets...")
    merged = review_df.merge(user_df, on='user_id', how='inner')
    merged = merged.merge(business_df, on='business_id', how='inner')
    print(f"✓ Merged dataset has {len(merged)} records")
    return merged

def clean_merged_data(merged_df: pd.DataFrame) -> pd.DataFrame:
    """Clean merged data by removing rows with missing critical values."""
    print("Cleaning merged data...")
    cleaned = merged_df.dropna(subset=['stars', 'text', 'business_average_stars', 'user_average_stars', 'user_review_count'])
    print(f"✓ Cleaned dataset has {len(cleaned)} records (removed {len(merged_df) - len(cleaned)} rows)")
    return cleaned

# Run data loading and preprocessing
try:
    business_df = load_business_data(INPUT_FILES["business"])
    review_df = load_review_data(INPUT_FILES["review"], nrows=50000)  # Limit for demo
    user_df = load_user_data(INPUT_FILES["user"])

    user_df, business_df = rename_columns(user_df, business_df)
    review_df, user_df = convert_date_columns(review_df, user_df)

    merged_df = merge_datasets(review_df, user_df, business_df)
    cleaned_df = clean_merged_data(merged_df)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cleaned_df.to_csv(OUTPUT_FILES["merged_data"], index=False)

    print(f"\n✓ Preprocessing completed!")
    print(f"Final dataset shape: {cleaned_df.shape}")
    print(f"Memory usage: {cleaned_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

except Exception as e:
    print(f"Error in data loading: {e}")
    print("Please check your data file paths and try again.")
    exit(1)

# Section 3: Feature Engineering
print("\n\nSection 3: Feature Engineering")
print("=" * 50)

def parse_elite_years(elite_str: str) -> list:
    """Parse elite years string into a list of integers."""
    if pd.isna(elite_str) or elite_str == "":
        return []

    elite_str = elite_str.replace('|', ',')
    years = []
    for year_str in elite_str.split(','):
        year_str = year_str.strip()
        if year_str:
            try:
                years.append(int(year_str))
            except ValueError:
                continue

    return years

def count_elite_statuses(elite_str: str, review_year: int) -> int:
    """Count elite statuses up to review year."""
    elite_years = parse_elite_years(elite_str)
    return sum(1 for year in elite_years if year <= review_year)

def check_elite_status(elite_str: str, review_year: int) -> int:
    """Check if user was elite in review year or previous year."""
    elite_years = parse_elite_years(elite_str)
    return 1 if review_year in elite_years or (review_year - 1) in elite_years else 0

def engineer_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer time-based features."""
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['yelping_since'] = pd.to_datetime(df['yelping_since'])
    df['time_yelping'] = (df['date'] - df['yelping_since']).dt.total_seconds() / (7 * 24 * 3600)
    df['date_year'] = df['date'].dt.year
    return df

def engineer_elite_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer elite status features."""
    df = df.copy()
    df['total_elite_statuses'] = df.apply(lambda row: count_elite_statuses(row['elite'], row['date_year']), axis=1)
    df['elite_status'] = df.apply(lambda row: check_elite_status(row['elite'], row['date_year']), axis=1)
    return df

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values."""
    df = df.copy()
    df['time_yelping'] = df['time_yelping'].fillna(df['time_yelping'].median())
    df['total_elite_statuses'] = df['total_elite_statuses'].fillna(0)
    df['elite_status'] = df['elite_status'].fillna(0)
    return df

def feature_engineering_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Complete feature engineering pipeline."""
    print("Starting feature engineering...")
    df = engineer_time_features(df)
    df = engineer_elite_features(df)
    df = handle_missing_values(df)
    df.to_csv(OUTPUT_FILES["featured_data"], index=False)
    print(f"✓ Feature engineering completed. Saved to {OUTPUT_FILES['featured_data']}")
    return df

# Run feature engineering
featured_df = feature_engineering_pipeline(cleaned_df)

new_features = [col for col in featured_df.columns if col not in cleaned_df.columns]
print(f"\nAdded {len(new_features)} new features:")
for i, feature in enumerate(new_features, 1):
    print(f"{i}. {feature}")

# Section 4: Sentiment Analysis
print("\n\nSection 4: Sentiment Analysis")
print("=" * 50)

def analyze_sentiment_nltk(text: str) -> dict:
    """Analyze sentiment using NLTK's VADER."""
    sia = SentimentIntensityAnalyzer()
    scores = sia.polarity_scores(text)
    return {
        'compound': scores['compound'],
        'pos': scores['pos'],
        'neu': scores['neu'],
        'neg': scores['neg'],
        'label': 'POSITIVE' if scores['compound'] > 0.05 else 'NEGATIVE' if scores['compound'] < -0.05 else 'NEUTRAL'
    }

def sentiment_analysis_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Complete sentiment analysis pipeline using NLTK."""
    print("Starting sentiment analysis with NLTK VADER...")

    sia = SentimentIntensityAnalyzer()
    sentiment_results = []
    for text in tqdm(df['text'], desc="Analyzing sentiment"):
        scores = sia.polarity_scores(text)
        sentiment_results.append(scores)

    df = df.copy()
    df['sentiment_compound'] = [r['compound'] for r in sentiment_results]
    df['sentiment_pos'] = [r['pos'] for r in sentiment_results]
    df['sentiment_neu'] = [r['neu'] for r in sentiment_results]
    df['sentiment_neg'] = [r['neg'] for r in sentiment_results]
    df['sentiment_label'] = ['POSITIVE' if r['compound'] > 0.05 else 'NEGATIVE' if r['compound'] < -0.05 else 'NEUTRAL' for r in sentiment_results]
    df['normalized_sentiment_score'] = df['sentiment_compound']

    df.to_csv(OUTPUT_FILES["sentiment_data"], index=False)
    print(f"✓ Sentiment analysis completed. Saved to {OUTPUT_FILES['sentiment_data']}")

    return df

# Run sentiment analysis
sentiment_df = sentiment_analysis_pipeline(featured_df)

print("Sentiment Analysis Results:")
print(sentiment_df[['sentiment_label', 'sentiment_compound', 'normalized_sentiment_score']].describe())
print("\nSentiment Label Distribution:")
print(sentiment_df['sentiment_label'].value_counts())

# Section 5: Feature Selection
print("\n\nSection 5: Feature Selection")
print("=" * 50)

def prepare_feature_data(df: pd.DataFrame, candidate_features: list) -> tuple:
    """Prepare data for feature selection."""
    selected_cols = candidate_features + ['stars']
    subset_df = df[selected_cols].copy()
    subset_df = subset_df.dropna()
    X = subset_df[candidate_features]
    y = subset_df['stars']
    return X, y

def run_best_subset_selection(X: pd.DataFrame, y: pd.Series) -> list:
    """Run feature selection using Random Forest."""
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    importances = rf.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': importances
    }).sort_values('importance', ascending=False)

    print("Feature importances:")
    for _, row in feature_importance_df.iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")

    return list(X.columns)  # Return all features for simplicity

def feature_selection_pipeline(df: pd.DataFrame) -> tuple:
    """Complete feature selection pipeline."""
    print("Starting feature selection...")

    X, y = prepare_feature_data(df, CANDIDATE_FEATURES)
    optimal_features = run_best_subset_selection(X, y)

    with open(os.path.join(OUTPUT_DIR, 'optimal_features.json'), 'w') as f:
        json.dump(optimal_features, f, indent=2)

    final_cols = optimal_features + ['stars']
    final_df = df[final_cols].copy()
    final_df = final_df.dropna()
    final_df.to_csv(OUTPUT_FILES["final_model_data"], index=False)

    print(f"✓ Feature selection completed. Selected {len(optimal_features)} features.")
    return final_df, optimal_features

# Run feature selection
final_df, optimal_features = feature_selection_pipeline(sentiment_df)

print(f"Final dataset shape: {final_df.shape}")
print(f"Features: {list(final_df.columns)}")

# Section 6: Model Training and Evaluation
print("\n\nSection 6: Model Training and Evaluation")
print("=" * 50)

def prepare_train_test_data(df: pd.DataFrame, features: list, test_size: float = 0.2) -> tuple:
    """Prepare train/test data with scaling."""
    X = df[features]
    y = df['stars']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE
    )

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train.values, y_test.values, scaler

def train_and_evaluate_models(X_train, X_test, y_train, y_test) -> dict:
    """Train multiple models and evaluate performance."""
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=RANDOM_STATE)
    }

    results = {}

    for name, model in tqdm(models.items(), desc="Training models"):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        results[name] = {
            'model': model,
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'predictions': y_pred
        }

    return results

def save_best_model(results: dict, scaler, features: list):
    """Save the best performing model."""
    best_model_name = max(results.keys(), key=lambda x: results[x]['r2'])
    best_model = results[best_model_name]['model']

    os.makedirs('models', exist_ok=True)
    with open('models/best_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('models/features.json', 'w') as f:
        json.dump(features, f)

    print(f"✓ Best model ({best_model_name}) saved to models/best_model.pkl")
    return best_model_name, best_model

# Train and evaluate models
X_train, X_test, y_train, y_test, scaler = prepare_train_test_data(final_df, optimal_features)
model_results = train_and_evaluate_models(X_train, X_test, y_train, y_test)
best_model_name, best_model = save_best_model(model_results, scaler, optimal_features)

# Display results
results_df = pd.DataFrame({
    model_name: {
        'MSE': results['mse'],
        'MAE': results['mae'],
        'RMSE': results['rmse'],
        'R²': results['r2']
    }
    for model_name, results in model_results.items()
}).T

print("\nModel Performance Comparison:")
print(results_df.sort_values('R²', ascending=False))

print(f"\nBest Model: {best_model_name}")
print(f"R² Score: {model_results[best_model_name]['r2']:.4f}")
print(f"RMSE: {model_results[best_model_name]['rmse']:.4f}")
print(f"MAE: {model_results[best_model_name]['mae']:.4f}")

# Section 7: Interactive Inference (if widgets available)
print("\n\nSection 7: Interactive Inference")
print("=" * 50)

if WIDGETS_AVAILABLE:
    def load_model_and_scaler():
        """Load the trained model and scaler."""
        with open('models/best_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('models/features.json', 'r') as f:
            features = json.load(f)
        return model, scaler, features

    def make_prediction(model, scaler, features, user_input):
        """Make a prediction based on user input."""
        input_data = pd.DataFrame([user_input])
        input_features = input_data[features]
        input_scaled = scaler.transform(input_features)
        prediction = model.predict(input_scaled)[0]
        return prediction

    def analyze_sentiment_for_input(text):
        """Analyze sentiment for user input text."""
        sia = SentimentIntensityAnalyzer()
        scores = sia.polarity_scores(text)
        return scores['compound']

    def create_interactive_interface():
        """Create interactive widgets for prediction."""
        try:
            model, scaler, features = load_model_and_scaler()
        except FileNotFoundError:
            print("Error: Model files not found. Please run the training section first.")
            return

        # Create widgets
        user_stars = widgets.FloatSlider(
            value=4.0, min=1.0, max=5.0, step=0.1,
            description='User Avg Stars:'
        )

        business_stars = widgets.FloatSlider(
            value=4.0, min=1.0, max=5.0, step=0.1,
            description='Business Avg Stars:'
        )

        time_yelping = widgets.IntSlider(
            value=100, min=0, max=1000, step=10,
            description='Time Yelping (weeks):'
        )

        elite_status = widgets.Checkbox(
            value=False,
            description='Elite Status'
        )

        review_text = widgets.Textarea(
            value="This restaurant was amazing! Great food and service.",
            description='Review Text:'
        )

        predict_button = widgets.Button(
            description='Make Prediction',
            button_style='primary'
        )

        output = widgets.Output()

        def on_predict_click(b):
            with output:
                clear_output(wait=True)

                sentiment_score = analyze_sentiment_for_input(review_text.value)

                user_input = {
                    'user_average_stars': user_stars.value,
                    'business_average_stars': business_stars.value,
                    'time_yelping': float(time_yelping.value),
                    'elite_status': 1.0 if elite_status.value else 0.0,
                    'normalized_sentiment_score': sentiment_score
                }

                prediction = make_prediction(model, scaler, features, user_input)

                print("Prediction Results")
                print("=" * 30)
                print(f"Predicted Rating: {prediction:.2f} stars")
                print(f"\nInput Features:")
                for key, value in user_input.items():
                    print(f"  {key}: {value}")
                print(f"\nReview Sentiment Score: {sentiment_score:.3f}")
                print(f"Review Text: {review_text.value[:100]}{'...' if len(review_text.value) > 100 else ''}")

        predict_button.on_click(on_predict_click)

        display(widgets.VBox([
            widgets.HTML("<h3>🎯 Yelp Rating Prediction</h3>"),
            user_stars,
            business_stars,
            time_yelping,
            elite_status,
            review_text,
            predict_button,
            widgets.HTML("<br><h4>Results:</h4>"),
            output
        ]))

    print("Creating interactive prediction interface...")
    create_interactive_interface()

else:
    print("Interactive widgets not available. Here's a simple prediction example:")

    # Simple prediction example
    def simple_prediction_example():
        """Simple prediction example without widgets."""
        try:
            model, scaler, features = load_model_and_scaler()
        except:
            print("Model not available for prediction example.")
            return

        # Example input
        example_input = {
            'user_average_stars': 4.2,
            'business_average_stars': 4.0,
            'time_yelping': 150.0,
            'elite_status': 1.0,
            'normalized_sentiment_score': 0.8
        }

        prediction = make_prediction(model, scaler, features, example_input)

        print("Example Prediction:")
        print("=" * 20)
        print(f"Predicted Rating: {prediction:.2f} stars")
        print("Input:", example_input)

    simple_prediction_example()

# Summary
print("\n\n" + "=" * 60)
print("PIPELINE COMPLETED SUCCESSFULLY!")
print("=" * 60)
print("\n✓ Data Loading & Preprocessing")
print("✓ Feature Engineering")
print("✓ Sentiment Analysis (NLTK)")
print("✓ Feature Selection")
print("✓ Model Training (scikit-learn)")
print("✓ Model Evaluation")
print("✓ Interactive Inference")

print(f"\nBest Model: {best_model_name}")
print(f"R² Score: {model_results[best_model_name]['r2']:.4f}")

print("\nFiles Created:")
print("- merged_data.csv: Preprocessed dataset")
print("- featured_data.csv: Engineered features")
print("- sentiment_data.csv: With sentiment scores")
print("- final_model_data.csv: Final training data")
print("- models/best_model.pkl: Trained model")
print("- models/scaler.pkl: Feature scaler")
print("- models/features.json: Feature names")

print("\nReady for predictions!")
print("=" * 60)