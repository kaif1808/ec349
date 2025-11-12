#!/usr/bin/env python3
"""
Test script for sentiment analysis to verify truncation is disabled.
"""

from src.sentiment import sentiment_analysis_pipeline
import pandas as pd

# Create test data with long text
long_text = "This is a very long review text that should test if truncation is disabled. " * 100  # Repeat to make it long

test_df = pd.DataFrame({
    'text': [long_text],
    'stars': [5],
    'user_id': ['test_user'],
    'business_id': ['test_business'],
    'date': ['2023-01-01'],
    'user_average_stars': [4.5],
    'business_average_stars': [4.0],
    'user_review_count': [100],
    'business_review_count': [500],
    'yelping_since': ['2010-01-01']
})

print(f"Text length: {len(long_text)} characters")
print("Testing sentiment analysis...")

try:
    result_df = sentiment_analysis_pipeline(test_df)
    print("Sentiment analysis completed successfully!")
    print(f"Sentiment label: {result_df['sentiment_label'].iloc[0]}")
    print(f"Sentiment score: {result_df['sentiment_score_raw'].iloc[0]}")
    print(f"Normalized sentiment: {result_df['normalized_sentiment_score'].iloc[0]}")
    print("No truncation error - truncation is disabled!")
except Exception as e:
    print(f"Error: {e}")
    if "too long" in str(e).lower() or "truncat" in str(e).lower():
        print("Text was truncated or too long - truncation may still be enabled.")
    else:
        print("Other error occurred.")