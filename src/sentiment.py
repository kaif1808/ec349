from transformers import pipeline, AutoTokenizer
import torch
from typing import List, Dict
import pandas as pd
from tqdm import tqdm
import os
from src.utils import smart_truncate_text
def initialize_sentiment_pipeline(device: str = "mps"):
    """
    Initialize sentiment analysis pipeline with device detection.

    Args:
        device: Device to use ('mps' or 'cpu'). Defaults to 'mps'.

    Returns:
        Hugging Face pipeline for sentiment analysis
    """
    # Detect device
    if device == "mps" and not torch.backends.mps.is_available():
        device = "cpu"
    elif device not in ["cpu", "mps"]:
        device = "cpu"  # fallback

    # Initialize pipeline
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=device,
        truncation=False
    )

    return sentiment_pipeline


def process_sentiment_batch(texts: List[str], pipeline, batch_size: int = 64) -> List[Dict]:
    """
    Process batch of texts through sentiment analysis pipeline.

    Args:
        texts: List of text strings to analyze
        pipeline: Hugging Face sentiment analysis pipeline
        batch_size: Number of texts to process in each batch

    Returns:
        List of sentiment analysis results (dicts with 'label' and 'score')
    """
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_results = pipeline(batch)
        results.extend(batch_results)
    return results


def normalize_sentiment_scores(sentiment_results: List[Dict]) -> pd.Series:
    """
    Normalize sentiment scores to range [-1, 1].

    Args:
        sentiment_results: List of sentiment analysis results

    Returns:
        Pandas Series with normalized scores (-1 for negative, +1 for positive)
    """
    scores = []
    for result in sentiment_results:
        label = result['label']
        score = result['score']
        if label == 'NEGATIVE':
            scores.append(-score)
        elif label == 'POSITIVE':
            scores.append(score)
        else:
            # Handle unexpected labels (e.g., neutral) by setting to 0
            scores.append(0.0)
    return pd.Series(scores)


def sentiment_analysis_pipeline(df: pd.DataFrame, batch_size: int = 64) -> pd.DataFrame:
    """
    Complete sentiment analysis pipeline.

    Args:
        df: DataFrame containing review texts in 'text' column
        batch_size: Number of texts to process in each batch

    Returns:
        DataFrame with added sentiment columns
    """
    # Initialize sentiment pipeline
    sentiment_pipeline = initialize_sentiment_pipeline()

    # Load tokenizer for truncation
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

    # Process in batches with tqdm progress bar
    results = []
    for i in tqdm(range(0, len(df), batch_size), desc="Processing sentiment analysis"):
        batch_texts = df['text'].iloc[i:i + batch_size].tolist()
        # Apply smart truncation to each text
        truncated_texts = [smart_truncate_text(text, tokenizer, max_tokens=500) for text in batch_texts]
        batch_results = process_sentiment_batch(truncated_texts, sentiment_pipeline, batch_size)
        results.extend(batch_results)

    # Normalize sentiment scores
    normalized_scores = normalize_sentiment_scores(results)

    # Add columns
    df['sentiment_label'] = [r['label'] for r in results]
    df['sentiment_score_raw'] = [r['score'] for r in results]
    df['normalized_sentiment_score'] = normalized_scores

    # Save to CSV with checkpointing (save every 1000 rows or at end)
    output_path = 'data/processed/sentiment_data.csv'
    checkpoint_interval = 1000
    for i in range(0, len(df), checkpoint_interval):
        end_idx = min(i + checkpoint_interval, len(df))
        temp_df = df.iloc[:end_idx]
        temp_df.to_csv(output_path, index=False)
        if end_idx < len(df):
            # Intermediate save
            pass

    return df