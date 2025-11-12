import random
import numpy as np
import pandas as pd
import torch
import logging
from typing import List

logger = logging.getLogger(__name__)


def parse_elite_years(elite_str: str) -> List[int]:
    """
    Parse elite years string into a list of integers.

    Handles empty strings, NaN values, and comma/pipe-separated years.

    Args:
        elite_str: String containing elite years, e.g., "2018,2019,2020" or "2018|2019"

    Returns:
        List of integers representing elite years, or empty list for invalid input
    """
    if pd.isna(elite_str) or elite_str == "":
        return []

    # Replace pipe with comma for consistent splitting
    elite_str = elite_str.replace('|', ',')

    # Split by comma and convert to integers, filtering out empty strings
    years = []
    for year_str in elite_str.split(','):
        year_str = year_str.strip()
        if year_str:
            try:
                years.append(int(year_str))
            except ValueError:
                # Skip invalid year strings
                continue

    return years


def count_elite_statuses(elite_str: str, review_year: int) -> int:
    """
    Count the number of elite statuses up to and including the review year.

    Args:
        elite_str: String containing elite years
        review_year: The year of the review

    Returns:
        Number of elite years <= review_year
    """
    elite_years = parse_elite_years(elite_str)
    return sum(1 for year in elite_years if year <= review_year)


def check_elite_status(elite_str: str, review_year: int) -> int:
    """
    Check if the user was elite in the review year or the previous year.

    Args:
        elite_str: String containing elite years
        review_year: The year of the review

    Returns:
        1 if elite in review_year or (review_year - 1), 0 otherwise
    """
    elite_years = parse_elite_years(elite_str)
    return 1 if review_year in elite_years or (review_year - 1) in elite_years else 0
def smart_truncate_text(text: str, tokenizer, max_tokens: int = 500) -> str:
    """
    Tokenize text, keep first 250 + last 250 tokens if over max_tokens, convert back to string.
    """
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= max_tokens:
        return text
    # Keep first 250 and last 250
    first_part = tokens[:250]
    last_part = tokens[-250:]
    truncated_tokens = first_part + last_part
    return tokenizer.decode(truncated_tokens)


def set_seed(seed: int = 1) -> None:
    """
    Set random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def verify_gpu_support() -> bool:
    """
    Check MPS GPU support availability.
    """
    available = torch.backends.mps.is_available()
    status = "available" if available else "not available"
    logger.info(f"MPS GPU support is {status}.")
    return available