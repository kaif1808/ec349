# R/sentiment_analysis.R

# This script performs sentiment analysis on the review text data.

library(data.table)
library(text)
library(dplyr)

# --- Configuration ---
INPUT_FILE <- "featured_data.rds"
OUTPUT_FILE <- "sentiment_data.rds"
MODEL_NAME <- "distilbert-base-uncased-finetuned-sst-2-english"
MAX_LENGTH <- 512

# --- Load Data ---
cat("Loading data for sentiment analysis...\n")
featured_data <- readRDS(INPUT_FILE)

# --- Truncate Text ---
cat("Truncating review text...\n")
source("R/utils.R") # for truncate_text function
reviews <- sapply(featured_data$text, truncate_text, max_length = MAX_LENGTH)

# --- Perform Sentiment Analysis ---
cat("Performing sentiment analysis...\n")
sentiment <- textClassify(
  reviews,
  model = MODEL_NAME,
  device = "cpu",
  tokenizer_parallelism = FALSE,
  logging_level = "error",
  return_incorrect_results = FALSE,
  function_to_apply = "none",
  set_seed = "default"
)

# --- Process Sentiment Scores ---
cat("Processing sentiment scores...\n")
sentiment$score_x <- ifelse(sentiment$label_x == "NEGATIVE",
                            -sentiment$score_x,
                            sentiment$score_x)

sentiment$label_x <- as.integer(ifelse(sentiment$label_x == "NEGATIVE",
                                         1,
                                         0))

featured_data$sentiment_score <- sentiment$score_x

# --- Save Data ---
cat("Saving data with sentiment scores to", OUTPUT_FILE, "...\n")
saveRDS(featured_data, file = OUTPUT_FILE)

# --- Clean up workspace ---
rm(featured_data, reviews, sentiment)

cat("Sentiment analysis complete.\n")
