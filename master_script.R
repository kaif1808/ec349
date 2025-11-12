# master_script.R

# This master script executes the entire data science pipeline, from
# data preprocessing to model training and evaluation.

# --- Introduction ---
cat("Starting the Yelp Star Rating Prediction Pipeline...\n\n")

# --- Step 1: Data Preprocessing ---
cat("--- Step 1: Running Data Preprocessing ---\\n")
source("R/data_preprocessing.R")
cat("\n")

# --- Step 2: Feature Engineering ---
cat("--- Step 2: Running Feature Engineering ---\\n")
source("R/feature_engineering.R")
cat("\n")

# --- Step 3: Sentiment Analysis ---
cat("--- Step 3: Running Sentiment Analysis ---\\n")
source("R/sentiment_analysis.R")
cat("\n")

# --- Step 4: Feature Selection ---
cat("--- Step 4: Running Feature Selection ---\\n")
source("R/feature_selection.R")
cat("\n")

# --- Step 5: Model Training ---
cat("--- Step 5: Running Model Training ---\\n")
source("R/model_training.R")
cat("\n")

# --- Completion ---
cat("Pipeline execution complete.\n")
cat("The final model has been saved to 'final_model.h5'.\n")
cat("A file with the data used for modeling is 'final_model_data.rds'.\n")
cat("The optimal features are saved in 'optimal_features.rds'.\n")
