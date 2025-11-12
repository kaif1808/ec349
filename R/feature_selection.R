# R/feature_selection.R

# This script uses Recursive Feature Elimination (RFE) to identify the
# most important predictors for the model.

library(data.table)
library(caret)
library(randomForest)

# --- Configuration ---
INPUT_FILE <- "sentiment_data.rds"
OUTPUT_FILE <- "final_model_data.rds"
SEED <- 1

# --- Load Data ---
cat("Loading data with sentiment scores...\n")
if (!file.exists(INPUT_FILE)) {
  stop("Sentiment data file not found. Please run sentiment_analysis.R first.")
}
dt <- readRDS(INPUT_FILE)

# --- Feature Selection with RFE ---
cat("Running Recursive Feature Elimination (RFE)...
")

# Define predictor variables for RFE
# Note: 'text' and other non-numeric/ID fields are excluded.
selected_features <- c(
  "business_average_stars", "user_average_stars", "user_review_count",
  "total_elite_statuses", "time_yelping", "elite_status",
  "normalized_sentiment_score"
)

# RFE requires the data to not have missing values in the selected columns
# A simple approach is to omit rows with NA's in the predictors or target.
dt_complete <- na.omit(dt, cols = c(selected_features, "stars"))

# Configure the RFE algorithm
control <- rfeControl(
  functions = rfFuncs, # Using Random Forest for feature importance
  method = "cv",
  number = 5 # Using 5-fold cross-validation for speed
)

# Set seed for reproducibility
set.seed(SEED)

# Run RFE
rfe_results <- rfe(
  x = dt_complete[, ..selected_features],
  y = dt_complete$stars,
  sizes = c(1:length(selected_features)), # Test subsets of features from 1 to all
  rfeControl = control
)

# Print the results
cat("RFE Results:\n")
print(rfe_results)

# Get the list of optimal features
optimal_features <- predictors(rfe_results)
cat("\nOptimal features selected by RFE:\n")
print(optimal_features)

# --- Prepare Final Dataset ---
# Subset the data to include only the optimal features and the target variable
final_dt <- dt_complete[, c(optimal_features, "stars"), with = FALSE]

# --- Save Final Data ---
cat("Saving final data for modeling to", OUTPUT_FILE, "...\n")
saveRDS(final_dt, file = OUTPUT_FILE)
# Also save the list of optimal features for the training script
saveRDS(optimal_features, file = "optimal_features.rds")


cat("Feature selection complete.\n")
