# R/feature_engineering.R

# This script loads the merged data and engineers new features
# required for the model.

library(data.table)
library(lubridate)

# --- Configuration ---
INPUT_FILE <- "merged_data.rds"
OUTPUT_FILE <- "featured_data.rds"
UTILS_FILE <- "R/utils.R"

# --- Load Utilities and Data ---
cat("Loading data and utilities...\n")
source(UTILS_FILE)
if (!file.exists(INPUT_FILE)) {
  stop("Merged data file not found. Please run data_preprocessing.R first.")
}
dt <- readRDS(INPUT_FILE)

# --- Feature Engineering ---
cat("Engineering features...\n")

# 1. Time since yelping
# Calculate the duration in weeks between when the user joined and when the review was posted.
dt[, time_yelping := as.numeric(difftime(date, yelping_since, units = "weeks"))]

# 2. Elite status features
# Extract the year from the review date
dt[, review_year := format(as.Date(date), "%Y")]

# Count total elite statuses up to the review year
dt[, total_elite_statuses := mapply(count_elite_statuses, strsplit(elite, ","), review_year)]

# Check if the user had elite status in the year of the review
dt[, elite_status := mapply(check_elite_status, strsplit(elite, ","), review_year)]


# --- Save Processed Data ---
cat("Saving data with new features to", OUTPUT_FILE, "...\n")
saveRDS(dt, file = OUTPUT_FILE)

cat("Feature engineering complete.\n")
