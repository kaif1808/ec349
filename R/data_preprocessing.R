# R/data_preprocessing.R

# This script loads the raw Yelp data, merges the relevant tables,
# and performs initial cleaning and type conversion.

library(data.table)
library(jsonlite)
library(lubridate)

# --- Configuration ---
REVIEW_DATA_FILE <- "yelp_review_small.Rda"
USER_DATA_FILE <- "yelp_user_small.Rda"
BUSINESS_DATA_FILE <- "yelp_academic_dataset_business.json"
OUTPUT_FILE <- "merged_data.rds"

# --- Load Data ---
cat("Loading datasets...\n")

# Check if all required files exist
if (!file.exists(REVIEW_DATA_FILE) || !file.exists(USER_DATA_FILE) || !file.exists(BUSINESS_DATA_FILE)) {
  stop("One or more data files are missing. Please ensure all required data is in the project root.")
}

# Load Rda files
load(REVIEW_DATA_FILE)
load(USER_DATA_FILE)

# Load JSON data using stream_in for efficiency
business_data <- stream_in(file(BUSINESS_DATA_FILE))

# Convert to data.table for performance
setDT(review_data_small)
setDT(user_data_small)
business_data <- as.data.table(business_data)


# --- Data Cleaning and Renaming ---
cat("Cleaning and renaming columns...\n")

# Rename columns to avoid conflicts and improve clarity
setnames(user_data_small, c("useful", "funny", "cool", "review_count", "name", "average_stars"),
         c("total_useful", "total_funny", "total_cool", "user_review_count", "user_name", "user_average_stars"))

setnames(business_data, c("stars", "review_count", "name"),
         c("business_average_stars", "business_review_count", "business_name"))

# --- Type Conversion ---
cat("Converting data types...\n")

# Convert date strings to datetime objects
review_data_small[, date := ymd_hms(date)]
user_data_small[, yelping_since := ymd_hms(yelping_since)]


# --- Merging Data ---
cat("Merging datasets...\n")

# Merge reviews with users, then with businesses
merged_dt <- merge(review_data_small, user_data_small, by = "user_id")
merged_dt <- merge(merged_dt, business_data, by = "business_id")


# --- Save Merged Data ---
cat("Saving merged data to", OUTPUT_FILE, "...\n")
saveRDS(merged_dt, file = OUTPUT_FILE)

# --- Clean up workspace ---
rm(review_data_small, user_data_small, business_data, merged_dt)

cat("Data preprocessing complete.\n")