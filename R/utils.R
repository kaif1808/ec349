# Shared utility functions for EC349 project

# Determine if user held elite status in review year or prior year
check_elite_status <- function(elite_years, review_year) {
  review_year <- as.numeric(review_year)
  as.integer(review_year %in% elite_years | (review_year - 1) %in% elite_years)
}

# Count total elite statuses up to and including the review year
count_elite_statuses <- function(elite_years, review_year) {
  elite_years <- as.numeric(elite_years[elite_years != ""])  # Remove empty strings and convert to numeric
  sum(elite_years <= review_year)
}

# Truncate text to a maximum length (characters)
truncate_text <- function(text, max_length) {
  if (nchar(text) > max_length) {
    substr(text, 1, max_length)
  } else {
    text
  }
}
