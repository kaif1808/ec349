library(tidyverse)

#converting date into date & time value instead of string
#checkin_data <- checkin_data %>%
#separate_rows(date, sep = ",\\s*")
#checkin_data$date <- ymd_hms(checkin_data$date)
review_data$date <- ymd_hms(review_data$date)
#tip_data$date <- ymd_hms(tip_data$date)
user_data$yelping_since <- ymd_hms(user_data$yelping_since)


#renaming user variables to total

library(dplyr)

user_data <- user_data %>%
  rename(
    total_useful = useful,
    total_funny = funny,
    total_cool = cool,
    user_review_count = review_count,
    user_name = name,
    user_average_stars = average_stars
  )

#renaming business variables
business_data <- business_data %>%
  rename(
    business_average_stars = stars,
    business_review_count = review_count,
    business_name = name
  )


#parsing what year a user was elite in
user_data$elite_list <- strsplit(user_data$elite, ",")
all_years <- unique(unlist(user_data$elite_list))
all_years <- all_years[all_years != ""]  # Remove empty 

for(year in all_years) {
  user_data[[paste0("elite_", year)]] <- sapply(user_data$elite_list, function(x) as.integer(year %in% x))
}

no_elite_indices <- which(user_data$elite == "")
for(year in all_years) {
  if(paste0("elite_", year) %in% names(user_data)) {
    user_data[no_elite_indices, paste0("elite_", year)] <- 0
  }
}

user_data$elite_list <- NULL



#merging user, review and business datasets
final_data <- review_data %>%
  inner_join(user_data, by = "user_id") 

final_data <- final_data %>%
  inner_join(business_data, by = "business_id")

rm(business_data, review_data, user_data)

#creating variable to show total amount of time user has been yelping by time they have published review
final_data$time_yelping <- difftime(final_data$date, final_data$yelping_since, units = "weeks")
final_data$time_yelping <- as.numeric(final_data$time_yelping)

#constructing dummy variable for holding elite status in year of review and variable showing amount of elite years held
final_data$date_year <- as.Date(final_data$date)
final_data$review_year <- format(final_data$date_year, "%Y")
# moved to R/utils.R
source("R/utils.R")
final_data$total_elite_statuses <- mapply(count_elite_statuses, strsplit(final_data$elite, ","), final_data$review_year)


final_data <- final_data %>%
  arrange(user_id, date)

final_data <- final_data %>%
  group_by(user_id) %>%
  mutate(cumulative_stars = cumsum(stars)) %>%
  ungroup()


final_data$elite_status <- mapply(check_elite_status, strsplit(final_data$elite, ","), final_data$review_year)

as.data.table(final_data)

selected_features <- c("business_average_stars", "user_average_stars", "user_review_count", "total_elite_statuses", "time_yelping", "text", "elite_status", "normalized_sentiment_score")
final_data <- final_data[, selected_features]
fwrite(final_data, "final_data_sentimen140k.csv")