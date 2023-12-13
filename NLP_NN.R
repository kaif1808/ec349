setwd()
#Pre-Processing Yelp Academic Data for the Assignment
library(jsonlite)

#Clear
cat("\014")  
rm(list=ls())

#Load Different Data
business_data <- stream_in(file("yelp_academic_dataset_business.json")) #note that stream_in reads the json lines (as the files are json lines, not json)
checkin_data  <- stream_in(file("yelp_academic_dataset_checkin.json")) #note that stream_in reads the json lines (as the files are json lines, not json)
tip_data  <- stream_in(file("yelp_academic_dataset_tip.json")) #note that stream_in reads the json lines (as the files are json lines, not json)
user_data <- stream_in(file("yelp_academic_dataset_user.json")) #note that stream_in reads the json lines (as the files are json lines, not json)
review_data  <- stream_in(file("yelp_academic_dataset_review.json")) #note that stream_in reads the json lines (as the files are json lines, not json)

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
check_elite_status <- function(elite_years, review_year) {
  review_year <- as.numeric(review_year)
  as.integer(review_year %in% elite_years | (review_year - 1) %in% elite_years)
}

count_elite_statuses <- function(elite_years, review_year) {
  elite_years <- as.numeric(elite_years[elite_years != ""])  # Convert to numeric and remove empty strings
  sum(elite_years <= review_year)  # Count elite statuses up to the review year
}
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

                                                
library(tm)
library(text)
library(reticulate)
library(wordcloud)
library(ggplot2)
library(randomForest)
library(xgboost)
library(data.table)
library(arrow)
library(duckdb)
library(stringr)
library(lubridate)
library(ggplot2)

textrpp_install()
                                                
use_python("/Users/kai/Library/r-miniconda-arm64/envs/r-reticulate/bin/python", required = TRUE)
                                                
py_config()


rm(list=ls())


set.seed(1)  # For reproducibility

# Calculate the number of samples per group
samples_per_group <- 130000 / length(unique(final_data$stars))

# Perform stratified sampling
final_data <- final_data %>%
  group_by(stars) %>%
  sample_n(size = samples_per_group, replace = FALSE) %>%
  ungroup()

truncate_text <- function(text, max_length) {
  if (nchar(text) > max_length) {
    return(substr(text, 1, max_length))
  } else {
    return(text)
  }
}


max_length <- 512  # Set your desired max length

reviews <- final_data$text
reviews <- sapply(reviews, truncate_text, max_length = max_length)


sentiment <- textClassify(
  reviews,
  model = "distilbert-base-uncased-finetuned-sst-2-english",
  device = "cpu",
  tokenizer_parallelism = FALSE,
  logging_level = "error",
  return_incorrect_results = FALSE,
  function_to_apply = "none",
  set_seed = "default"
)

sentiment$score_x <- ifelse(sentiment$label_x == "NEGATIVE", 
                             -sentiment$score_x, 
                            sentiment$score_x)

sentiment$label_x <- as.integer(ifelse(sentiment$label_x == "NEGATIVE", 
                            1, 
                            0)
)



selected_features <- c("business_average_stars", "user_average_stars", "user_review_count", "total_elite_statuses", "time_yelping", "elite_status", "sentiment_score")
library(caret)
library(keras)
#rfe model used for features selection
control <- rfeControl(
  functions = rfFuncs,  # Assuming you're using a random forest model; change as needed
  method = "cv",
  number = 10  # Number of folds in cross-validation
)

model <- rfFuncs

set.seed(1)  # For reproducibility

rfe_results <- rfe(
  x = final_data[, selected_features],  # Predictor variables
  y = final_data$stars,       # Target variable
  sizes = c(1:5),  # Number of features to include
  rfeControl = control
)

print(rfe_results)

optimal_features <- predictors(rfe_results)
final_data <- final_data[, c(optimal_features, "stars")]




library(caret)
library(keras)
set.seed(1) #ensuring reproducibility
partition <- createDataPartition(final_data$stars, p = 0.2, list = FALSE, times = 1)
test_data <- final_data[partition, ]
train_data <- final_data[-partition, ]
nrow(test_data)
nrow(train_data)
test_data <- test_data[, c(optimal_features, "stars")]
train_data <- train_data[, c(optimal_features, "stars")]

rm(final_data, partition)

x_train <- as.matrix(train_data[, c(optimal_features)])
x_test <- as.matrix(test_data[, c(optimal_features)])

dimnames(x_train) <- NULL    
dimnames(x_test) <- NULL                                                
# Now we can normalise the independent variables (using keras::normalize function)
x_train <- keras::normalize(x_train[, c(optimal_features)])
x_test <- keras::normalize(x_test[, c(optimal_features)])

# Define the last variable, NSP, as numeric
x_train[,5] <- as.numeric(x_train[,5])-1   # the minus 1 ensures values become 0,1,2
x_test[,5] <- as.numeric(x_test[,5])-1                                                  

# Prepare the target variable
y_train <- train_data$stars
y_test <- test_data$stars
rm(test_data, train_data)

library(keras)
install_tensorflow()


install.packages("remotes")
remotes::install_github("rstudio/tensorflow")
install_keras()

model <- keras_model_sequential() %>%
  layer_dense(units = 64, activation = 'relu', input_shape = c(optimal_features)) %>%
  layer_batch_normalization() %>%
  layer_activation('relu') %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 32) %>%
  layer_activation('relu') %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 1)  # Regression 

callback_early_stopping <- callback_early_stopping(monitor = "val_loss", patience = 5)

                                                
model %>% compile(
  loss = 'mean_squared_error',  
  optimizer = optimizer_adabound(),
  metrics = c('mean_absolute_error')
)

history <- model %>% fit(
  x_train, y_train,
  epochs = 20,  
  batch_size = 64,
  validation_split = 0.2
  callbacks = list(callback_early_stopping)
)

model_performance <- model %>% evaluate(x_test, y_test)
print(model_performance)
