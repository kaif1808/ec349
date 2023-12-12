library(tm)
library(text)
library(reticulate)
library(dplyr)
library(tm)
library(tidyr)
library(wordcloud)
library(ggplot2)
library(jsonlite)
library(tidyverse)
library(randomForest)
library(xgboost)
library(data.table)
library(arrow)
library(duckdb)
library(stringr)
library(lubridate)
library(ggplot2)
use_python("/Users/kai/Library/r-miniconda-arm64/envs/r-reticulate/bin/python", required = TRUE)
py_config()


rm(list=ls())
final_data <- fread("final_data.csv")



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


min_score <- min(sentiment$score_x)
max_score <- max(sentiment$score_x)
final_data$normalized_sentiment_score <- 1 + ((sentiment$score_x - min_score) * (5 - 1)) / (max_score - min_score)

optimal_features <- c("business_average_stars", "user_average_stars", "normalized_user_review_count", "normalized_sentiment_score")
min_value <- min(final_data$user_review_count)
max_value <- max(final_data$user_review_count)

final_data$normalized_user_review_count <- 1 + (final_data$user_review_count - min_value) / (max_value - min_value) * 4
selected_features <- c("business_average_stars", "user_average_stars", "user_review_count", "total_elite_statuses", "time_yelping","normalized_user_review_count", "elite_status", "normalized_sentiment_score")
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
  sizes = c(1:5),  # Number of features to include (adjust as needed)
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
# Define the neural network
model <- keras_model_sequential() %>%
  layer_dense(units = 64, input_shape = c(length(optimal_features))) %>%
  layer_batch_normalization() %>%
  layer_activation('relu') %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 64) %>%
  layer_batch_normalization() %>%
  layer_activation('relu') %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 5, activation = 'softmax')

model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_adagrad(),
  metrics = c('accuracy')
)

# Train the model
y_train <- to_categorical(train_data$stars - 1, num_classes = 5)
y_test <- to_categorical(test_data$stars - 1, num_classes = 5)


history <- model %>% fit(
  as.matrix(train_data[, optimal_features]), y_train,
  epochs = 50,
  batch_size = 64,
  validation_split = 0.2,
)

# Evaluate the model
model_performance <- model %>% evaluate(
  as.matrix(test_data[, optimal_features]), y_test
)
print(model_performance)

