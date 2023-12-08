#rfe to find important features
variables_to_check <- c("business_review_count", "business_average_stars", "cool", "date", 
                        "fans", "funny", "time_yelping", "total_cool", 
                        "user_average_stars", "total_elite_statuses", "user_review_count", 
                        "elite_status", "is_open")
library(data.table)

missing_values <- sapply(final_data[, variables_to_check], anyNA)
print(missing_values)

rm(final_data, all_years, missing_values, no_elite_indices, variables_to_check, year, check_elite_status, count_elite_statuses)
library(caret)
library(dplyr)
library(keras)

# Assuming train_data is your training dataset
# Convert 'date' to a numerical value if it's not already (e.g., time since a reference date)
# Convert 'is_open' and 'elite_status' to numeric if they are factors
train_data <- train_data %>%
  mutate(date = as.numeric(as.Date(date))) %>%
  mutate_at(vars(is_open, elite_status), as.numeric)


# Assuming these are the selected features
selected_features <- c("business_review_count", "business_average_stars", "cool", "fans", 
                       "funny", "time_yelping", "total_cool", "user_average_stars", 
                       "total_elite_statuses", "user_review_count", "elite_status", "is_open")



train_data <- train_data[, c(selected_features, "stars")]
test_data <- test_data[, c(selected_features, "stars")]



as.data.table(test_data)
as.data.table(train_data)
# Prepare the data
x_train <- as.matrix(train_data[selected_features])
x_test <- as.matrix(test_data[selected_features])

# Normalize the features
mean <- apply(x_train, 2, mean)
std <- apply(x_train, 2, sd)
x_train <- scale(x_train, center = mean, scale = std)
x_test <- scale(x_test, center = mean, scale = std)

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
  layer_dense(units = 64, activation = 'relu', input_shape = c(length(selected_features))) %>%
  layer_dense(units = 64, activation = 'relu') %>%
  layer_dense(units = 1)  # Assuming regression (if classification, adjust accordingly)

model %>% compile(
  loss = 'mean_squared_error',  # or 'binary_crossentropy' for classification
  optimizer = optimizer_rmsprop(),
  metrics = c('mean_absolute_error')
)

history <- model %>% fit(
  x_train, y_train,
  epochs = 20,  # Adjust based on your requirement
  batch_size = 128,
  validation_split = 0.2
)

model_performance <- model %>% evaluate(x_test, y_test)
print(model_performance)
