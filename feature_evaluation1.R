#rfe to find important features
variables_to_check <- c("business_review_count", "business_average_stars", "cool", "date", 
                        "fans", "funny", "time_yelping", "total_cool", 
                        "user_average_stars", "total_elite_statuses", "user_review_count", 
                        "elite_status", "is_open")
library(data.table)

missing_values <- sapply(final_data[, variables_to_check], anyNA)
print(missing_values)

rm(final_data, all_years, missing_values, no_elite_indices, variables_to_check, year)
library(caret)
library(dplyr)

# Assuming train_data is your training dataset
# Convert 'date' to a numerical value if it's not already (e.g., time since a reference date)
# Convert 'is_open' and 'elite_status' to numeric if they are factors
train_data <- train_data %>%
  mutate(date = as.numeric(as.Date(date))) %>%
  mutate_at(vars(is_open, elite_status), as.numeric)

# Ensure all features are in the model dataframe
features <- c("business_review_count", "business_average_stars", "cool", "date", "fans", 
              "funny", "time_yelping", "total_cool", "user_average_stars", 
              "total_elite_statuses", "user_review_count", "elite_status", "is_open")
model_data <- train_data[, c(features, "stars")]
test_data <- test_data[, c(features, "stars")]
rm(train_data)
as.data.table(model_data)
as.data.table(test_data)


control <- rfeControl(functions = rfFuncs,  # for random forest, change if using a different model
                      method = "cv",       # cross-validation
                      number = 10)         # number of folds in CV

set.seed(123)  # for reproducibility
rfe_results <- rfe(model_data[, features], model_data$stars,
                   sizes = c(1:length(features)),  # sizes of feature subsets to evaluate
                   rfeControl = control)

final_features <- predictors(rfe_results)
final_model <- train(stars ~ ., data = model_data[, c(final_features, "stars")], method = "rf")  # change method as needed
