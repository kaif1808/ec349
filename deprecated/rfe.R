control <- rfeControl(
  functions = rfFuncs,  # Assuming you're using a random forest model; change as needed
  method = "cv",
  number = 10  # Number of folds in cross-validation
)

model <- rfFuncs

set.seed(1)  # For reproducibility

rfe_results <- rfe(
  x = train_data[, selected_features],  # Predictor variables
  y = train_data$stars,       # Target variable
  sizes = c(1:5),  # Number of features to include (adjust as needed)
  rfeControl = control
)

print(rfe_results)

optimal_features <- predictors(rfe_results)
final_data <- final_data[, c(optimal_features, "stars")]


set.seed(1) #ensuring reproducibility
partition <- createDataPartition(final_data$stars, p = 0.2, list = FALSE, times = 1)
test_data <- final_data[partition, ]
train_data <- final_data[-partition, ]
nrow(test_data)
nrow(train_data)

#> print(rfe_results)

#Recursive feature selection

#Outer resampling method: Cross-Validated (10 fold) 

#Resampling performance over subset size:
  
#  Variables   RMSE Rsquared    MAE  RMSESD RsquaredSD   MAESD Selected
#1 1.0131   0.5099 0.7921 0.02459    0.02439 0.02058         
#2 0.8452   0.6455 0.6425 0.02441    0.02155 0.01437         
#3 0.7858   0.6937 0.6126 0.03040    0.02553 0.01989         
#4 0.7747   0.7037 0.6035 0.03351    0.02812 0.02224        *
#  5 0.8248   0.6921 0.6727 0.02440    0.02775 0.02226         
#7 0.7790   0.6992 0.6079 0.03504    0.02989 0.02221         

#The top 4 variables (out of 4):
 # normalized_sentiment_score, user_average_stars, business_average_stars, user_review_count