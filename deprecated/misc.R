#dividing date into date and time
checkin_data <- checkin_data %>%
  mutate(
    new_date = substr(date, 1, 10),
    new_time = substr(date, 12, 19)
  )

review_data <- review_data %>%
  mutate(
    new_date = substr(date, 1, 10),
    new_time = substr(date, 12, 19)
  )

tip_data <- tip_data %>%
  mutate(
    new_date = substr(date, 1, 10),
    new_time = substr(date, 12, 19)
  )

#checkin_data$new_date <- as.Date(checkin_data$new_date, format = "%Y-%m-%d")
#review_data$new_date <- as.Date(review_data$new_date, format = "%Y-%m-%d")
#tip_data$new_date <- as.Date(tip_data$new_date, format = "%Y-%m-%d")

#checkin_data$new_time <- hms(checkin_data$new_time)
#review_data$new_time <- hms(review_data$new_time)
#tip_data$new_time <- hms(tip_data$new_time)
# Create start and end date columns for the 7-day range in both tables
checkin_data[, `:=`(start_date = date - days(3), end_date = date + days(4))]
tip_data[, `:=`(start_date = date - days(4), end_date = date + days(3))]

# Set keys for the interval join
setkey(checkin_data, business_id, start_date, end_date)
setkey(tip_data, business_id, start_date, end_date)

# Perform the non-equi join
checktip_data <- foverlaps(checkin_data, tip_data, by.x = c("business_id", "start_date", "end_date"), by.y = c("business_id", "start_date", "end_date"), type = "any", mult = "all")



# This will show unique characters in the date column
unique(unlist(strsplit(as.character(checkin_data$date), "")))

#checkin_data has multiple date-time strings concatentated so need to split and create new observations for each check in
checkin_data <- checkin_data %>%
  separate_rows(date, sep = ",\\s*")  # The separator is a comma followed by any number of spaces




#connecting by date-time value and business id
setDT(checkin_data)
setDT(tip_data)


checktip_data <- checkin_data[tip_data, 
                              on = .(business_id = business_id, 
                                     date = date),
                              nomatch = 0L,  # Ensures an inner join
                              allow.cartesian = TRUE]


# Function to preprocess text
preprocess_text <- Vectorize(function(text) {
  text <- tolower(text)  # Lowercase
  text <- removePunctuation(text)  # Remove punctuation
  text <- removeWords(text, stopwords("en"))  # Remove stopwords
  return(text)
})
final_data$text <- preprocess_text(final_data$text)


# Function to get sentiment score
# get_sentiment_score retained here as script-specific
get_sentiment_score <- function(text) {
  text <- preprocess_text(text)
  sentiment <- get_sentiment(text, method = "syuzhet")
  mean(sentiment)
}

# Apply sentiment analysis to each review
final_data$sentiment_score <- sapply(final_data$text, get_sentiment_score)

# virtualenv_install(...)  # avoid runtime installs in scripts

library(reticulate)

# Specify the path to the Python executable in your virtual environment
# use_python(...)  # configure outside scripts

# Check the configuration
# py_config()


# You can then download or use models directly in your Python scripts or in R using reticulate

# Train the model
y_train <- to_categorical(as.numeric(train_data$stars) - 1)
y_test <- to_categorical(as.numeric(test_data$stars) - 1)
x_train <- as.matrix(train_data[numerical_features])
x_test <- as.matrix(test_data[numerical_features])

history <- model %>% fit(
  x_train, y_train,
  epochs = 20,
  batch_size = 128,
  validation_split = 0.2
)

