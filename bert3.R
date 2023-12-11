library(text)
library(reticulate)
library(dplyr)
library(tm)
library(tidyr)
library(wordcloud)
library(ggplot2)

####Install miniconda to create a python environment in rstuido
install_miniconda(path = miniconda_path(), update = TRUE, force = FALSE)

##Using conda to create python environment
reticulate::conda_create()

## installing keras in your python env
install_keras( 
  method = c("auto", "virtualenv", "conda"), 
  conda = "auto", 
  version = "default", 
  extra_packages = NULL, 
  pip_ignore_installed = TRUE )

library(keras)
###Use reticulate to install transformers in R python environment
reticulate::py_install("transformers")

transformers <- import("transformers")
torch <- import("torch")

model_name <- "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer <- transformers$BertTokenizer$from_pretrained(model_name)
model <- transformers$BertForSequenceClassification$from_pretrained(model_name)

# Assuming final_data is already loaded
# Preprocess the text data
reviews <- tolower(final_data$text)
reviews <- sapply(reviews, removePunctuation)


inputs <- lapply(reviews, function(review) {
  tokenizer$encode_plus(
    review,
    add_special_tokens = TRUE,
    max_length = 512,  # You can adjust this based on your longest review
    truncation = TRUE,
    padding = "max_length",
    return_tensors = "pt"
  )
})


# Model inference
outputs <- lapply(inputs, function(input) {
  with(torch$no_grad(), {
    model(input$input_ids, attention_mask = input$attention_mask)$logits
  })
})


# Assuming outputs are the logits from the model
mean_sentiment_scores <- sapply(outputs, function(output) {
  probabilities <- torch$softmax(output, dim = 1)
  mean_score <- torch$mean(probabilities, dim = 1)$item()
  return(mean_score)
})

# Attach the scores to the dataset
final_data$mean_sentiment_score <- mean_sentiment_scores

model_name <- "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer <- transformers$BertTokenizer$from_pretrained(model_name)
model <- transformers$BertForSequenceClassification$from_pretrained(model_name)

reviews <- tolower(final_data$text)
reviews <- sapply(reviews, removePunctuation)
rm(final_data)

inputs <- lapply(reviews, tokenizer$encode_plus, add_special_tokens = TRUE, truncation = TRUE, return_tensors = "pt")



# Model inference
outputs <- lapply(inputs, function(input) {
  with(torch$no_grad(), {
    model(input_ids = input$input_ids, token_type_ids = input$token_type_ids, attention_mask = input$attention_mask)$logits
  })
})


# Assuming outputs are the logits from the model
mean_sentiment_scores <- sapply(outputs, function(output) {
  torch$argmax(output)$item()
})


# Attach the scores to the dataset
final_data <- fread("final_data.csv")
final_data$mean_sentiment_score <- mean_sentiment_scores
