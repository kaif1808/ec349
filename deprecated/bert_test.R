# Load the reticulate package to interface with Python
if (!requireNamespace("reticulate", quietly = TRUE)) {
  install.packages("reticulate")
}
library(reticulate)

# Ensure the transformers Python package is installed
if (!py_module_available("transformers")) {
  py_install("transformers")
}
if (!py_module_available("tensorflow")) {
  py_install("tensorflow")
}


# Import the necessary Python modules
transformer <- import("transformers")
tf <- import("tensorflow")

# Specify the pre-trained model name
model_name <- "nlptown/bert-base-multilingual-uncased-sentiment"

# Initialize the tokenizer and model from the pre-trained model
# This is done once and reused for all predictions
tokenizer <- transformer$BertTokenizer$from_pretrained(model_name)
model <- transformer$TFBertForSequenceClassification$from_pretrained(model_name)

#' Get sentiment scores for a batch of texts using a pre-trained BERT model.
#'
#' @param texts A character vector of texts to analyze.
#' @return A numeric vector of sentiment scores (from 1 to 5).
get_sentiment_scores_bert <- function(texts) {
  # Tokenize the batch of texts
  encodings <- tokenizer$batch_encode_plus(
    texts,
    add_special_tokens = TRUE,
    max_length = 512,
    truncation = TRUE,
    padding = "longest",
    return_attention_mask = TRUE,
    return_tensors = 'tf'  # Use TensorFlow tensors
  )
  
  # Perform sentiment analysis on the entire batch
  outputs <- model(encodings$input_ids, attention_mask = encodings$attention_mask)
  
  # Extract the logits (raw prediction scores)
  logits <- outputs$logits
  
  # Get the predicted sentiment class for each text in the batch
  # The sentiment score is the index of the highest logit + 1 (since classes are 1-5)
  sentiment_scores <- as.integer(tf$argmax(logits, axis = 1L)$numpy()) + 1
  
  return(sentiment_scores)
}

# --- Example Usage ---
# texts_to_analyze <- c(
#   "This is a fantastic product! I love it.",
#   "The service was terrible. I am very disappointed.",
#   "It's an okay movie, not great but not bad either."
# )
# 
# sentiment_scores <- get_sentiment_scores_bert(texts_to_analyze)
# 
# # Print the results
# for (i in seq_along(texts_to_analyze)) {
#   cat(sprintf("Text: \"%s\"\nSentiment Score: %d\n\n", texts_to_analyze[i], sentiment_scores[i]))
# }