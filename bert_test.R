#BERT
# Use the transformers package
transformer <- import("transformers")

k_bert = import('keras_bert')

token_dict = k_bert$load_vocabulary("nlptown/bert-base-multilingual-uncased-sentiment")



g# Install and load necessary packages
install.packages('keras')
install.packages('tensorflow')
library(keras)
library(tensorflow)

# Initialize the BERT tokenizer and model
tokenizer <- text_field_preprocessor('nlptown/bert-base-multilingual-uncased-sentiment')
model <- TFBertForSequenceClassification$from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')



get_sentiment_score_bert <- function(text) {
  # Tokenize the text
  encoding <- tokenizer$encode_plus(
    text,
    add_special_tokens = TRUE,
    max_length = 512,
    truncation = TRUE,
    padding = TRUE,
    return_attention_mask = TRUE,
    return_tensors = 'tf'  # Use TensorFlow tensors
  )
  
  # Convert to TensorFlow tensors
  input_ids <- tf$convert_to_tensor(encoding$input_ids, dtype = tf$int32)
  attention_mask <- tf$convert_to_tensor(encoding$attention_mask, dtype = tf$int32)
  
  # Perform the sentiment analysis
  output <- model(input_ids, attention_mask = attention_mask)
  
  # Extract the sentiment scores and find the maximum
  logits <- output[[1]][[1]]$numpy()
  sentiment_score <- which.max(logits)  # Returns a score between 1 and 5
  return(sentiment_score)
}
