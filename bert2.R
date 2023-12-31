transformers <- import("transformers")
torch <- import("torch")

model_name <- "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer <- transformers$BertTokenizer$from_pretrained(model_name)
model <- transformers$BertForSequenceClassification$from_pretrained(model_name)

final_data <- fread("final_data.csv")

inputs <- lapply(final_data$text, tokenizer$encode_plus, add_special_tokens = TRUE, truncation = TRUE, 
                 return_tensors = "pt")

# Perform sentiment analysis using the Huggingface Transformer model
# Predict sentiment for each review
outputs <- lapply(inputs, function(input) {
  with(torch$no_grad(), {
    model(input_ids = input$input_ids, token_type_ids = input$token_type_ids, 
          attention_mask = input$attention_mask)$logits
  })
})


predictions <- sapply(outputs, function(output) {
  torch$argmax(output)$item()
})

threshold <- 2.5
dataset$binary_scores <- ifelse(predictions >= threshold, 1, 0)

dataset %>% 
  mutate(sentiment = ifelse(dataset$binary_scores > 0, "Positive", "Negative")) %>%
  group_by(sentiment) %>%
  summarise(count = n())

dataset$sentimentresult <- ifelse(dataset$binary_scores >0 , "Positive", "Negative")
dataset$sentimentresult