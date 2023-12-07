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
save(review_data, file = "review_data.Rda")
