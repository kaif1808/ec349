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


