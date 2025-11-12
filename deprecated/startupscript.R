#Startup Script

library(reticulate)
library(jsonlite)
library(tidyverse)
library(randomForest)
library(xgboost)
library(data.table)
library(arrow)
library(duckdb)
library(stringr)
library(lubridate)
library(ggplot2)
# use_python(...)  # configure via .Rprofile or outside script
# py_config()


# rm(list=ls())  # avoid clearing workspace
#load("~/ec349/business_data.Rda")
#load("~/ec349/user_data.Rda")
#load("~/ec349/review_data.Rda")
#load("~/ec349/checkin_data.Rda")
#load("~/ec349/tip_data.Rda")