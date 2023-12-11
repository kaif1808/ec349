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
use_python("/Users/kai/.virtualenvs/r-tensorflow/bin/python", required = TRUE)
py_config()


rm(list=ls())
final_data <- fread("final_data.csv")
#load("~/ec349/business_data.Rda")
#load("~/ec349/user_data.Rda")
#load("~/ec349/review_data.Rda")
#load("~/ec349/checkin_data.Rda")
#load("~/ec349/tip_data.Rda")

