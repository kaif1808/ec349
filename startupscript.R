#Startup Script

library(jsonlite)
library(tidyverse)
library(randomForest)
library(xgboost)
library(tensorflow)
library(data.table)
library(arrow)
library(duckdb)
library(stringr)
library(lubridate)


rm(list=ls())
load("~/ec349/business_data.Rda")
load("~/ec349/user_data.Rda")
load("~/ec349/review_data.Rda")
#load("~/ec349/checkin_data.Rda")
#load("~/ec349/tip_data.Rda")

