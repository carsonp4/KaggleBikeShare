library(vroom)
library(tidyverse)
library(gridExtra)

# Reading in Data
train <- vroom("train.csv")

# Changing Variables
# datetime - hourly date + timestamp  
train$season <- as.factor(train$season) # 1 = spring, 2 = summer, 3 = fall, 4 = winter 
train$holiday <- as.factor(train$holiday) # whether the day is considered a holiday
train$workingday <- as.factor(train$workingday) # whether the day is neither a weekend nor holiday
train$weather <- as.factor(train$weather) #1: Clear, Few clouds, Partly cloudy, Partly cloudy 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog 
# temp - temperature in Celsius
# atemp - "feels like" temperature in Celsius
# humidity - relative humidity
# windspeed - wind speed
# casual - number of non-registered user rentals initiated
# registered - number of registered user rentals initiated
# count - number of total rentals

# Different EDA
dplyr::glimpse(train) # lists the variable type of each column
skimr::skim(train) # dataset overview
DataExplorer::plot_intro(train) # visualization of glimpse()
DataExplorer::plot_correlation(train) # correlation heat map between variables
DataExplorer::plot_bar(train) # bar charts of all discrete variables
DataExplorer::plot_histogram(train) # histograms of all numerical variables
DataExplorer::plot_missing(train) # percent missing in each column
#GGally::ggpairs(train) # 1/2 scatterplot and 1/2 correlation heat map


# Plot to upload
data_info <- DataExplorer::plot_intro(train)
corr_con <- DataExplorer::plot_correlation(train, type="continuous")
bar_dis <- DataExplorer::plot_bar(train)
hist_con <- DataExplorer::plot_histogram(train)

grid.arrange(data_info, corr_con, bar_dis[[1]], hist_con[[1]], ncol=2)


