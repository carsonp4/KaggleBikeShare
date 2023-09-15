# Loading Packages
library(tidyverse)
library(tidymodels)
library(vroom)

# Reading in Data
train <- vroom("train.csv")
test <- vroom("test.csv")

# Dplyr cleaning section
train <- train %>% select(-casual, -registered) #removing columns can't use

train <- train %>% #adding day average temperature metric
  group_by(date = as.Date(datetime)) %>%
  mutate(day_avg_temp = mean(temp, na.rm = TRUE))

train <- train %>% #adding day average humidity metric
  group_by(date = as.Date(datetime)) %>%
  mutate(day_avg_humidty = mean(humidity, na.rm = TRUE))

train <- train %>% #adding day average wind metric
  group_by(date = as.Date(datetime)) %>%
  mutate(day_avg_wind = mean(windspeed, na.rm = TRUE))

test <- test %>%
  group_by(date = as.Date(datetime)) %>%
  mutate(day_avg_temp = mean(temp, na.rm = TRUE))

test <- test %>%
  group_by(date = as.Date(datetime)) %>%
  mutate(day_avg_humidty = mean(humidity, na.rm = TRUE))

test <- test %>%
  group_by(date = as.Date(datetime)) %>%
  mutate(day_avg_wind = mean(windspeed, na.rm = TRUE))


# Creating Recipe 
my_recipe <- recipe(count ~ ., data = train) %>% 
  step_mutate(weather=ifelse(weather==4, 3, weather)) %>%
  step_num2factor(weather, levels=c("Sunny", "Mist", "Rain")) %>%
  step_num2factor(season, levels=c("Spring", "Summer", "Fall", "Winter")) %>%
  step_mutate(holiday=factor(holiday, levels=c(0,1), labels=c("No", "Yes"))) %>%
  step_mutate(workingday=factor(workingday,levels=c(0,1), labels=c("No", "Yes"))) %>%
  step_date(datetime, features="dow") %>% 
  step_time(datetime, features="hour") %>% 
  #step_dummy(all_nominal_predictors()) %>% 
  #step_zv(all_predictors()) %>% 
  step_rm(datetime, date)

# Using recipe to make dataset
prepped_recipe <- prep(my_recipe)
wrangled <- bake(prepped_recipe, new_data = train)

# Makign linear regression model
my_mod <- linear_reg() %>% #Type of model
  set_engine("lm") # Engine = What R function to use

# Fitting model to train data
bike_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_mod) %>%
  fit(data = train) # Fit the workflow

# Predicting test data
bike_predictions <- predict(bike_workflow,
                            new_data=test) # Use fit to predict

# Changing all negative values to 1 (never 0 in train)
bike_predictions[bike_predictions < 0] <- 1

# Making submit file with correct column names
submit <- cbind(test$datetime, bike_predictions)
colnames(submit) <- c("datetime", "count")
submit$datetime <- as.character(submit$datetime)

# Exporting submit file
write_csv(submit, "submit.csv")
