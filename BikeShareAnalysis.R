# Loading Packages
library(tidyverse)
library(tidymodels)
library(vroom)
library(poissonreg)

# Reading in Data
train <- vroom("train.csv")
test <- vroom("test.csv")

# Dplyr section
train <- train %>% select(-casual, -registered) #removing columns can't use
train$count <- log(train$count)

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
  step_dummy(all_nominal_predictors()) %>% 
  #step_zv(all_predictors()) %>% 
  step_rm(datetime, date)

# Using recipe to make dataset
prepped_recipe <- prep(my_recipe)
bake(prepped_recipe, new_data = train)
bake(prepped_recipe, new_data = test)


# Linear Regression -------------------------------------------------------
# Making linear regression model
lr_mod <- linear_reg() %>% #Type of model
  set_engine("lm") # Engine = What R function to use

# Fitting model to train data
lr_bike_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(lr_mod) %>%
  fit(data = train) # Fit the workflow

extract_fit_engine(lr_bike_workflow) %>%
  summary()

# Predicting test data
lr_bike_predictions <- predict(lr_bike_workflow,
                            new_data=test) # Use fit to predict

# Changing all negative values to 1 (never 0 in train)
lr_bike_predictions[lr_bike_predictions < 0] <- 1

# Making submit file with correct column names
lr_submit <- cbind(test$datetime, lr_bike_predictions)
colnames(lr_submit) <- c("datetime", "count")
lr_submit$datetime <- as.character(lr_submit$datetime)
lr_submit$count <- exp(lr_submit$count)

# Exporting submit file
write_csv(lr_submit, "lr_submit.csv")


# Poisson Regression ------------------------------------------------------
# Making poisson regression model
pois_mod <- poisson_reg() %>% #Type of model
  set_engine("glm") # GLM = generalized linear model

# Fitting model to train data
pois_bike_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(pois_mod) %>%
  fit(data = train) # Fit the workflow

extract_fit_engine(pois_bike_workflow) %>%
  summary()

# Predicting test data
pois_bike_predictions <- predict(pois_bike_workflow,
                               new_data=test) # Use fit to predict

# Making submit file with correct column names
pois_submit <- cbind(test$datetime, pois_bike_predictions)
colnames(pois_submit) <- c("datetime", "count")
pois_submit$datetime <- as.character(pois_submit$datetime)
pois_submit$count <- exp(pois_submit$count)

# Exporting submit file
write_csv(pois_submit, "pois_submit.csv")


# Penalized Regression ----------------------------------------------------
# Making penalized regression model
pen_mod <- linear_reg(penalty=0.5, mixture=0) %>% #Type of model
  set_engine("glm") # GLM = generalized linear model

# Fitting model to train data
pen_bike_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(pen_mod) %>%
  fit(data = train) # Fit the workflow

extract_fit_engine(pen_bike_workflow) %>%
  summary()

# Predicting test data
pen_bike_predictions <- predict(pen_bike_workflow,
                                 new_data=test) # Use fit to predict

# Making submit file with correct column names
pen_submit <- cbind(test$datetime, pen_bike_predictions)
colnames(pen_submit) <- c("datetime", "count")
pen_submit$datetime <- as.character(pen_submit$datetime)
pen_submit$count <- exp(pen_submit$count)

# Exporting submit file
write_csv(pen_submit, "pen_submit.csv")


# Cross-Validation --------------------------------------------------------
# Making model
cv_mod <- linear_reg(penalty=tune(), mixture=tune()) %>% #Type of model
  set_engine("glmnet")

# Building work flow
cv_bike_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(cv_mod)

# Grid of values to tune over
tuning_grid <- grid_regular(penalty(), mixture(), levels=20)

# Split data for CV
folds <- vfold_cv(train, v = 15, repeats = 1)

# Run CV
cv_results <- cv_bike_workflow %>% 
              tune_grid(resamples = folds,
                        grid = tuning_grid,
                        metrics = metric_set(rmse, mae, rsq))

# Find best tune
bestTune <- cv_results %>% 
  select_best("rmse")

# Finalize and fit
cv_final_wf <- cv_bike_workflow %>% 
  finalize_workflow(bestTune) %>% 
  fit(data=train)

# Predicting test data
cv_bike_predictions <- predict(cv_final_wf,
                                new_data=test) # Use fit to predict

# Making submit file with correct column names
cv_submit <- cbind(test$datetime, cv_bike_predictions)
colnames(cv_submit) <- c("datetime", "count")
cv_submit$datetime <- as.character(cv_submit$datetime)
cv_submit$count <- exp(cv_submit$count)

# Exporting submit file
write_csv(cv_submit, "cv_submit.csv")

