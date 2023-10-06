# Loading Packages
library(tidyverse)
library(tidymodels)
library(vroom)
library(poissonreg)
library(rpart)
library(ranger)
library(stacks)

# Reading in Data
setwd("~/Desktop/Stat348/KaggleBikeShare/")
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



# Model -------------------------------------------------------------------
# Split data for CV
folds <- vfold_cv(train, v = 10)

# Control settings for Stacking Models
untunedModel <- control_stack_grid()
tunedModel <- control_stack_resamples()

# Linear Regression model
stack_linreg_model <- linear_reg(penalty = tune(),
                                 mixture = tune()) %>% 
  set_engine("glmnet")
stack_linreg_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(stack_linreg_model)
stack_linreg_tuning_grid <- grid_regular(penalty(),
                                         mixture(),
                                         levels = 5)
stack_linreg <- stack_linreg_wf %>% 
  tune_grid(resamples = folds,
            grid = stack_linreg_tuning_grid,
            metrics = metric_set(rmse, mae, rsq),
            control = untunedModel)

# Poisson Regression model
stack_poireg_model <- poisson_reg(penalty = tune(),
                                  mixture = tune()) %>%
  set_engine("glmnet")
stack_poireg_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(stack_poireg_model)
stack_poireg_tuning_grid <- grid_regular(penalty(),
                                         mixture(),
                                         levels = 5)
stack_poireg <- stack_poireg_wf %>% 
  tune_grid(resamples = folds,
            grid = stack_poireg_tuning_grid,
            metrics = metric_set(rmse, mae, rsq),
            control = untunedModel)

# Regression trees
stack_tree_model <- decision_tree(tree_depth = tune(),
                                  cost_complexity = tune(),
                                  min_n = tune()) %>% 
  set_engine("rpart") %>% 
  set_mode("regression")
stack_tree_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(stack_tree_model)
stack_tree_tuning_grid <- grid_regular(tree_depth(), cost_complexity(), min_n(), levels=5)
stack_tree <- stack_tree_wf %>% 
  tune_grid(resamples = folds,
            grid = stack_tree_tuning_grid,
            metrics = metric_set(rmse, mae, rsq),
            control = untunedModel)

# Random Forrest
stack_forrest_model <- rand_forest(mtry = tune(),
                                   min_n = tune(),
                                   trees = 500) %>% 
  set_engine("ranger") %>% 
  set_mode("regression")
stack_forrest_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(stack_forrest_model)
stack_forrest_tuning_grid <- grid_regular(mtry(c(1,ncol(train - 1))), min_n(), levels=5)
stack_forrest <- stack_forrest_wf %>% 
  tune_grid(resamples = folds,
            grid = stack_forrest_tuning_grid,
            metrics = metric_set(rmse, mae, rsq),
            control = untunedModel)

# Boosted
stack_xgboost_model <- boost_tree(mode = "regression",
                            trees = 100,
                            min_n = tune(),
                            tree_depth = tune(),
                            learn_rate = tune(),
                            loss_reduction = tune()) %>%
                 set_engine("xgboost", objective = "reg:squarederror")
stack_xgboost_wf <- workflow() %>%
  add_recipe(my_recipe) %>% 
  add_model(stack_xgboost_model) 
xgboost_params <- dials::parameters(min_n(),
                                    tree_depth(),
                                    learn_rate(),
                                    loss_reduction())
stack_xgboost_tuning_grid <- grid_max_entropy(xgboost_params, size = 30)
stack_xgboost <- stack_xgboost_wf %>% 
  tune_grid(resamples = folds,
            grid = stack_xgboost_tuning_grid,
            metrics = metric_set(rmse, rsq, mae),
            control = untunedModel)
  

# Stacking Models
bike_stack <- stacks() %>% 
  add_candidates(stack_linreg) %>% 
  add_candidates(stack_poireg) %>% 
  add_candidates(stack_tree) %>% 
  add_candidates(stack_forrest) %>% 
  add_candidates(stack_xgboost)

stack_model <- bike_stack %>% 
  blend_predictions() %>% 
  fit_members()

stack_bike_predictions <- predict(stack_model,
                                  new_data=test)

# Making submit file with correct column names
stack_submit <- cbind(test$datetime, stack_bike_predictions)
colnames(stack_submit) <- c("datetime", "count")
stack_submit$datetime <- as.character(stack_submit$datetime)
stack_submit$count <- exp(stack_submit$count)

# Exporting submit file
write_csv(stack_submit, "stack_submit.csv")

