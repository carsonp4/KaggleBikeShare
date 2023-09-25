# Loading Packages
library(tidyverse)
library(tidymodels)
library(vroom)

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
  #step_dummy(all_nominal_predictors()) %>% 
  #step_zv(all_predictors()) %>% 
  step_rm(datetime, date) %>% 
  prep()

# Using recipe to make dataset
train_baked <- bake(my_recipe, new_data = train)
test_baked <- bake(my_recipe, new_data = test)




# XGBoost -------------------------------------------------------------------------
# https://www.r-bloggers.com/2020/05/using-xgboost-with-tidymodels/

train_split <- rsample::initial_split(train, prop = 0.2, strata = count)

train_cv_folds <- recipes::bake(my_recipe, new_data = training(train_split)) %>%  
                  rsample::vfold_cv(v = 20)


xgboost_model <- parsnip::boost_tree(mode = "regression",
                                    trees = 1000,
                                    min_n = tune(),
                                    tree_depth = tune(),
                                    learn_rate = tune(),
                                    loss_reduction = tune()) %>%
                set_engine("xgboost", objective = "reg:squarederror")

xgboost_params <- dials::parameters(min_n(),
                                    tree_depth(),
                                    learn_rate(),
                                    loss_reduction())

xgboost_grid <- dials::grid_max_entropy(xgboost_params, size = 60)

xgboost_wf <- workflows::workflow() %>%
              add_model(xgboost_model) %>% 
              add_formula(count ~ .)

xgboost_tuned <- tune::tune_grid(object = xgboost_wf,
                                resamples = train_cv_folds,
                                grid = xgboost_grid,
                                metrics = yardstick::metric_set(rmse, rsq, mae),
                                control = tune::control_grid(verbose = TRUE))    


xgboost_best_params <- xgboost_tuned %>%
                      tune::select_best("rmse")

xgboost_model_final <- xgboost_model %>% 
                      finalize_model(xgboost_best_params)

test_prediction <- xgboost_model_final %>%
  # fit the model on all the training data
  fit(formula = count ~ ., data    = train_baked) %>%
  # use the training model fit to predict the test data
  predict(new_data = test_baked) %>%
  bind_cols(test)

xg_submit <- test_prediction %>% 
             select(datetime, .pred)
colnames(xg_submit) <- c("datetime", "count")
xg_submit$datetime <- as.character(xg_submit$datetime)
xg_submit$count <- exp(xg_submit$count)

write_csv(xg_submit, "xg_submit.csv")
