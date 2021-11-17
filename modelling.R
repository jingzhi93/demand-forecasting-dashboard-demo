# Time Series ML
library(tidymodels)
library(modeltime)
library(modeltime.ensemble)

# Timing & Parallel Processing
library(tictoc)
library(future)
library(doFuture)

# Core 
library(tidyquant)
library(tidyverse)
library(timetk)
library(skimr)

# * Parallel Processing ----

registerDoFuture()
n_cores <- parallel::detectCores()
plan(
  strategy = cluster,
  workers  = parallel::makeCluster(n_cores)
)

# Params ----
horizon <-  365

# Raw table ----
data_raw_clean_tbl <- read_rds("data/cleaned_raw_partial_tbl.rds")

data_raw_clean_tbl %>% glimpse()

data_raw_clean_tbl %>% 
  plot_acf_diagnostics(date, sales, .lags = 365:600)

# Full data table -----
# we are not going to fix empty data since all data are available
full_data_tbl <- data_raw_clean_tbl %>%
  #feature transformation
  mutate(sales = log1p(sales)) %>% 
  
  #add future frame
  group_by(store, item) %>% 
  future_frame(date, .length_out = horizon, .bind_data = TRUE ) %>% 
  ungroup() %>% 
  
  #lags and rolling features
  group_by(store, item) %>%
  tk_augment_fourier(date, .periods = c(7, 14, 28)) %>% 
  tk_augment_lags(sales, .lags = 464) %>% 
  tk_augment_slidify(sales_lag464,
                     .f = ~mean(.x, na.rm = TRUE),
                     .period = c(7, 28),
                     .partial = TRUE,
                     .align = "center") %>% 
  bind_rows() %>% 
  rowid_to_column(var="rowid")

# full_data_tbl %>% write_rds("data/full_data_tbl.rds")
full_data_tbl %>% glimpse()

# Data Prepared table ----
data_prepared_tbl <- full_data_tbl %>% 
  filter(!is.na(sales)) %>% 
  drop_na()

data_prepared_tbl %>% glimpse()

# Future table ----
future_tbl <- full_data_tbl %>% 
  filter(is.na(sales))

future_tbl %>% glimpse()

#check for NA data in future table
future_tbl %>% filter(is.nan(sales_lag7_roll_28)) %>% glimpse()

#replace NAN with NA
future_tbl <- future_tbl %>% 
  mutate(
    across(
      .cols= contains("_lag"),
      .fns = function(x) ifelse(is.nan(x), NA, x)
    )
  ) %>% 
  mutate(
    across(
      .cols = contains("_lag"),
      .fns = ~replace_na(.x, 0)
    )
  )

future_tbl %>% filter(is.nan(sales_lag7_roll_28)) %>% glimpse()

# Time Split ----
splits <- data_prepared_tbl %>% 
  time_series_split(date, assess = horizon, cumulative = TRUE)

splits %>% 
  tk_time_series_cv_plan() %>% 
  plot_time_series_cv_plan(date, sales)

# Recipe ----
train_cleaned <- training(splits) %>% 
  group_by(store, item) %>% 
  mutate(sales = ts_clean_vec(sales, period = 7)) %>% 
  ungroup()

#view training data for store 1
train_cleaned %>% 
  group_by(store, item) %>% 
  plot_time_series(date, sales, 
                   .facet_ncol = 2,
                   .smooth = FALSE,
                   .interactive = FALSE)

# Prepare Recipe ----
recipe_base <- recipe(sales ~ ., data=train_cleaned) %>% 
  update_role(rowid, new_role = "indicator") %>% 
  step_timeseries_signature(date) %>% 
  step_rm(matches("(.iso)|(.xts)|(hour)|(minute)|(second)|(am.pm)")) %>% 
  step_normalize(date_index.num, date_year) %>% 
  step_dummy(all_nominal(), one_hot = TRUE)

recipe_base %>% prep() %>% juice() %>% glimpse()

recipe_wo_date <- recipe_base %>% 
  update_role(date, new_role = "indicator")

recipe_wo_date %>% prep() %>% juice() %>% glimpse()

# Modelling ----
# * Prophet ----
wflw_fit_prophet <- workflow() %>% 
  add_model(
    spec=prophet_reg(
      seasonality_daily  = TRUE, 
      seasonality_weekly = TRUE, 
      seasonality_yearly = TRUE
    ) %>% set_engine("prophet")
  ) %>% 
  add_recipe(recipe_base) %>% 
  fit(train_cleaned)

wflw_fit_prophet
# * XGBoost ----
wflw_fit_xgboost <- workflow() %>% 
  add_model(
    spec=boost_tree(mode = "regression") %>% set_engine("xgboost")
  ) %>% 
  add_recipe(recipe_wo_date) %>% 
  fit(train_cleaned)

wflw_fit_xgboost
# * PROPHET BOOST ----

wflw_fit_prophet_boost <- workflow() %>%
  add_model(
    spec = prophet_boost(
      seasonality_daily  = TRUE, 
      seasonality_weekly = TRUE, 
      seasonality_yearly = TRUE
    ) %>% 
      set_engine("prophet_xgboost")
  ) %>%
  add_recipe(recipe_base) %>%
  fit(train_cleaned)

wflw_fit_prophet_boost
# * SVM ----

wflw_fit_svm <- workflow() %>%
  add_model(
    spec = svm_rbf(mode = "regression") %>% set_engine("kernlab")
  ) %>%
  add_recipe(recipe_wo_date) %>%
  fit(train_cleaned)



# * RANDOM FOREST ----

wflw_fit_rf <- workflow() %>%
  add_model(
    spec = rand_forest(mode = "regression") %>% set_engine("ranger")
  ) %>%
  add_recipe(recipe_wo_date) %>%
  fit(train_cleaned)

# * NNET ----

wflw_fit_nnet <- workflow() %>%
  add_model(
    spec = mlp(mode = "regression") %>% set_engine("nnet")
  ) %>%
  add_recipe(recipe_wo_date) %>%
  fit(train_cleaned)

# * MARS ----

wflw_fit_earth <- workflow() %>%
  add_model(
    spec = mars(mode = "regression") %>% set_engine("earth")
  ) %>%
  add_recipe(recipe_wo_date) %>%
  fit(train_cleaned)


# * ACCURACY CHECK ----

submodels_1_tbl <- modeltime_table(
  wflw_fit_prophet,
  wflw_fit_xgboost,
  wflw_fit_prophet_boost,
  wflw_fit_svm,
  wflw_fit_rf,
  wflw_fit_nnet,
  wflw_fit_earth
)

submodels_1_tbl %>%
  modeltime_accuracy(testing(splits)) %>%
  arrange(rmse)

data_prepared_tbl_cleaned <- data_prepared_tbl %>% 
  group_by(store, item) %>% 
  mutate(sales = ts_clean_vec(sales, period = 180)) %>% 
  ungroup()

# plot forecast test value
submodels_1_tbl %>%
  modeltime_refit(
    train_cleaned
  ) %>%
  modeltime_forecast(
    new_data = testing(splits),
    actual_data = data_prepared_tbl_cleaned,
    keep_data = TRUE
  ) %>%
  group_by(store, item) %>%
  plot_modeltime_forecast()


# plot forecast value
submodels_1_tbl %>% 
  modeltime_refit(
    data_prepared_tbl_cleaned
  ) %>% 
  modeltime_calibrate(
    data_prepared_tbl_cleaned
  ) %>% 
  modeltime_forecast(
    new_data = future_tbl,
    actual_data = data_prepared_tbl_cleaned,
    keep_data = TRUE
  ) %>% 
  group_by(store, item) %>% 
  plot_modeltime_forecast()

# Pick earth, ranger and kern lab
# * Hyperparameter Tuning ----

# * Resample using Kfold ----

set.seed(123)
resamples_kfold <- train_cleaned %>% vfold_cv(v = 5)

resamples_kfold %>%
  tk_time_series_cv_plan() %>%
  plot_time_series_cv_plan(date, sales, .facet_ncol = 2)


# * XGBOOST TUNE ----
model_spec_rf_tune <- rand_forest(
  mode            = "regression", 
  mtry            = tune(),
  trees           = tune(),
  min_n           = tune()
) %>% set_engine("ranger")

wflw_spec_rf_tune <- wflw_fit_rf %>% 
  update_model(model_spec_rf_tune) %>% 
  update_recipe(recipe_wo_date)

# * TUning

tic()
set.seed(123)
tune_results_rf <- wflw_spec_rf_tune %>%
  tune_grid(
    resamples  = resamples_kfold,
    param_info = parameters(wflw_spec_rf_tune),
    grid = 10,
    control = control_grid(verbose = TRUE, allow_par = TRUE)
  )
toc()

# ** Results

tune_results_rf %>% show_best("rmse", n = Inf)


# ** Finalize

wflw_fit_rf_tuned <- wflw_spec_rf_tune %>%
  finalize_workflow(select_best(tune_results_rf, "rmse")) %>%
  fit(train_cleaned)

# * MARS TUNE ----
model_spec_earth_tune  <- mars(
  mode = "regression",
  num_terms = tune(),
  prod_degree = tune()
) %>% set_engine("earth")

wflw_spec_earth_tune <-  workflow() %>% 
  add_model(model_spec_earth_tune) %>% 
  add_recipe(recipe_wo_date)

# ** Tuning
tic()
set.seed(123)
tune_results_earth <- wflw_spec_earth_tune %>% 
  tune_grid(
    resamples = resamples_kfold,
    grid = 10,
    control = control_grid(allow_par = TRUE, verbose = TRUE)
  )
toc()

# * Results
tune_results_earth %>% show_best("rmse")

wflw_fit_earth_tuned <- wflw_fit_earth %>% 
  finalize_workflow(tune_results_earth %>% select_best("rmse")) %>% 
  fit(train_cleaned)

# * SVM TUNE ----
model_spec_svm_tune <- svm_rbf(
  mode = "regression",
  cost = tune(),
  rbf_sigma = tune(),
  margin = tune()
) %>% 
  set_engine("kernlab")

wflw_spec_svm_tune <-  workflow() %>% 
  add_model(model_spec_svm_tune) %>% 
  add_recipe(recipe_wo_date)

# ** Tuning
tic()
set.seed(123)
tune_results_svm <-  wflw_spec_svm_tune %>% 
  tune_grid(
    resamples = resamples_kfold,
    grid = 10,
    control = control_grid(allow_par = TRUE, verbose = TRUE)
  )
toc()

# ** Results
tune_results_svm %>% show_best("rmse")

wflw_fit_svm_tuned <- wflw_fit_svm %>% 
  finalize_workflow(tune_results_svm %>% select_best("rmse")) %>% 
  fit(train_cleaned)

# Evaluate Panel forecast ----

submodels_2_tbl <- modeltime_table(
  wflw_fit_rf_tuned,
  wflw_fit_earth_tuned,
  wflw_fit_svm_tuned
) %>% 
  update_model_description(1, "RANGER - Tuned") %>% 
  update_model_description(2, "Earth - Tuned") %>% 
  update_model_description(3, "SVM - Tuned") %>% 
  combine_modeltime_tables(submodels_1_tbl)

# * Calibration ----
calibration_tbl <- submodels_2_tbl %>%
  modeltime_calibrate(testing(splits))

# * Accuracy ----
calibration_tbl %>% 
  modeltime_accuracy() %>%
  arrange(rmse)

calibration_tbl %>% 
  modeltime_forecast(
    new_data = testing(splits),
    actual_data = data_prepared_tbl,
    keep_data = TRUE
  ) %>% 
  group_by(store, item) %>% 
  plot_modeltime_forecast(
    .facet_ncol = 2,
    .conf_interval_show = FALSE,
    .interactive = TRUE
  )

# Resample ----
# Test for model stability over time

# * Time series CV ----
resamples_tscv <- train_cleaned %>% 
  time_series_cv(
    assess = horizon,
    skip = 180,
    cumulative = TRUE,
    slice_limit = 5
  )

resamples_tscv %>%
  tk_time_series_cv_plan() %>%
  plot_time_series_cv_plan(date, sales)

# * Fitting resamples
model_tbl_tuned_resamples <-  submodels_2_tbl %>% 
  modeltime_fit_resamples(
    resamples = resamples_tscv,
    control = control_resamples(verbose = TRUE, allow_par = TRUE)
  )

model_tbl_tuned_resamples %>% 
  modeltime_resample_accuracy(
    metric_set = metric_set(rmse, rsq),
    summary_fns = list(mean = mean, sd = sd)
  ) %>% 
  arrange(rmse_mean)

# Ensemble models ----

# keeping earth - tuned, ranger (default params), SVM
submodels_2_ids_to_keep <- c(2, 8, 3)

ensemble_fit <-  submodels_2_tbl %>% 
  filter(.model_id %in% submodels_2_ids_to_keep) %>% 
  ensemble_average(type="median")

model_ensemble_tbl <- modeltime_table(
  ensemble_fit
)

model_ensemble_tbl %>% 
  modeltime_accuracy(testing(splits))

# * Forecast ----
forecast_ensemble_test_tbl <- model_ensemble_tbl %>% 
  modeltime_forecast(
    new_data = testing(splits),
    actual_data = data_prepared_tbl,
    keep_data = TRUE
  ) %>% 
  mutate(
    across(.cols = c(.value, sales), .fns = expm1)
  )

# Plot forecast values
forecast_ensemble_test_tbl %>% 
  group_by(store, item) %>% 
  plot_modeltime_forecast(
    .facet_ncol = 2
  )

# rmse by store and product
forecast_ensemble_test_tbl %>%
  filter(.key == "prediction") %>%
  select(store, item, .value, sales) %>%
  group_by(store, item) %>%
  summarize_accuracy_metrics(
    truth      = sales, 
    estimate   = .value,
    metric_set = metric_set(mae, rmse, rsq)
  )

# * Refit ----
data_prepared_tbl_cleaned <- data_prepared_tbl %>% 
  group_by(store, item) %>% 
  mutate(sales = ts_clean_vec(sales, period = 7)) %>% 
  ungroup()

model_ensemble_refit_tbl <-  model_ensemble_tbl %>% 
  modeltime_refit(
    data_prepared_tbl_cleaned
  ) %>% 
  modeltime_calibrate(
    data_prepared_tbl_cleaned
  )
  

model_ensemble_refit_tbl %>% 
  modeltime_forecast(
    new_data = future_tbl,
    actual_data = data_prepared_tbl_cleaned,
    keep_data = TRUE
  ) %>% 
  mutate(
    .value = expm1(.value),
    sales = expm1(sales),
    .conf_lo = expm1(.conf_lo),
    .conf_hi = expm1(.conf_hi)
  ) %>% 
  group_by(store, item) %>% 
  plot_modeltime_forecast(
    .facet_ncol = 2,
    .y_intercept = 0
  )

#to obtain rmse of test set in the future
forecast_ensemble_test_tbl %>% write_rds("models/forecast_ensemble_test_tbl.rds")
#to get model for future forecast
model_ensemble_refit_tbl %>% write_rds("models/ensemble_models.rds")

data_prepared_tbl %>% write_rds("data/data_prepare_tbl.rds")
data_prepared_tbl_cleaned %>% write_rds("data/data_prepared_tbl_cleaned.rds")
future_tbl %>% write_rds("data/future_tbl.rds")
