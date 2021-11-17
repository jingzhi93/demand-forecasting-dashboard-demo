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

# registerDoFuture()
# n_cores <- parallel::detectCores()
# plan(
#   strategy = cluster,
#   workers  = parallel::makeCluster(n_cores)
# )


model_ensemble_refit_tbl <-  read_rds("models/ensemble_models.rds")
data_prepared_tbl <-  read_rds("data/data_prepare_tbl.rds")
data_prepared_tbl_cleaned  <-  read_rds("data/data_prepared_tbl_cleaned.rds")
future_tbl <-  read_rds("data/future_tbl.rds")


forecast_demand_chart <-  model_ensemble_refit_tbl %>% 
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
