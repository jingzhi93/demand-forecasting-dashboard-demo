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

#visualization
library(ggplot2)

# * Parallel Processing ----

registerDoFuture()
n_cores <- parallel::detectCores()
plan(
  strategy = cluster,
  workers  = parallel::makeCluster(n_cores)
)

# plan(sequential)
data_raw_tbl <- read_csv("data/train.csv")

#eda
data_raw_tbl %>% skim()
#no missing values

data_raw_clean_tbl <- data_raw_tbl %>% 
  mutate(store = as.factor(store)) %>% 
  mutate(item = as.factor(item))

# total number of items in a store ----
data_raw_clean_tbl %>% 
  group_by(store) %>% 
  summarise(count_item = length(unique(item)))

# summarized statistics for each sales by stores ----
data_raw_clean_tbl %>% 
  group_by(store) %>% 
  summarise(
    sales_count = n(),
    total_sales = sum(sales),
    mean = mean(sales),
    median = median(sales),
    std = sd(sales),
    min = min(sales),
    max = max(sales)
  )

# summarized statistics for each sales by items ----
data_raw_clean_tbl %>% 
  group_by(item) %>% 
  summarise(
    sales_count = n(),
    total_sales = sum(sales),
    mean = mean(sales),
    median = median(sales),
    std = sd(sales),
    min = min(sales),
    max = max(sales)
  )

# histogram of store sales ----
data_raw_clean_tbl %>% 
  ggplot()+
  geom_histogram(aes(x=sales), binwidth=0.5)+
  facet_wrap(~store)

# Sales distribution of store 1 ----
data_raw_clean_tbl %>% 
  group_by(store, item) %>% 
  filter(store == 1) %>% 
  plot_time_series(
    date, sales,
    .facet_ncol = 5,
    .interactive = FALSE,
    .smooth = FALSE
  )

data_raw_clean_tbl %>% 
  plot_acf_diagnostics(date, sales)

plan(sequential)

prep_raw_data <- data_raw_clean_tbl %>% 
  mutate(store = as.numeric(store), item = as.numeric(item)) %>% 
  filter(item < 3) %>% 
  filter(store < 4) %>% 
  mutate(store = as.factor(store), item = as.factor(item)) 

prep_raw_data %>% 
  group_by(store, item) %>% 
  plot_time_series(date, sales, .facet_ncol = 3)

prep_raw_data %>% 
  plot_seasonal_diagnostics(date, sales)

prep_raw_data%>% 
  write_rds("data/cleaned_raw_partial_tbl.rds")

data_raw_clean_tbl %>% 
  write_rds("data/cleaned_raw_full_tbl.rds")
