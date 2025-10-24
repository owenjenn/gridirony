# src/r/train_linear.R
library(tidyverse)
library(arrow)
library(janitor)
library(glmnet)
library(yardstick)

df <- arrow::read_parquet("data/features/model_df.parquet") %>% clean_names()
df <- df %>% filter(!is.na(margin))

train <- df %>% filter(season <= 2019)
val   <- df %>% filter(season == 2020)
test  <- df %>% filter(season >= 2021)

# pick feature columns safely (include only those present)
roll_cols <- df %>%
  select(matches("(home|away)_(points_|pass_yards_|rush_yards_|to_diff_).*"), .preserve = TRUE) %>%
  select(where(is.numeric)) %>%
  colnames()

opt_cols <- intersect(c("spread_close", "total_close", "is_divisional"), colnames(df))

x_cols <- c(roll_cols, opt_cols)

if (length(x_cols) == 0) {
  stop("No feature columns found. Check that Python created rolling features in data/features/model_df.parquet")
}

to_mm <- function(d) model.matrix(~ . , data = d %>% select(all_of(x_cols)))[,-1]
x_train <- to_mm(train); y_train <- train$margin
x_val   <- to_mm(val);   y_val   <- val$margin

set.seed(42)
cvfit <- cv.glmnet(x_train, y_train, alpha=0.5)
fit   <- glmnet(x_train, y_train, alpha=0.5, lambda=cvfit$lambda.min)

val$pred_margin <- as.numeric(predict(fit, x_val))
mae  <- mean(abs(val$pred_margin - y_val))
rmse <- yardstick::rmse_vec(truth=y_val, estimate=val$pred_margin)
cat(paste0("VAL  MAE=", round(mae,3), "  RMSE=", round(rmse,3), "\n"))

# refit on train+val, score test
x_trv <- to_mm(bind_rows(train, val)); y_trv <- c(train$margin, val$margin)
cv2 <- cv.glmnet(x_trv, y_trv, alpha=0.5)
fit2 <- glmnet(x_trv, y_trv, alpha=0.5, lambda=cv2$lambda.min)

x_test <- to_mm(test)
test$pred_margin <- as.numeric(predict(fit2, x_test))
mae_t  <- mean(abs(test$pred_margin - test$margin))
rmse_t <- yardstick::rmse_vec(truth=test$margin, estimate=test$pred_margin)
cat(paste0("TEST MAE=", round(mae_t,3), "  RMSE=", round(rmse_t,3), "\n"))

saveRDS(fit2, "artifacts/linear_glmnet.rds")
write_csv(test %>% select(game_id, season, week, pred_margin, margin),
          "reports/linear_test_predictions.csv")
