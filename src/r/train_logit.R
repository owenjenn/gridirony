# src/r/train_logit.R
library(tidyverse)
library(arrow)
library(janitor)
library(glmnet)
library(yardstick)
library(MLmetrics)
library(pROC)
library(isotone)   # optional calibration

df <- arrow::read_parquet("data/features/model_df.parquet") %>% clean_names()

# keep only rows with labels
df <- df %>% filter(!is.na(home_win))

# time-based split (adjust cutoffs as you like)
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
x_train <- to_mm(train)
x_val   <- to_mm(val)
y_train <- train$home_win
y_val   <- val$home_win

# elastic-net logistic
set.seed(42)
cvfit <- cv.glmnet(x_train, y_train, family="binomial", alpha=0.5, nfolds=10)
fit   <- glmnet(x_train, y_train, family="binomial", alpha=0.5, lambda=cvfit$lambda.min)

# validation metrics
val$pred_prob <- as.numeric(predict(fit, x_val, type="response"))
brier   <- mean((val$pred_prob - y_val)^2)
logloss <- MLmetrics::LogLoss(val$pred_prob, y_val)
auc     <- pROC::roc(y_val, val$pred_prob)$auc

cat(paste0("VAL  Brier=", round(brier,4),
           "  LogLoss=", round(logloss,4),
           "  ROC-AUC=", round(auc,4), "\n"))

# simple isotonic calibration (optional)
# iso <- isoreg(val$pred_prob, y_val) # then use to map probs if desired

# refit on train+val, evaluate on test
x_trv <- to_mm(bind_rows(train %>% mutate(split="tr"), val %>% mutate(split="va")))
y_trv <- c(train$home_win, val$home_win)
cv2   <- cv.glmnet(x_trv, y_trv, family="binomial", alpha=0.5, nfolds=10)
fit2  <- glmnet(x_trv, y_trv, family="binomial", alpha=0.5, lambda=cv2$lambda.min)

x_test <- to_mm(test)
test$pred_prob <- as.numeric(predict(fit2, x_test, type="response"))
brier_t   <- mean((test$pred_prob - test$home_win)^2, na.rm=TRUE)
logloss_t <- MLmetrics::LogLoss(test$pred_prob, test$home_win)
auc_t     <- pROC::roc(test$home_win, test$pred_prob)$auc
cat(paste0("TEST Brier=", round(brier_t,4),
           "  LogLoss=", round(logloss_t,4),
           "  ROC-AUC=", round(auc_t,4), "\n"))

# save model
saveRDS(fit2, "artifacts/logit_glmnet.rds")
write_csv(test %>% select(game_id, season, week, pred_prob, home_win),
          "reports/logit_test_predictions.csv")
