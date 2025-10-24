# install.R
# --------------------------------------------------------------------
# Gridirony: R package setup script
# --------------------------------------------------------------------
# Run this file once to install all R dependencies needed for modeling.
# Usage:
#   Rscript install.R
# --------------------------------------------------------------------

# Recommended CRAN mirror
options(repos = c(CRAN = "https://cloud.r-project.org"))

# Vector of required packages
required_pkgs <- c(
  "tidyverse",   # Data wrangling, visualization, piping
  "arrow",       # Read/write Parquet files shared with Python
  "janitor",     # Clean column names
  "glmnet",      # Regularized linear/logistic regression
  "yardstick",   # Model evaluation metrics
  "pROC",        # ROC/AUC curves
  "MLmetrics",   # LogLoss, Brier Score, etc.
  "ggplot2",     # Visualization
  "isotone"      # Calibration tools (isotonic regression)
)

# Identify any missing packages
to_install <- setdiff(required_pkgs, installed.packages()[, "Package"])

if (length(to_install) > 0) {
  message("Installing missing packages: ", paste(to_install, collapse = ", "))
  install.packages(to_install, dependencies = TRUE)
} else {
  message("All required R packages are already installed.")
}

message("✅ R environment setup complete.")
