############################################################
# C-G07-final-regression.R
# Final Project - Introduction to Data Science (Option 1)
# Task: Regression (Linear Regression)
#
# Dataset Source (Kaggle):
# https://www.kaggle.com/datasets/chershi/house-price-prediction-dataset-2000-rows
#
# Dataset Access (Google Drive file used in code):
# https://drive.google.com/file/d/1yvhlZ4yS6yjJf2DuU9HRn6ZmE-73GJau/view?usp=sharing
#
# Goal:
# Predict house Price using Linear Regression.
#
# NOTE:
# All plots are SAVED using ggsave() into a folder named "plots"
############################################################

# ---------- Package setup (auto-install if missing) ----------
install_if_missing <- function(pkgs) {
  missing <- pkgs[!pkgs %in% installed.packages()[, "Package"]]
  if (length(missing) > 0) install.packages(missing, dependencies = TRUE)
}

install_if_missing(c("readr", "dplyr", "ggplot2", "caret", "janitor"))

library(readr)
library(dplyr)
library(ggplot2)
library(caret)
library(janitor)

set.seed(123)

# ---------- Plot saving setup ----------
PLOT_DIR <- "plots"
if (!dir.exists(PLOT_DIR)) dir.create(PLOT_DIR)

save_plot_safe <- function(filename, plot_obj, width = 8, height = 5, dpi = 300) {
  out_path <- file.path(PLOT_DIR, filename)
  ggsave(filename = out_path, plot = plot_obj, width = width, height = height, dpi = dpi)
  cat("Saved plot:", out_path, "\n")
}

# ---------- A) Data Collection ----------
FILE_ID <- "1yvhlZ4yS6yjJf2DuU9HRn6ZmE-73GJau"
DATA_URL <- paste0("https://drive.google.com/uc?export=download&id=", FILE_ID)

cat("Loading dataset from Google Drive...\n")
df_raw <- read_csv(DATA_URL, show_col_types = FALSE)
cat("Data loaded successfully.\n\n")

# ---------- Helper functions ----------
mode_value <- function(x) {
  x <- x[!is.na(x)]
  if (length(x) == 0) return(NA)
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

fit_preprocess <- function(train_df, target_name) {
  df <- train_df
  
  # Identify numeric vs categorical columns (excluding target from predictors)
  num_cols <- names(df)[sapply(df, is.numeric)]
  cat_cols <- names(df)[sapply(df, function(x) is.character(x) || is.factor(x))]
  
  # Remove target from lists if present
  num_cols <- setdiff(num_cols, target_name)
  cat_cols <- setdiff(cat_cols, target_name)
  
  # Imputation stats
  medians <- sapply(df[num_cols], function(x) median(x, na.rm = TRUE))
  modes <- sapply(df[cat_cols], mode_value)
  
  # Outlier caps (IQR rule) for numeric predictors
  caps <- lapply(df[num_cols], function(x) {
    q1 <- quantile(x, 0.25, na.rm = TRUE)
    q3 <- quantile(x, 0.75, na.rm = TRUE)
    iqr <- q3 - q1
    list(low = q1 - 1.5 * iqr, high = q3 + 1.5 * iqr)
  })
  
  # Store factor levels to keep train/test consistent
  factor_levels <- lapply(df[cat_cols], function(x) levels(as.factor(x)))
  
  list(
    num_cols = num_cols,
    cat_cols = cat_cols,
    medians = medians,
    modes = modes,
    caps = caps,
    factor_levels = factor_levels,
    target_name = target_name
  )
}

apply_preprocess <- function(df, pp) {
  out <- df
  
  # Numeric: median impute + outlier caps
  for (c in pp$num_cols) {
    if (c %in% names(out)) {
      out[[c]][is.na(out[[c]])] <- pp$medians[[c]]
      low <- pp$caps[[c]]$low
      high <- pp$caps[[c]]$high
      out[[c]] <- pmin(pmax(out[[c]], low), high)
    }
  }
  
  # Categorical: mode impute + factor levels align
  for (c in pp$cat_cols) {
    if (c %in% names(out)) {
      out[[c]][is.na(out[[c]])] <- pp$modes[[c]]
      out[[c]] <- as.factor(out[[c]])
      if (!is.null(pp$factor_levels[[c]])) {
        out[[c]] <- factor(out[[c]], levels = pp$factor_levels[[c]])
      }
    }
  }
  
  out
}

rmse <- function(y, yhat) sqrt(mean((y - yhat)^2))
mae  <- function(y, yhat) mean(abs(y - yhat))
r2   <- function(y, yhat) 1 - sum((y - yhat)^2) / sum((y - mean(y))^2)

# ---------- B) Data Understanding & Exploration ----------
cat("========== B) DATA UNDERSTANDING & EDA ==========\n")
cat("Raw shape (rows, cols): ", nrow(df_raw), ", ", ncol(df_raw), "\n\n")

# Clean column names (avoids spaces/special characters in formulas)
df <- df_raw %>% clean_names()

cat("Cleaned column names:\n")
print(names(df))

cat("\nData types (structure):\n")
print(str(df))

cat("\nSummary statistics:\n")
print(summary(df))

# Remove common ID column(s) if present
id_candidates <- c("id", "index")
for (idc in id_candidates) {
  if (idc %in% names(df)) df <- df %>% select(-all_of(idc))
}

# Define target
TARGET <- "price"
if (!(TARGET %in% names(df))) stop("ERROR: 'price' column not found after cleaning names. Check the dataset columns.")

cat("\nMissing values per column:\n")
print(sapply(df, function(x) sum(is.na(x))))

# Normalize common Yes/No style values and convert character -> factor
df <- df %>%
  mutate(across(where(is.character), ~ trimws(.))) %>%
  mutate(across(where(is.character),
                ~ ifelse(. %in% c("YES", "Yes", "yes"), "Yes",
                         ifelse(. %in% c("NO", "No", "no"), "No", .)))) %>%
  mutate(across(where(is.character), as.factor))

# ----- Plot 1: Price distribution (SAVE with ggsave) -----
p_price_hist <- ggplot(df, aes(x = .data[[TARGET]])) +
  geom_histogram(bins = 30) +
  labs(title = "Price Distribution (Regression Target)", x = "Price", y = "Frequency")

print(p_price_hist)
save_plot_safe("01_price_distribution.png", p_price_hist, width = 8, height = 5)

# ----- Plot 2: Correlation heatmap (numeric only) -----
num_for_corr <- df %>% select(where(is.numeric))
if (ncol(num_for_corr) >= 2) {
  corr <- cor(num_for_corr, use = "complete.obs")
  corr_df <- as.data.frame(as.table(corr))
  
  p_corr <- ggplot(corr_df, aes(Var1, Var2, fill = Freq)) +
    geom_tile() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    labs(title = "Correlation Heatmap (Numeric Features)", x = "", y = "")
  
  print(p_corr)
  save_plot_safe("02_correlation_heatmap.png", p_corr, width = 9, height = 7)
} else {
  cat("Skipped correlation heatmap: not enough numeric columns.\n")
}

# ---------- Feature engineering ----------
# 1) Area bins (categorical)
if ("area" %in% names(df) && is.numeric(df$area)) {
  df <- df %>% mutate(area_level = cut(
    area,
    breaks = quantile(area, probs = c(0, 0.33, 0.66, 1), na.rm = TRUE),
    include.lowest = TRUE,
    labels = c("low", "mid", "high")
  ))
}

# 2) Rooms per bathroom (numeric)
if (all(c("bedrooms", "bathrooms") %in% names(df))) {
  df <- df %>% mutate(rooms_per_bathroom = bedrooms / pmax(bathrooms, 1))
}

cat("\nEDA complete.\n\n")

# ---------- C) Data Preprocessing ----------
cat("========== C) PREPROCESSING ==========\n")

# Train/test split (80/20)
idx <- createDataPartition(df[[TARGET]], p = 0.80, list = FALSE)
train_df <- df[idx, ]
test_df  <- df[-idx, ]

# Fit preprocessing on TRAIN only (no leakage), apply to both
pp <- fit_preprocess(train_df, target_name = TARGET)
train_pp <- apply_preprocess(train_df, pp)
test_pp  <- apply_preprocess(test_df, pp)

# One-hot encode categorical variables
x_train <- model.matrix(as.formula(paste(TARGET, "~ .")), data = train_pp)[, -1, drop = FALSE]
y_train <- train_pp[[TARGET]]

x_test  <- model.matrix(as.formula(paste(TARGET, "~ .")), data = test_pp)[, -1, drop = FALSE]
y_test  <- test_pp[[TARGET]]

# Standardize predictors (fit on train, apply to test)
scaler <- preProcess(x_train, method = c("center", "scale"))
x_train <- predict(scaler, x_train)
x_test  <- predict(scaler, x_test)

cat("Preprocessing complete.\n\n")

# ---------- D) Modeling (Linear Regression) ----------
cat("========== D) MODELING (Linear Regression) ==========\n")

train_model_df <- data.frame(price = y_train, x_train)
test_model_df  <- data.frame(price = y_test, x_test)

lm_fit <- lm(price ~ ., data = train_model_df)
cat("Linear Regression model trained.\n\n")

cat("Model summary (coefficients show direction/magnitude, p-values show significance):\n")
print(summary(lm_fit))

# ---------- E) Model Evaluation & Interpretation ----------
cat("\n========== E) EVALUATION & INTERPRETATION ==========\n")

pred <- predict(lm_fit, newdata = test_model_df)

RMSE <- rmse(y_test, pred)
MAE  <- mae(y_test, pred)
R2   <- r2(y_test, pred)

cat("\n--- Regression Metrics on Test Set ---\n")
cat("RMSE:", round(RMSE, 4), "\n")
cat("MAE :", round(MAE, 4), "\n")
cat("R^2 :", round(R2, 4), "\n\n")

# ----- Plot 3: Predicted vs Actual (SAVE) -----
plot_df <- data.frame(actual = y_test, predicted = pred)

p_pred_actual <- ggplot(plot_df, aes(x = actual, y = predicted)) +
  geom_point(alpha = 0.5) +
  geom_abline(slope = 1, intercept = 0) +
  labs(title = "Predicted vs Actual Price", x = "Actual Price", y = "Predicted Price")

print(p_pred_actual)
save_plot_safe("03_predicted_vs_actual.png", p_pred_actual, width = 7.5, height = 5.5)

# ----- Plot 4: Residuals vs Fitted (SAVE) -----
resid_df <- data.frame(
  fitted = pred,
  residuals = y_test - pred
)

p_resid_fitted <- ggplot(resid_df, aes(x = fitted, y = residuals)) +
  geom_point(alpha = 0.5) +
  geom_hline(yintercept = 0) +
  labs(title = "Residuals vs Fitted", x = "Fitted (Predicted Price)", y = "Residuals (Actual - Predicted)")

print(p_resid_fitted)
save_plot_safe("04_residuals_vs_fitted.png", p_resid_fitted, width = 7.5, height = 5.5)

# ----- Plot 5: Q-Q Plot of Residuals (SAVE) -----
p_qq <- ggplot(resid_df, aes(sample = residuals)) +
  stat_qq() +
  stat_qq_line() +
  labs(title = "Q-Q Plot of Residuals", x = "Theoretical Quantiles", y = "Sample Quantiles")

print(p_qq)
save_plot_safe("05_qq_plot_residuals.png", p_qq, width = 7.5, height = 5.5)

cat("Interpretation (brief):\n")
cat("- RMSE and MAE measure average prediction error in the same unit as Price.\n")
cat("- R^2 shows how much variance in Price is explained by the model (closer to 1 is better).\n")
cat("- Residual plots help check linear regression assumptions (random scatter is desirable).\n\n")

cat("All plots have been saved in the folder: ", PLOT_DIR, "\n")
