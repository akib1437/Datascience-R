############################################################
# C-G01-final-regression.R
# Final Project - Introduction to Data Science (Option 1)
# Task: Regression (Linear Regression)
#
# Dataset Source (Kaggle):
# https://www.kaggle.com/datasets/yasserh/wine-quality-dataset
# Dataset Access (Google Drive file used in code):
# https://drive.google.com/file/d/1pEa9dRR7dAKu5lMiF_XfdyTIEb5ZXz0i/view
#
# Goal:
# Predict numeric wine quality score using Linear Regression.
############################################################

# ---------- Package setup (auto-install if missing) ----------
install_if_missing <- function(pkgs) {
  missing <- pkgs[!pkgs %in% installed.packages()[, "Package"]]
  if (length(missing) > 0) install.packages(missing, dependencies = TRUE)
}
install_if_missing(c("readr", "dplyr", "ggplot2", "caret"))

library(readr)
library(dplyr)
library(ggplot2)
library(caret)

set.seed(123)

# ---------- A) Data Collection ----------
FILE_ID <- "1pEa9dRR7dAKu5lMiF_XfdyTIEb5ZXz0i"
DATA_URL <- paste0("https://drive.google.com/uc?export=download&id=", FILE_ID)

cat("Loading dataset from Google Drive...\n")
df_raw <- read_csv(DATA_URL, show_col_types = FALSE)
cat("\nData loaded successfully.\n")

# ---------- Helper functions ----------
mode_value <- function(x) {
  x <- x[!is.na(x)]
  if (length(x) == 0) return(NA)
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

fit_preprocess <- function(train_df, target_name) {
  df <- train_df
  num_cols <- names(df)[sapply(df, is.numeric)]
  cat_cols <- names(df)[sapply(df, function(x) is.character(x) || is.factor(x))]
  
  medians <- sapply(df[num_cols], function(x) median(x, na.rm = TRUE))
  modes <- sapply(df[cat_cols], mode_value)
  
  caps <- lapply(df[num_cols], function(x) {
    q1 <- quantile(x, 0.25, na.rm = TRUE)
    q3 <- quantile(x, 0.75, na.rm = TRUE)
    iqr <- q3 - q1
    list(low = q1 - 1.5 * iqr, high = q3 + 1.5 * iqr)
  })
  
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
  
  # Impute
  for (c in pp$num_cols) {
    if (c %in% names(out)) out[[c]][is.na(out[[c]])] <- pp$medians[[c]]
  }
  for (c in pp$cat_cols) {
    if (c %in% names(out)) {
      out[[c]][is.na(out[[c]])] <- pp$modes[[c]]
      out[[c]] <- as.factor(out[[c]])
      if (!is.null(pp$factor_levels[[c]])) {
        out[[c]] <- factor(out[[c]], levels = pp$factor_levels[[c]])
      }
    }
  }
  
  # Outlier caps
  for (c in pp$num_cols) {
    if (c %in% names(out)) {
      low <- pp$caps[[c]]$low
      high <- pp$caps[[c]]$high
      out[[c]] <- pmin(pmax(out[[c]], low), high)
    }
  }
  
  out
}

rmse <- function(y, yhat) sqrt(mean((y - yhat)^2))
mae  <- function(y, yhat) mean(abs(y - yhat))
r2   <- function(y, yhat) 1 - sum((y - yhat)^2) / sum((y - mean(y))^2)

# ---------- B) Data Understanding & Exploration ----------
cat("\n========== B) DATA UNDERSTANDING & EDA ==========\n")
cat("Shape (rows, cols): ", nrow(df_raw), ", ", ncol(df_raw), "\n\n")

cat("Data types:\n")
print(str(df_raw))

cat("\nSummary statistics:\n")
print(summary(df_raw))

df <- df_raw
if ("Id" %in% names(df)) df <- df %>% select(-Id)

if (!("quality" %in% names(df))) stop("ERROR: 'quality' column not found in dataset.")

cat("\nMissing values per column:\n")
print(sapply(df, function(x) sum(is.na(x))))

# Visuals
ggplot(df, aes(x = quality)) +
  geom_histogram(bins = 10) +
  labs(title = "Quality Distribution (Regression Target)", x = "Quality", y = "Frequency")

# Feature engineering to ensure categorical + numeric exist (guideline requirement)
if ("alcohol" %in% names(df)) {
  df <- df %>% mutate(alcohol_level = cut(alcohol,
                                          breaks = quantile(alcohol, probs = c(0, .33, .66, 1), na.rm = TRUE),
                                          include.lowest = TRUE, labels = c("low", "mid", "high")))
}
if ("sulphates" %in% names(df)) {
  df <- df %>% mutate(sulphates_level = cut(sulphates,
                                            breaks = quantile(sulphates, probs = c(0, .33, .66, 1), na.rm = TRUE),
                                            include.lowest = TRUE, labels = c("low", "mid", "high")))
}

# Quick correlation plot (numeric)
num_for_corr <- df %>% select(where(is.numeric))
if (ncol(num_for_corr) >= 2) {
  corr <- cor(num_for_corr, use = "complete.obs")
  corr_df <- as.data.frame(as.table(corr))
  ggplot(corr_df, aes(Var1, Var2, fill = Freq)) +
    geom_tile() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    labs(title = "Correlation Heatmap (Numeric Features)", x = "", y = "")
}

# ---------- C) Data Preprocessing ----------
cat("\n========== C) PREPROCESSING ==========\n")

# Train/test split
idx <- createDataPartition(df$quality, p = 0.80, list = FALSE)
train_df <- df[idx, ]
test_df  <- df[-idx, ]

pp <- fit_preprocess(train_df, target_name = "quality")
train_pp <- apply_preprocess(train_df, pp)
test_pp  <- apply_preprocess(test_df, pp)

# Optional transformation to reduce skewness (safe log1p for non-negative numeric predictors)
# We apply to a few common skewed features if they exist and are >= 0.
skew_candidates <- intersect(
  c("residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide"),
  names(train_pp)
)
for (col in skew_candidates) {
  if (is.numeric(train_pp[[col]]) && min(train_pp[[col]], na.rm = TRUE) >= 0) {
    train_pp[[col]] <- log1p(train_pp[[col]])
    test_pp[[col]]  <- log1p(test_pp[[col]])
  }
}

# Encode categorical variables automatically (one-hot)
# model.matrix will create dummy variables for factor columns
x_train <- model.matrix(quality ~ ., data = train_pp)[, -1, drop = FALSE]
y_train <- train_pp$quality

x_test  <- model.matrix(quality ~ ., data = test_pp)[, -1, drop = FALSE]
y_test  <- test_pp$quality

# Standardize predictors (fit on train, apply to test)
scaler <- preProcess(x_train, method = c("center", "scale"))
x_train <- predict(scaler, x_train)
x_test  <- predict(scaler, x_test)

cat("Preprocessing complete.\n")

# ---------- D) Modeling (Linear Regression) ----------
cat("\n========== D) MODELING (Linear Regression) ==========\n")

train_model_df <- data.frame(quality = y_train, x_train)
test_model_df  <- data.frame(quality = y_test, x_test)

lm_fit <- lm(quality ~ ., data = train_model_df)
cat("Linear Regression model trained.\n")

cat("\nModel summary (key coefficients may indicate direction of impact):\n")
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
cat("R^2 :", round(R2, 4), "\n")

# Plot predicted vs actual
plot_df <- data.frame(actual = y_test, predicted = pred)

ggplot(plot_df, aes(x = actual, y = predicted)) +
  geom_point(alpha = 0.5) +
  geom_abline(slope = 1, intercept = 0) +
  labs(title = "Predicted vs Actual Quality", x = "Actual", y = "Predicted")

cat("\nInterpretation (brief):\n")
cat("- Lower RMSE/MAE indicates better prediction accuracy.\n")
cat("- Higher R^2 indicates the model explains more variance in quality.\n")
