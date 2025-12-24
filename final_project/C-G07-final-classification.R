############################################################
# C-G01-final-classification.R
# Final Project - Introduction to Data Science (Option 1)
# Task: Classification (Decision Tree)
#
# Dataset Source (Kaggle):
# https://www.kaggle.com/datasets/yasserh/wine-quality-dataset
# Dataset Access (Google Drive file used in code):
# https://drive.google.com/file/d/1pEa9dRR7dAKu5lMiF_XfdyTIEb5ZXz0i/view
#
# Goal:
# Predict wine quality class (high vs low) using a Decision Tree.
############################################################

# ---------- Package setup (auto-install if missing) ----------
install_if_missing <- function(pkgs) {
  missing <- pkgs[!pkgs %in% installed.packages()[, "Package"]]
  if (length(missing) > 0) install.packages(missing, dependencies = TRUE)
}
install_if_missing(c("readr", "dplyr", "ggplot2", "caret", "rpart", "rpart.plot"))

library(readr)
library(dplyr)
library(ggplot2)
library(caret)
library(rpart)
library(rpart.plot)

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
  
  # Identify columns
  num_cols <- names(df)[sapply(df, is.numeric)]
  cat_cols <- names(df)[sapply(df, function(x) is.character(x) || is.factor(x))]
  
  # Medians / modes for imputation
  medians <- sapply(df[num_cols], function(x) median(x, na.rm = TRUE))
  modes <- sapply(df[cat_cols], mode_value)
  
  # IQR caps for outlier handling (winsorization)
  caps <- lapply(df[num_cols], function(x) {
    q1 <- quantile(x, 0.25, na.rm = TRUE)
    q3 <- quantile(x, 0.75, na.rm = TRUE)
    iqr <- q3 - q1
    list(low = q1 - 1.5 * iqr, high = q3 + 1.5 * iqr)
  })
  
  # Factor levels for consistent encoding
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
  
  # 1) Impute missing values
  for (c in pp$num_cols) {
    if (c %in% names(out)) out[[c]][is.na(out[[c]])] <- pp$medians[[c]]
  }
  for (c in pp$cat_cols) {
    if (c %in% names(out)) {
      out[[c]][is.na(out[[c]])] <- pp$modes[[c]]
      out[[c]] <- as.factor(out[[c]])
      # align levels
      if (!is.null(pp$factor_levels[[c]])) {
        out[[c]] <- factor(out[[c]], levels = pp$factor_levels[[c]])
      }
    }
  }
  
  # 2) Outlier handling (cap numeric features)
  for (c in pp$num_cols) {
    if (c %in% names(out)) {
      low <- pp$caps[[c]]$low
      high <- pp$caps[[c]]$high
      out[[c]] <- pmin(pmax(out[[c]], low), high)
    }
  }
  
  out
}

# ---------- B) Data Understanding & Exploration ----------
cat("\n========== B) DATA UNDERSTANDING & EDA ==========\n")
cat("Shape (rows, cols): ", nrow(df_raw), ", ", ncol(df_raw), "\n\n")

cat("Data types:\n")
print(str(df_raw))

cat("\nSummary statistics:\n")
print(summary(df_raw))

# Drop ID if present (not predictive)
df <- df_raw
if ("Id" %in% names(df)) df <- df %>% select(-Id)

# Feature engineering to ensure categorical + numeric features exist (guideline requirement)
# Create categorical bins from numeric columns (robust: only if columns exist)
if ("alcohol" %in% names(df)) {
  df <- df %>% mutate(alcohol_level = cut(alcohol,
                                          breaks = quantile(alcohol, probs = c(0, .33, .66, 1), na.rm = TRUE),
                                          include.lowest = TRUE, labels = c("low", "mid", "high")))
}
if ("volatile acidity" %in% names(df)) {
  df <- df %>% mutate(va_level = cut(`volatile acidity`,
                                     breaks = quantile(`volatile acidity`, probs = c(0, .33, .66, 1), na.rm = TRUE),
                                     include.lowest = TRUE, labels = c("low", "mid", "high")))
}

# Basic missing values count
cat("\nMissing values per column:\n")
print(sapply(df, function(x) sum(is.na(x))))

# Visuals: target distribution (quality)
if (!("quality" %in% names(df))) stop("ERROR: 'quality' column not found in dataset.")
ggplot(df, aes(x = quality)) +
  geom_bar() +
  labs(title = "Distribution of Quality", x = "Quality", y = "Count")

# Correlation heatmap (numeric only)
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

# Classification target: high vs low (common split; can adjust if your instructor wants)
df <- df %>%
  mutate(quality_class = factor(ifelse(quality >= 6, "high", "low"))) %>%
  select(-quality)

cat("Target classes distribution:\n")
print(table(df$quality_class))

# Train/test split (stratified)
idx <- createDataPartition(df$quality_class, p = 0.80, list = FALSE)
train_df <- df[idx, ]
test_df  <- df[-idx, ]

# Fit preprocessing on train only (prevents leakage)
pp <- fit_preprocess(train_df, target_name = "quality_class")
train_pp <- apply_preprocess(train_df, pp)
test_pp  <- apply_preprocess(test_df, pp)

# Identify numeric predictors for scaling (exclude target)
num_cols <- pp$num_cols
num_cols <- setdiff(num_cols, "quality_class")

# Scale/standardize numeric predictors (fit on train, apply to test)
preproc_scaler <- preProcess(train_pp[, num_cols, drop = FALSE], method = c("center", "scale"))
train_pp[, num_cols] <- predict(preproc_scaler, train_pp[, num_cols, drop = FALSE])
test_pp[, num_cols]  <- predict(preproc_scaler, test_pp[, num_cols, drop = FALSE])

cat("Preprocessing complete.\n")

# ---------- D) Modeling (Decision Tree) ----------
cat("\n========== D) MODELING (Decision Tree) ==========\n")

tree_model <- rpart(quality_class ~ ., data = train_pp, method = "class",
                    control = rpart.control(cp = 0.01))

cat("Decision Tree trained.\n")

# Plot tree
rpart.plot(tree_model, main = "Decision Tree for Wine Quality Class")

# ---------- E) Evaluation & Interpretation ----------
cat("\n========== E) EVALUATION & INTERPRETATION ==========\n")

pred_class <- predict(tree_model, newdata = test_pp, type = "class")
cm <- confusionMatrix(pred_class, test_pp$quality_class, positive = "high")

print(cm)

# Extract Precision, Recall, F1
precision <- cm$byClass["Precision"]
recall    <- cm$byClass["Recall"]
f1        <- cm$byClass["F1"]
accuracy  <- cm$overall["Accuracy"]

cat("\n--- Key Metrics ---\n")
cat("Accuracy :", round(accuracy, 4), "\n")
cat("Precision:", round(precision, 4), "\n")
cat("Recall   :", round(recall, 4), "\n")
cat("F1-score :", round(f1, 4), "\n")

cat("\nInterpretation (brief):\n")
cat("- A higher Precision means fewer 'high' false positives.\n")
cat("- A higher Recall means fewer missed 'high' wines.\n")
cat("- F1 balances Precision and Recall.\n")
