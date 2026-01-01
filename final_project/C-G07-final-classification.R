############################################################
# C-G07-final-classification-ai-impact.R
# Final Project - Introduction to Data Science (Option 1)
# Task: Classification (Decision Tree)
#
# Dataset Source (Kaggle):
# https://www.kaggle.com/datasets/ankushnarwade/ai-impact-on-student-performance
#
# Dataset Access (Google Drive file used in code):
# https://drive.google.com/file/d/1leIKtC7S1MJVfCParSxCU4e8ZzRLJyNl/view?usp=sharing
#
# Goal:
# Predict student Performance Category using a Decision Tree.
# Evaluation: Confusion Matrix, Accuracy, Precision, Recall, F1-score
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

# ---------- Plot saving options ----------
SAVE_PLOTS <- TRUE
PLOT_DIR <- "plots"

if (SAVE_PLOTS) {
  dir.create(PLOT_DIR, showWarnings = FALSE)
}

save_plot <- function(p, filename, width = 8, height = 5, dpi = 300) {
  if (SAVE_PLOTS) {
    ggsave(
      filename = file.path(PLOT_DIR, filename),
      plot = p,
      width = width,
      height = height,
      dpi = dpi
    )
    cat("Saved plot:", file.path(PLOT_DIR, filename), "\n")
  }
}

save_tree_png <- function(model, filename, main_title = "Decision Tree", width_px = 1400, height_px = 900, res = 150) {
  if (SAVE_PLOTS) {
    png(file.path(PLOT_DIR, filename), width = width_px, height = height_px, res = res)
    rpart.plot(model, main = main_title, cex = 0.7)
    dev.off()
    cat("Saved tree plot:", file.path(PLOT_DIR, filename), "\n")
  }
}

# ---------- Column name cleaning (robust for spaces/slashes) ----------
clean_names_base <- function(nms) {
  nms2 <- tolower(nms)
  nms2 <- gsub("[^a-z0-9]+", "_", nms2)
  nms2 <- gsub("^_+|_+$", "", nms2)
  nms2
}

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
  
  # exclude target
  num_cols <- setdiff(num_cols, target_name)
  cat_cols <- setdiff(cat_cols, target_name)
  
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
  
  # Ensure character -> factor for categorical columns
  for (c in pp$cat_cols) {
    if (c %in% names(out)) out[[c]] <- as.factor(out[[c]])
  }
  
  # 1) Impute missing values
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

# OPTIONAL: drop extremely high-cardinality categorical predictors
drop_high_cardinality <- function(data, target, max_levels = 50, max_prop_unique = 0.50) {
  preds <- setdiff(names(data), target)
  to_drop <- c()
  n <- nrow(data)
  
  for (c in preds) {
    if (is.factor(data[[c]]) || is.character(data[[c]])) {
      lvls <- length(unique(data[[c]]))
      prop_unique <- lvls / max(1, n)
      if (lvls > max_levels || prop_unique > max_prop_unique) {
        to_drop <- c(to_drop, c)
      }
    }
  }
  
  if (length(to_drop) > 0) {
    cat("\nDropping high-cardinality predictors to reduce overfitting:\n")
    print(to_drop)
    data <- data %>% select(-all_of(to_drop))
  } else {
    cat("\nNo high-cardinality predictors dropped.\n")
  }
  
  data
}

# Create quantile bins for a numeric column if it exists
add_quantile_bins <- function(data, colname, newname) {
  if (colname %in% names(data) && is.numeric(data[[colname]])) {
    qs <- quantile(data[[colname]], probs = c(0, .33, .66, 1), na.rm = TRUE)
    if (length(unique(qs)) >= 3) {
      data[[newname]] <- cut(
        data[[colname]],
        breaks = qs,
        include.lowest = TRUE,
        labels = c("low", "mid", "high")
      )
    }
  }
  data
}

# ---------- A) Data Collection ----------
FILE_ID <- "1leIKtC7S1MJVfCParSxCU4e8ZzRLJyNl"
DATA_URL <- paste0("https://drive.google.com/uc?export=download&id=", FILE_ID)

cat("Loading dataset from Google Drive...\n")
df_raw <- read_csv(DATA_URL, show_col_types = FALSE)
cat("Data loaded successfully.\n\n")

# Clean column names
names(df_raw) <- clean_names_base(names(df_raw))

# ---------- B) Data Understanding & EDA ----------
cat("========== B) DATA UNDERSTANDING & EDA ==========\n")
cat("Shape (rows, cols): ", nrow(df_raw), ", ", ncol(df_raw), "\n\n")

cat("Data types (structure):\n")
print(str(df_raw))

cat("\nSummary statistics:\n")
print(summary(df_raw))

df <- df_raw

# Ensure required target exists
TARGET <- "performance_category"
if (!(TARGET %in% names(df))) {
  cat("\nERROR: Target column 'performance_category' not found after cleaning names.\n")
  cat("Available columns are:\n")
  print(names(df))
  stop("Please confirm the CSV has a column named performance_category.")
}

# Missing values per column
cat("\nMissing values per column:\n")
print(sapply(df, function(x) sum(is.na(x))))

# Convert text-like columns to character (then factor later)
char_like <- names(df)[sapply(df, function(x) is.character(x) || is.factor(x))]
for (c in char_like) df[[c]] <- as.character(df[[c]])

# Target as factor
df[[TARGET]] <- as.factor(df[[TARGET]])

# ---- Drop ID + leakage columns ----
DROP_COLS <- intersect(c("student_id", "final_score", "passed"), names(df))
DROP_COLS <- setdiff(DROP_COLS, TARGET)

if (length(DROP_COLS) > 0) {
  cat("\nDropping columns not used for prediction (ID/leakage):\n")
  print(DROP_COLS)
  df <- df %>% select(-all_of(DROP_COLS))
} else {
  cat("\nNo ID/leakage columns found to drop.\n")
}

cat("\nTarget classes distribution:\n")
print(table(df[[TARGET]]))

# Target distribution plot
p_target <- ggplot(df, aes(x = .data[[TARGET]])) +
  geom_bar() +
  labs(
    title = "Target Distribution: Performance Category",
    x = "Performance Category",
    y = "Count"
  ) +
  theme(axis.text.x = element_text(angle = 25, hjust = 1))
print(p_target)
save_plot(p_target, "01_target_distribution_performance_category.png")

# Numeric distributions (top 5 numeric columns, generic)
num_cols_all <- names(df)[sapply(df, is.numeric)]
num_cols_all <- setdiff(num_cols_all, TARGET)

if (length(num_cols_all) > 0) {
  top_num <- head(num_cols_all, 5)
  
  for (nc in top_num) {
    p_hist <- ggplot(df, aes(x = .data[[nc]])) +
      geom_histogram(bins = 25) +
      labs(title = paste("Distribution of", nc), x = nc, y = "Count")
    print(p_hist)
    save_plot(p_hist, paste0("hist_", nc, ".png"))
    
    p_box <- ggplot(df, aes(x = .data[[TARGET]], y = .data[[nc]])) +
      geom_boxplot() +
      labs(title = paste("Boxplot:", nc, "by Performance Category"),
           x = "Performance Category", y = nc) +
      theme(axis.text.x = element_text(angle = 25, hjust = 1))
    print(p_box)
    save_plot(p_box, paste0("box_", nc, "_by_performance_category.png"))
  }
  
  num_for_corr <- df %>% select(where(is.numeric))
  if (ncol(num_for_corr) >= 2) {
    corr <- cor(num_for_corr, use = "complete.obs")
    corr_df <- as.data.frame(as.table(corr))
    p_corr <- ggplot(corr_df, aes(Var1, Var2, fill = Freq)) +
      geom_tile() +
      theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
      labs(title = "Correlation Heatmap (Numeric Features)", x = "", y = "")
    print(p_corr)
    save_plot(p_corr, "02_correlation_heatmap_numeric.png")
  }
}

# ---------- C) Data Preprocessing ----------
cat("\n========== C) PREPROCESSING ==========\n")

# Feature engineering with correct column names
df <- add_quantile_bins(df, "study_hours_per_day", "study_hours_level")
df <- add_quantile_bins(df, "ai_usage_time_minutes", "ai_usage_time_level")
df <- add_quantile_bins(df, "attendance_percentage", "attendance_level")

# Make all character predictors factors
pred_cols <- setdiff(names(df), TARGET)
for (c in pred_cols) {
  if (is.character(df[[c]])) df[[c]] <- as.factor(df[[c]])
}

# Drop high-cardinality predictors (if any)
df <- drop_high_cardinality(df, TARGET)

# Target as factor (again)
df[[TARGET]] <- as.factor(df[[TARGET]])

cat("\nFinal target classes distribution:\n")
print(table(df[[TARGET]]))

# Train/test split (stratified 80/20)
idx <- createDataPartition(df[[TARGET]], p = 0.80, list = FALSE)
train_df <- df[idx, ]
test_df  <- df[-idx, ]

cat("\nTrain size:", nrow(train_df), " (", round(nrow(train_df)/nrow(df)*100, 2), "% )\n")
cat("Test  size:", nrow(test_df),  " (", round(nrow(test_df)/nrow(df)*100, 2), "% )\n")

# Fit preprocessing on train only
pp <- fit_preprocess(train_df, target_name = TARGET)
train_pp <- apply_preprocess(train_df, pp)
test_pp  <- apply_preprocess(test_df, pp)

# Scale numeric predictors
num_cols <- pp$num_cols
if (length(num_cols) > 0) {
  preproc_scaler <- preProcess(train_pp[, num_cols, drop = FALSE], method = c("center", "scale"))
  train_pp[, num_cols] <- predict(preproc_scaler, train_pp[, num_cols, drop = FALSE])
  test_pp[, num_cols]  <- predict(preproc_scaler, test_pp[, num_cols, drop = FALSE])
}

cat("Preprocessing complete.\n")

# ---------- D) Modeling (Decision Tree) ----------
cat("\n========== D) MODELING (Decision Tree) ==========\n")

tree_model <- rpart(
  formula = as.formula(paste(TARGET, "~ .")),
  data = train_pp,
  method = "class",
  control = rpart.control(cp = 0.01)
)

cat("Decision Tree trained.\n")

# Plot tree (screen + save)
rpart.plot(tree_model, main = "Decision Tree: Performance Category (No ID/Leakage)", cex = 0.7)
save_tree_png(tree_model, "03_decision_tree_performance_category_no_leakage.png",
              main_title = "Decision Tree: Performance Category (No ID/Leakage)")

# ---------- E) Evaluation (Confusion Matrix + Accuracy/Precision/Recall/F1) ----------
cat("\n========== E) EVALUATION ==========\n")

pred_class <- predict(tree_model, newdata = test_pp, type = "class")

# Confusion Matrix (caret)
cm <- confusionMatrix(pred_class, test_pp[[TARGET]])
print(cm)

# Overall accuracy
accuracy <- as.numeric(cm$overall["Accuracy"])

cat("\n--- Overall Metric ---\n")
cat("Accuracy:", round(accuracy, 4), "\n")

# Per-class Precision / Recall / F1
# caret stores these in cm$byClass for multi-class as a matrix.
byClass <- cm$byClass

cat("\n--- Per-Class Metrics (Precision, Recall, F1) ---\n")
if (is.matrix(byClass)) {
  # For multi-class, caret provides:
  # Sensitivity = Recall
  # Pos Pred Value = Precision
  precision <- byClass[, "Pos Pred Value"]
  recall    <- byClass[, "Sensitivity"]
  f1 <- ifelse((precision + recall) == 0, NA, 2 * precision * recall / (precision + recall))
  
  metrics_df <- data.frame(
    Class = gsub("^Class: ", "", rownames(byClass)),
    Precision = round(precision, 4),
    Recall = round(recall, 4),
    F1_Score = round(f1, 4),
    row.names = NULL
  )
  
  print(metrics_df)
} else {
  # Binary case
  precision <- as.numeric(byClass["Pos Pred Value"])
  recall    <- as.numeric(byClass["Sensitivity"])
  f1 <- ifelse((precision + recall) == 0, NA, 2 * precision * recall / (precision + recall))
  
  cat("Precision:", round(precision, 4), "\n")
  cat("Recall   :", round(recall, 4), "\n")
  cat("F1-score :", round(f1, 4), "\n")
}

cat("\nTop variable importance (Decision Tree):\n")
if (!is.null(tree_model$variable.importance)) {
  print(sort(tree_model$variable.importance, decreasing = TRUE))
} else {
  cat("No variable importance available (tree may be too small).\n")
}

cat("\nDONE: Classification pipeline finished successfully.\n")
cat("Check the 'plots' folder for saved charts and the decision tree image.\n")
