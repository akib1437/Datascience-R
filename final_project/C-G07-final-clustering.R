############################################################
# C-G07-final-clustering-insurance.R
# Final Project - Introduction to Data Science (Option 1)
# Task: Clustering (K-Means)
#
# Dataset Source (Kaggle):
# https://www.kaggle.com/datasets/vivekbaaganps/insurence
# Dataset Access (Google Drive file used in code):
# https://drive.google.com/file/d/1drUbxfCrbw6IDPbeKqyoWM5JSlnjJz3l/view?usp=sharing
#
# Goal:
# Cluster insurance customers using K-Means,
# select k via Elbow + Silhouette, and visualize clusters using PCA.

############################################################

# ---------- Package setup (auto-install if missing) ----------
install_if_missing <- function(pkgs) {
  missing <- pkgs[!pkgs %in% installed.packages()[, "Package"]]
  if (length(missing) > 0) install.packages(missing, dependencies = TRUE)
}

install_if_missing(c("readr", "dplyr", "ggplot2", "caret", "cluster"))

library(readr)
library(dplyr)
library(ggplot2)
library(caret)
library(cluster)

set.seed(123)

# ---------- Plot saving setup ----------
# All figures will be saved to: ./plots/
PLOT_DIR <- "plots"
if (!dir.exists(PLOT_DIR)) dir.create(PLOT_DIR)

save_plot <- function(plot_obj, filename, width = 8, height = 5, dpi = 300) {
  ggplot2::ggsave(
    filename = file.path(PLOT_DIR, filename),
    plot = plot_obj,
    width = width,
    height = height,
    dpi = dpi
  )
}

# ---------- A) Data Collection (load from Google Drive) ----------
FILE_ID <- "1drUbxfCrbw6IDPbeKqyoWM5JSlnjJz3l"
DATA_URL <- paste0("https://drive.google.com/uc?export=download&id=", FILE_ID)

read_drive_csv_safe <- function(url) {
  # 1) Try direct read (works for most small/medium Drive CSVs)
  out <- tryCatch(
    readr::read_csv(url, show_col_types = FALSE),
    error = function(e) NULL
  )
  if (!is.null(out)) return(out)
}

cat("Loading dataset from Google Drive...\n")
df_raw <- read_drive_csv_safe(DATA_URL)
cat("Data loaded successfully.\n\n")

# ---------- Helper functions ----------
mode_value <- function(x) {
  x <- x[!is.na(x)]
  if (length(x) == 0) return(NA)
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

fit_preprocess_full <- function(df) {
  num_cols <- names(df)[sapply(df, is.numeric)]
  cat_cols <- names(df)[sapply(df, function(x) is.character(x) || is.factor(x) || is.logical(x))]
  
  # Imputation targets
  medians <- sapply(df[num_cols], function(x) median(x, na.rm = TRUE))
  modes   <- sapply(df[cat_cols], mode_value)
  
  # Outlier caps (IQR rule)
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
    factor_levels = factor_levels
  )
}

apply_preprocess_full <- function(df, pp) {
  out <- df
  
  # Ensure logical -> character before factor to avoid odd level behavior
  for (c in pp$cat_cols) {
    if (is.logical(out[[c]])) out[[c]] <- as.character(out[[c]])
  }
  
  # Impute numeric
  for (c in pp$num_cols) {
    out[[c]][is.na(out[[c]])] <- pp$medians[[c]]
  }
  
  # Impute categorical and enforce factor levels
  for (c in pp$cat_cols) {
    out[[c]][is.na(out[[c]])] <- pp$modes[[c]]
    out[[c]] <- as.factor(out[[c]])
    out[[c]] <- factor(out[[c]], levels = pp$factor_levels[[c]])
  }
  
  # Outlier caps for numeric
  for (c in pp$num_cols) {
    low <- pp$caps[[c]]$low
    high <- pp$caps[[c]]$high
    out[[c]] <- pmin(pmax(out[[c]], low), high)
  }
  
  out
}

avg_silhouette <- function(x_mat, cluster_vec) {
  sil <- silhouette(cluster_vec, dist(x_mat))
  mean(sil[, "sil_width"])
}

# ---------- B) Data Understanding & Exploration ----------
cat("========== B) DATA UNDERSTANDING & EDA ==========\n")
cat("Shape (rows, cols): ", nrow(df_raw), ", ", ncol(df_raw), "\n\n")

cat("Data types:\n")
print(str(df_raw))

cat("\nSummary statistics:\n")
print(summary(df_raw))

df <- df_raw

# Standardize common column names (in case Kaggle file differs slightly)
# Expecting: age, sex, bmi, children, smoker, region, charges
names(df) <- tolower(names(df))

cat("\nMissing values per column:\n")
print(sapply(df, function(x) sum(is.na(x))))

# Quick sanity check for expected columns (won't stop if extra columns exist)
expected_cols <- c("age", "sex", "bmi", "children", "smoker", "region", "charges")
missing_expected <- setdiff(expected_cols, names(df))
if (length(missing_expected) > 0) {
  cat("\nWARNING: These expected columns were not found:\n")
  print(missing_expected)
  cat("Continuing with available columns...\n\n")
}

# ---------- Basic EDA plots + SAVE ----------
# 1) Distribution of charges (Histogram)
if ("charges" %in% names(df) && is.numeric(df$charges)) {
  p_hist_charges <- ggplot(df, aes(x = charges)) +
    geom_histogram(bins = 30) +
    labs(title = "Distribution of Insurance Charges", x = "Charges", y = "Count")
  print(p_hist_charges)
  save_plot(p_hist_charges, "01_distribution_charges.png", width = 8, height = 5)
}

# 2) Charges by smoker status (Boxplot)
if (all(c("smoker", "charges") %in% names(df))) {
  p_box_smoker <- ggplot(df, aes(x = as.factor(smoker), y = charges)) +
    geom_boxplot() +
    labs(title = "Charges by Smoker Status", x = "Smoker", y = "Charges")
  print(p_box_smoker)
  save_plot(p_box_smoker, "02_charges_by_smoker.png", width = 7, height = 5)
}

# 3) Charges by region (Boxplot)
if (all(c("region", "charges") %in% names(df))) {
  p_box_region <- ggplot(df, aes(x = as.factor(region), y = charges)) +
    geom_boxplot() +
    labs(title = "Charges by Region", x = "Region", y = "Charges") +
    theme(axis.text.x = element_text(angle = 20, hjust = 1))
  print(p_box_region)
  save_plot(p_box_region, "03_charges_by_region.png", width = 8, height = 5)
}

# ---------- C) Data Preprocessing ----------
cat("\n========== C) PREPROCESSING ==========\n")

# Choose whether to include charges in clustering features:
# - TRUE: clusters influenced by cost (charges) heavily
# - FALSE: clusters based on demographics/health/region; charges used for interpretation
INCLUDE_CHARGES_IN_CLUSTERING <- FALSE

features_df <- df

# Remove non-informative ID-like columns if present (common in some CSVs)
id_like <- c("id", "index", "customer_id")
id_like <- id_like[id_like %in% names(features_df)]
if (length(id_like) > 0) features_df <- features_df %>% select(-all_of(id_like))

if (!INCLUDE_CHARGES_IN_CLUSTERING && "charges" %in% names(features_df)) {
  features_df <- features_df %>% select(-charges)
}

# Fit + apply preprocessing (impute + cap outliers)
pp <- fit_preprocess_full(features_df)
features_pp <- apply_preprocess_full(features_df, pp)

# One-hot encode categoricals
X <- model.matrix(~ . , data = features_pp)[, -1, drop = FALSE]

# Standardize (important for K-means)
scaler <- preProcess(X, method = c("center", "scale"))
X_scaled <- predict(scaler, X)

cat("Preprocessing complete. Feature matrix size: ", nrow(X_scaled), "x", ncol(X_scaled), "\n")

# ---------- D) Modeling (K-Means) ----------
cat("\n========== D) MODELING (K-Means) ==========\n")

k_values <- 2:10
wss <- numeric(length(k_values))
sil <- numeric(length(k_values))

for (i in seq_along(k_values)) {
  k <- k_values[i]
  km <- kmeans(X_scaled, centers = k, nstart = 25)
  wss[i] <- km$tot.withinss
  sil[i] <- avg_silhouette(X_scaled, km$cluster)
}

elbow_df <- data.frame(k = k_values, wss = wss)
sil_df   <- data.frame(k = k_values, avg_silhouette = sil)

# Plot elbow + SAVE
p_elbow <- ggplot(elbow_df, aes(x = k, y = wss)) +
  geom_line() + geom_point() +
  labs(title = "Elbow Method (Within-Cluster Sum of Squares)", x = "k", y = "WSS")
print(p_elbow)
save_plot(p_elbow, "04_elbow_wss.png", width = 7, height = 5)

# Plot silhouette + SAVE
p_sil <- ggplot(sil_df, aes(x = k, y = avg_silhouette)) +
  geom_line() + geom_point() +
  labs(title = "Average Silhouette Score vs k", x = "k", y = "Avg Silhouette")
print(p_sil)
save_plot(p_sil, "05_silhouette_vs_k.png", width = 7, height = 5)

best_k <- sil_df$k[which.max(sil_df$avg_silhouette)]
cat("\nSelected k (max average silhouette): ", best_k, "\n")

final_km <- kmeans(X_scaled, centers = best_k, nstart = 50)
clusters <- factor(final_km$cluster)

cat("K-means clustering complete.\n")

# ---------- E) Evaluation, Visualization & Interpretation ----------
cat("\n========== E) EVALUATION & INTERPRETATION ==========\n")

final_sil <- avg_silhouette(X_scaled, final_km$cluster)
cat("Average Silhouette Score (final): ", round(final_sil, 4), "\n")

# PCA for 2D visualization
pca <- prcomp(X_scaled)
pca_df <- data.frame(PC1 = pca$x[, 1], PC2 = pca$x[, 2], cluster = clusters)

p_pca <- ggplot(pca_df, aes(x = PC1, y = PC2, color = cluster)) +
  geom_point(alpha = 0.6) +
  labs(title = "Cluster Visualization (PCA Projection)", x = "PC1", y = "PC2")
print(p_pca)
save_plot(p_pca, "06_pca_clusters.png", width = 8, height = 5)

# Attach clusters back to original dataframe for profiling
df_with_clusters <- df %>% mutate(cluster = clusters)

cat("\nCluster sizes:\n")
print(table(df_with_clusters$cluster))

# Numeric summary (only for numeric columns that exist)
num_cols_all <- names(df_with_clusters)[sapply(df_with_clusters, is.numeric)]
num_cols_all <- setdiff(num_cols_all, character(0))  # no-op; kept for clarity

cat("\nCluster-wise numeric means:\n")
print(
  df_with_clusters %>%
    group_by(cluster) %>%
    summarise(across(all_of(num_cols_all), ~ mean(.x, na.rm = TRUE)), .groups = "drop")
)

# Categorical distributions (if present)
cat_cols_all <- names(df_with_clusters)[sapply(df_with_clusters, function(x) is.character(x) || is.factor(x) || is.logical(x))]
cat_cols_all <- setdiff(cat_cols_all, "cluster")

if (length(cat_cols_all) > 0) {
  cat("\nCluster-wise categorical distributions (top levels shown):\n")
  for (cc in cat_cols_all) {
    cat("\n---", cc, "---\n")
    print(
      df_with_clusters %>%
        mutate(tmp = as.factor(.data[[cc]])) %>%
        count(cluster, tmp) %>%
        group_by(cluster) %>%
        mutate(pct = round(100 * n / sum(n), 2)) %>%
        arrange(cluster, desc(n)) %>%
        slice_head(n = 5) %>%
        ungroup()
    )
  }
}

cat("\nInterpretation (brief):\n")
cat("- Higher silhouette indicates better-separated clusters.\n")
cat("- Use the cluster profiles (means + categorical distributions) to describe each segment.\n")
cat("- If charges were excluded from clustering, treat charges as an outcome variable to compare across segments.\n")

cat("\nAll plots saved to folder: ", normalizePath(PLOT_DIR), "\n")
