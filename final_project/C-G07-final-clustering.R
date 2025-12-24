############################################################
# C-G01-final-clustering.R
# Final Project - Introduction to Data Science (Option 1)
# Task: Clustering (K-Means)
#
# Dataset Source (Kaggle):
# https://www.kaggle.com/datasets/yasserh/wine-quality-dataset
# Dataset Access (Google Drive file used in code):
# https://drive.google.com/file/d/1pEa9dRR7dAKu5lMiF_XfdyTIEb5ZXz0i/view
#
# Goal:
# Cluster wines by chemical properties using K-Means,
# evaluate using Silhouette Score, and visualize clusters.
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

fit_preprocess_full <- function(df) {
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
  
  list(num_cols = num_cols, cat_cols = cat_cols, medians = medians, modes = modes,
       caps = caps, factor_levels = factor_levels)
}

apply_preprocess_full <- function(df, pp) {
  out <- df
  
  # Impute
  for (c in pp$num_cols) out[[c]][is.na(out[[c]])] <- pp$medians[[c]]
  for (c in pp$cat_cols) {
    out[[c]][is.na(out[[c]])] <- pp$modes[[c]]
    out[[c]] <- as.factor(out[[c]])
    out[[c]] <- factor(out[[c]], levels = pp$factor_levels[[c]])
  }
  
  # Outlier caps
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

# Feature engineering to ensure categorical + numeric exist (guideline requirement)
if ("alcohol" %in% names(df)) {
  df <- df %>% mutate(alcohol_level = cut(alcohol,
                                          breaks = quantile(alcohol, probs = c(0, .33, .66, 1), na.rm = TRUE),
                                          include.lowest = TRUE, labels = c("low", "mid", "high")))
}
if ("pH" %in% names(df)) {
  df <- df %>% mutate(ph_level = cut(pH,
                                     breaks = quantile(pH, probs = c(0, .33, .66, 1), na.rm = TRUE),
                                     include.lowest = TRUE, labels = c("low", "mid", "high")))
}

# ---------- C) Data Preprocessing ----------
cat("\n========== C) PREPROCESSING ==========\n")

# For clustering, we DO NOT use the label for training clusters.
# We'll cluster on features; then we can compare clusters with quality afterwards.
features_df <- df %>% select(-quality)

pp <- fit_preprocess_full(features_df)
features_pp <- apply_preprocess_full(features_df, pp)

# One-hot encode categoricals
X <- model.matrix(~ . , data = features_pp)[, -1, drop = FALSE]

# Standardize numeric scale (important for K-means)
scaler <- preProcess(X, method = c("center", "scale"))
X_scaled <- predict(scaler, X)

cat("Preprocessing complete. Feature matrix size: ", nrow(X_scaled), "x", ncol(X_scaled), "\n")

# ---------- D) Modeling (K-Means) ----------
cat("\n========== D) MODELING (K-Means) ==========\n")

# Choose k using Elbow + Silhouette
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

# Plot elbow
ggplot(elbow_df, aes(x = k, y = wss)) +
  geom_line() + geom_point() +
  labs(title = "Elbow Method (Within-Cluster Sum of Squares)", x = "k", y = "WSS")

# Plot silhouette
ggplot(sil_df, aes(x = k, y = avg_silhouette)) +
  geom_line() + geom_point() +
  labs(title = "Average Silhouette Score vs k", x = "k", y = "Avg Silhouette")

best_k <- sil_df$k[which.max(sil_df$avg_silhouette)]
cat("\nSelected k (max average silhouette): ", best_k, "\n")

final_km <- kmeans(X_scaled, centers = best_k, nstart = 50)
clusters <- factor(final_km$cluster)

cat("K-means clustering complete.\n")

# ---------- E) Evaluation & Interpretation ----------
cat("\n========== E) EVALUATION & INTERPRETATION ==========\n")

final_sil <- avg_silhouette(X_scaled, final_km$cluster)
cat("Average Silhouette Score (final): ", round(final_sil, 4), "\n")

# PCA for 2D visualization
pca <- prcomp(X_scaled)
pca_df <- data.frame(PC1 = pca$x[,1], PC2 = pca$x[,2], cluster = clusters)

ggplot(pca_df, aes(x = PC1, y = PC2, color = cluster)) +
  geom_point(alpha = 0.6) +
  labs(title = "Cluster Visualization (PCA Projection)", x = "PC1", y = "PC2")

# Post-hoc interpretation using original quality
df_with_clusters <- df %>% mutate(cluster = clusters)

cat("\nCluster summary vs quality (post-hoc):\n")
print(
  df_with_clusters %>%
    group_by(cluster) %>%
    summarise(
      n = n(),
      avg_quality = mean(quality, na.rm = TRUE),
      sd_quality = sd(quality, na.rm = TRUE),
      avg_alcohol = if ("alcohol" %in% names(df_with_clusters)) mean(alcohol, na.rm = TRUE) else NA_real_
    )
)

cat("\nInterpretation (brief):\n")
cat("- Higher silhouette indicates better-separated clusters.\n")
cat("- The cluster summary shows whether certain clusters tend to have higher average quality.\n")
