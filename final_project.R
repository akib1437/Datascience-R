# ---------- Package setup ----------

packages <- c("dplyr", "ggplot2", "reshape2" )

# Install any packages that are not already installed
installed_packages <- rownames(installed.packages())
for (pkg in packages) {
  if (!(pkg %in% installed_packages)) {
    install.packages(pkg)
  }
  library(pkg, character.only = TRUE)
}

get_mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

# ---------- A. DATA UNDERSTANDING ----------


# ------- 1. Load the dataset into R -------

url <- "https://drive.google.com/uc?id=16yAZ4w_dYxirRuf2iaZNZ1HnvdRc0i_5"
study <- read.csv(url, fill = TRUE, stringsAsFactors = FALSE)

# --- 2. Display the first few rows of the dataset ----
head(study)

# --- 3. Show shape (rows × columns) -------
dim(study)      # rows, columns
nrow(study)     # just rows
ncol(study)     # just columns

# --- 4. Display data types of each column ---- 
str(study)
sapply(study, class)

# --- 5. Basic descriptive statistics (numeric columns) ---- 

numeric_cols <- sapply(study, is.numeric)
study_num <- study[, numeric_cols]

summary(study_num)

descriptive_stats <- data.frame(
  Feature = names(study_num),
  Mean   = sapply(study_num, mean),
  Median = sapply(study_num, median),
  Mode   = sapply(study_num, get_mode),
  SD     = sapply(study_num, sd),
  Min    = sapply(study_num, min),
  Max    = sapply(study_num, max),
  Count  = sapply(study_num, length)
)

descriptive_stats

#--- 6. Identify categorical and numerical features ----
cat_cols <- !numeric_cols
categorical_features <- names(study)[cat_cols]
numeric_features     <- names(study)[numeric_cols]

categorical_features
numeric_features


#--- B. DATA EXPLORATION & VISUALIZATION ----


#--- 1. Univariate Analysis ----

# Histograms and boxplots for numeric features
dir.create("plots", showWarnings = FALSE)  # create folder once

for (col in numeric_features) {
  # Histogram 
  p_hist <- ggplot(study, aes_string(x = col)) +
    geom_histogram(bins = 30, color = "black") +
    labs(title = paste("Histogram of", col)) +
    theme_minimal()
  
  # show in Plots pane (optional)
  print(p_hist)
  
  # save as PNG
  ggsave(
    filename = file.path("plots", paste0("hist_", col, ".png")),
    plot     = p_hist,
    width    = 7,
    height   = 5,
    dpi      = 300
  )
  
  # Boxplot 
  p_box <- ggplot(study, aes_string(y = col)) +
    geom_boxplot() +
    labs(title = paste("Boxplot of", col), y = col) +
    theme_minimal()
  
  print(p_box)
  
  ggsave(
    filename = file.path("plots", paste0("box_", col, ".png")),
    plot     = p_box,
    width    = 7,
    height   = 5,
    dpi      = 300
  )
}

# Bar charts / frequency of categorical variables
for (col in categorical_features) {
 
    p_bar <- ggplot(study, aes_string(x = col)) +
    geom_bar() +
    labs(title = paste("Frequency of", col),
         x = col, y = "Count") +
    theme_minimal() +
    coord_flip()
  
  print(p_bar)
  
  ggsave(
    filename = file.path("plots", paste0("bar_", col, ".png")),
    plot     = p_bar,
    width    = 7,
    height   = 5,
    dpi      = 300
  )
}


#--- 2. Bivariate Analysis ----

# Correlation matrix (numeric)
study_cor <- cor(study_num)
study_cor

cor_long <- reshape2::melt(study_cor)   # Var1, Var2, value

ggplot(cor_long, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile(color = "white") +
  geom_text(aes(label = round(value, 2)), size = 4) +
  scale_fill_gradient2(
    limits = c(-1, 1),
    breaks = seq(-1, 1, by = 0.5),
    name = "Correlation"
  ) +
  labs(
    title = "Correlation Heatmap (Numeric Features)",
    x = "",
    y = ""
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1),
    panel.grid = element_blank()
  )

ggsave("plots/correlation_heatmap.png", width = 7, height = 5, dpi = 300)


# Scatter plots for numeric pairs
if (all(c("math_score", "reading_score", "writing_score") %in% numeric_features)) {
  
  # Math vs Reading
  p1 <- ggplot(study, aes(x = math_score, y = reading_score)) +
    geom_point(alpha = 0.6) +
    labs(title = "Math vs Reading score") +
    theme_minimal()
  print(p1)
  ggsave(
    filename = file.path("plots", "scatter_math_vs_reading.png"),
    plot     = p1,
    width    = 7, height = 5, dpi = 300
  )
  
  # Math vs Writing
  p2 <- ggplot(study, aes(x = math_score, y = writing_score)) +
    geom_point(alpha = 0.6) +
    labs(title = "Math vs Writing score") +
    theme_minimal()
  print(p2)
  ggsave(
    filename = file.path("plots", "scatter_math_vs_writing.png"),
    plot     = p2,
    width    = 7, height = 5, dpi = 300
  )
  
  # Reading vs Writing
  p3 <- ggplot(study, aes(x = reading_score, y = writing_score)) +
    geom_point(alpha = 0.6) +
    labs(title = "Reading vs Writing score") +
    theme_minimal()
  print(p3)
  ggsave(
    filename = file.path("plots", "scatter_reading_vs_writing.png"),
    plot     = p3,
    width    = 7, height = 5, dpi = 300
  )
}

# Boxplots between categorical and numeric features
for (cat_col in categorical_features) {
  for (num_col in numeric_features) {
    
    p_box <- ggplot(study, aes_string(x = cat_col, y = num_col)) +
      geom_boxplot() +
      labs(title = paste(num_col, "by", cat_col),
           x = cat_col, y = num_col) +
      theme_minimal() +
      coord_flip()
    
    print(p_box)
    
    ggsave(
      filename = file.path("plots",
                           paste0("box_", num_col, "_by_", cat_col, ".png")),
      plot     = p_box,
      width    = 7,
      height   = 5,
      dpi      = 300
    )
  }
}


#--- 3. (Patterns, skewness, outliers are interpreted from the above plots) ----

#--- C. DATA PREPROCESSING ----

#--- 1. Handling Missing Values ----

# Detect existing missing values
colSums(is.na(study))

# Create a copy to work on
study_missing <- study

# Add 2 random NA values in EACH column
set.seed(123)  # for reproducibility

for (col in names(study_missing)) {
  rows_to_na <- sample(seq_len(nrow(study_missing)), size = 2, replace = FALSE)
  study_missing[rows_to_na, col] <- NA
}

# Check how many NAs per column now
colSums(is.na(study_missing))

## Handling Missing Values (Imputation)


# Helper: mode function for categorical columns
get_mode <- function(x) {
  ux <- unique(x[!is.na(x)])   # ignore NAs
  ux[which.max(tabulate(match(x, ux)))]
}

# Identify numeric and categorical columns in study_missing
numeric_cols <- sapply(study_missing, is.numeric)
numeric_features <- names(study_missing)[numeric_cols]
categorical_features <- names(study_missing)[!numeric_cols]

numeric_features
categorical_features

# Impute numeric columns (use mean OR median)

## Example: using mean
for (col in numeric_features) {
  mean_val <- mean(study_missing[[col]], na.rm = TRUE)
  study_missing[[col]][is.na(study_missing[[col]])] <- mean_val
}


# 4) Impute categorical columns with mode (most frequent category)
for (col in categorical_features) {
  mode_val <- get_mode(study_missing[[col]])
  study_missing[[col]][is.na(study_missing[[col]])] <- mode_val
}

# 5) Confirm all missing values handled
colSums(is.na(study_missing))

#--- 2. Handling Outliers for all score columns (no artificial creation) ----

# score columns
score_cols <- c("math_score", "reading_score", "writing_score")
score_cols <- intersect(score_cols, names(study))  # safety


### STEP 1: Detect outliers + compute IQR bounds

outlier_indices_list <- list()
bounds <- data.frame(
  column = score_cols,
  Q1 = NA, Q3 = NA, IQR = NA,
  lower = NA, upper = NA
)

for (i in seq_along(score_cols)) {
  col <- score_cols[i]
  x <- study[[col]]
  
  Q1  <- quantile(x, 0.25, na.rm = TRUE)
  Q3  <- quantile(x, 0.75, na.rm = TRUE)
  IQR <- Q3 - Q1
  lower <- Q1 - 1.5 * IQR
  upper <- Q3 + 1.5 * IQR
  
  bounds$Q1[i]    <- Q1
  bounds$Q3[i]    <- Q3
  bounds$IQR[i]   <- IQR
  bounds$lower[i] <- lower
  bounds$upper[i] <- upper
  
  idx <- which(x < lower | x > upper)
  outlier_indices_list[[col]] <- idx
  
  cat("\nColumn:", col,
      "\nQ1:", round(Q1, 2), " Q3:", round(Q3, 2),
      "\nLower bound:", round(lower, 2),
      "\nUpper bound:", round(upper, 2),
      "\nNumber of outliers:", length(idx),
      "\nIndices:", if (length(idx) == 0) "none" else idx, "\n")
}

bounds  # shows Q1, Q3, IQR, lower, upper for each score column

# rows that have an outlier in ANY score column
rows_with_any_outlier <- unique(unlist(outlier_indices_list))
cat("\nTotal rows with at least one outlier:", length(rows_with_any_outlier), "\n")


#handle missing values (remove because these outliers are errors or not realistic.)
study_remove <- study
if (length(rows_with_any_outlier) > 0) {
  study_remove <- study[-rows_with_any_outlier, ]
}

dim(study)         # original
dim(study_remove)  # after removing outlier rows

study_processed <- study_remove


#--- 3. Data Conversion (Encoding) ----

study_enc <- study_processed

# Label encoding
for (col in categorical_features) {
  study_enc[[col]] <- as.numeric(as.factor(study_enc[[col]]))
}

str(study_enc)
head(study_enc)

# From the encoded, cleaned dataset:
numeric_cols_final <- sapply(study_enc, is.numeric)
study_num_final <- study_enc[, numeric_cols_final]

# Just to check:
str(study_num_final)


#--- 4. Data Transformation (Scaling / Normalization) ----

# Z-score standardization
scaled_data <- as.data.frame(scale(study_num_final))
summary(scaled_data)

# Min–Max normalization
min_max_norm <- function(x) {
  (x - min(x)) / (max(x) - min(x))
}
norm_data <- as.data.frame(sapply(study_num, min_max_norm))
summary(norm_data)

#--- 5. Feature Selection (basic) ----

cor_matrix <- cor(study_num_final)
cor_matrix

# Find high-correlation pairs (|cor| > 0.9)
high_corr_pairs <- which(abs(cor_matrix) > 0.9 & abs(cor_matrix) < 1, arr.ind = TRUE)
high_corr_pairs

# Variance-based selection
variances <- sapply(study_num_final, var)
variances

threshold <- 10
selected_features <- names(variances[variances > threshold])
selected_features


# Final dataset after preprocessing + encoding + feature selection
study_final <- study_enc[, selected_features, drop = FALSE]

str(study_final)



#--- Save the new Dataset: after preprocessing + encoding + feature selection ----
write.csv(study_final, "study_performance_final.csv", row.names = FALSE)



