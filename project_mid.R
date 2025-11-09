
# install.packages("tidyverse")
# install.packages("corrplot")
# install.packages("caret")

library(tidyverse) # data manipulation (dplyr) and plotting (ggplot2)
library(corrplot)  # correlation heatmap
library(caret)     # preprocessing (encoding, scaling)


# --- TASK A: Data Understanding


# 1. Load the dataset

url <- "https://drive.google.com/uc?id=16yAZ4w_dYxirRuf2iaZNZ1HnvdRc0i_5"
df <- read.csv(url, fill = TRUE)

# 2. Display the first few rows
head(df)
tail(df)

# 3. Show shape (rows & columns)
dim(df)         # rows, columns
str(df)         # data types
sapply(df, class)

# 5. Generate basic descriptive statistics

summary(df)     # for both numeric & categorical

num_cols <- c("math_score", "reading_score", "writing_score")
sapply(df[, num_cols], mean)
sapply(df[, num_cols], sd)


# --- TASK B: Data Exploration & Visualization (EDA)

# 1. Univariate Analysis

ggplot(df, aes(x = math_score)) + geom_histogram(binwidth = 5)
ggplot(df, aes(y = math_score)) + geom_boxplot()

ggplot(df, aes(x = reading_score)) + geom_histogram(binwidth = 5)
ggplot(df, aes(y = reading_score)) + geom_boxplot()

ggplot(df, aes(x = writing_score)) + geom_histogram(binwidth = 5)
ggplot(df, aes(y = writing_score)) + geom_boxplot()


# 2. Categorical frequencies


  table(df$gender)
ggplot(df, aes(x = gender)) + geom_bar()

ggplot(df, aes(x = race_ethnicity)) + geom_bar()

ggplot(df, aes(x = parental_level_of_education)) + geom_bar() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggplot(df, aes(x = lunch)) + geom_bar()

ggplot(df, aes(x = test_preparation_course)) + geom_bar()


# 3. Bivariate EDA (Task B – Bivariate)


# 1. Correlation between scores (numeric–numeric)

num_cols <- c("math_score", "reading_score", "writing_score")
cor_matrix <- cor(df[, num_cols])
cor_matrix
#######  heatmap

# 2. Scatter plots between numeric pairs

ggplot(df, aes(x = math_score, y = reading_score)) + geom_point(alpha = 0.5)
ggplot(df, aes(x = reading_score, y = writing_score)) + geom_point(alpha = 0.5)
ggplot(df, aes(x = math_score, y = writing_score)) + geom_point(alpha = 0.5)

# 3. Boxplots: categorical vs numeric

ggplot(df, aes(x = gender, y = math_score)) + geom_boxplot()
ggplot(df, aes(x = lunch, y = math_score)) + geom_boxplot()
ggplot(df, aes(x = test_preparation_course, y = math_score)) + geom_boxplot()

ggplot(df, aes(x = gender, y = reading_score)) + geom_boxplot()
ggplot(df, aes(x = lunch, y = reading_score)) + geom_boxplot()
ggplot(df, aes(x = test_preparation_course, y = reading_score)) + geom_boxplot()

ggplot(df, aes(x = gender, y = writing_score)) + geom_boxplot()
ggplot(df, aes(x = lunch, y = writing_score)) + geom_boxplot()
ggplot(df, aes(x = test_preparation_course, y = writing_score)) + geom_boxplot()


# --- TASK C: Data Preprocessing 

# 1.  Missing values

colSums(is.na(df))

# 2. Create a few missing values

set.seed(123)
idx <- sample(1:nrow(df), 10)
df$math_score[idx] <- NA

colSums(is.na(df))

############################  Have to check
# 3. Impute them

#median_math <- median(df$math_score, na.rm = TRUE)
#df$math_score[is.na(df$math_score)] <- median_math

###########################  Have to check
# 4. Outliers

Q1 <- quantile(df$math_score, 0.25)
Q3 <- quantile(df$math_score, 0.75)
IQR_val <- Q3 - Q1
lower_bound <- Q1 - 1.5 * IQR_val
upper_bound <- Q3 + 1.5 * IQR_val
which(df$math_score < lower_bound | df$math_score > upper_bound)

# 5. Optionally create artificial outliers

df$math_score[1] <- 0    # extremely low
df$math_score[2] <- 100  # extremely high

# 6. Handle outliers
    # Cap them at lower/upper bound
df$math_score[df$math_score < lower_bound] <- lower_bound
df$math_score[df$math_score > upper_bound] <- upper_bound

#-- Encoding categorical variables

df$gender <- as.factor(df$gender)
df$race_ethnicity <- as.factor(df$race_ethnicity)
df$parental_level_of_education <- as.factor(df$parental_level_of_education)
df$lunch <- as.factor(df$lunch)
df$test_preparation_course <- as.factor(df$test_preparation_course)

df_dummy <- model.matrix(~ . - 1, data = df)  # one-hot encode
df_dummy <- as.data.frame(df_dummy)


head(df_dummy)


#-- Scaling / transformation

num_cols <- c("math_score", "reading_score", "writing_score")
df_dummy[num_cols] <- scale(df_dummy[num_cols])

#####If any numeric variable is clearly skewed, try log/sqrt (you’ll see from histograms).
#####Explain why scaling helps for models that are distance-based or sensitive to scale.


#-- Feature selection (for “potential ML”)

#  1. Use correlation among numeric features

# 2. Use simple methods
nzv <- nearZeroVar(df_dummy)
df_reduced <- df_dummy[, -nzv]

df_reduced
################ explain that low-variance features contribute little information.


# 7. Save cleaned data ---------------------------------------------------
write.csv(df_reduced, "study_performance_cleaned.csv", row.names = FALSE)

getwd()