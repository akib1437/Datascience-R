
# install.packages("tidyverse")
# install.packages("corrplot")
# install.packages("caret")

library(tidyverse) # data manipulation (dplyr) and plotting (ggplot2)
library(corrplot)  # correlation heatmap
library(caret)     # preprocessing (encoding, scaling)


# --- TASK A: Data Understanding ---------------------------------------------------


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


# --- TASK B: Data Exploration & Visualization (EDA)---------------------------------------------------

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


# --- TASK C: Data Preprocessing ---------------------------------------------------

# 1.  Missing values

colSums(is.na(df))

# 2. Create a few missing values

set.seed(123)
idx <- sample(1:nrow(df), 10)
df$math_score[idx] <- NA

colSums(is.na(df))



#-- Encoding categorical variables

df$gender <- as.factor(df$gender)
df$race_ethnicity <- as.factor(df$race_ethnicity)
df$parental_level_of_education <- as.factor(df$parental_level_of_education)
df$lunch <- as.factor(df$lunch)
df$test_preparation_course <- as.factor(df$test_preparation_course)

df_dummy <- model.matrix(~ . - 1, data = df)  # one-hot encode
df_dummy <- as.data.frame(df_dummy)


head(df_dummy)


#-- Save cleaned data ---------------------------------------------------
write.csv(df_dummy, "study_performance_cleaned.csv", row.names = FALSE)

getwd()