##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

##########################################################
# Splitting edx data set in train set and test set
##########################################################
edx_test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.01, list = FALSE)

train_set <- edx[-edx_test_index, ]
test_set <- edx[edx_test_index, ]

##########################################################
# Creating base model which predicts average of all ratings for each movie
##########################################################
avg_rating <- mean(train_set$rating)

base_model_predictions <- rep(avg_rating, nrow(test_set))

base_model_rmse <- RMSE(test_set$rating, base_model_predictions, na.rm=TRUE)

rmse_results <- data_frame(Method = "Base Model", RMSE = base_model_rmse)

##########################################################
# Creating model which predicts ratings based on average rating of each movie
##########################################################
movie_avgs <- train_set %>% 
  group_by(movieId) %>% 
  summarize(movie_avg = mean(rating - avg_rating))

movie_avgs %>% qplot(movie_avg, geom ="histogram", bins = 10, data = ., color = I("black"), xlab = "Movie Rating", ylab = "Count")

model_1_predictions <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  mutate(prediction = avg_rating + movie_avg) %>% .$prediction

model_1_rmse <- RMSE(test_set$rating, model_1_predictions, na.rm=TRUE)

rmse_results <- bind_rows(rmse_results,
                          data_frame(Method="Movie Effect Model",
                                     RMSE = model_1_rmse ))

rmse_results %>% knitr::kable()

##########################################################
# Creating model which predicts ratings based on average rating of each movie AND the average rating a user provides
##########################################################
user_avgs <- train_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(user_avg = mean(rating - avg_rating - movie_avg))

user_avgs %>% qplot(user_avg, geom ="histogram", bins = 10, data = ., color = I("black"), xlab = "User Rating", ylab = "Count")

model_2_predictions <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(prediction = avg_rating + movie_avg + user_avg) %>% .$prediction

model_2_rmse <- RMSE(test_set$rating, model_2_predictions, na.rm=TRUE)

rmse_results <- bind_rows(rmse_results,
                          data_frame(Method="Movie + User Effects Model",  
                                     RMSE = model_2_rmse ))

rmse_results %>% knitr::kable()

##########################################################
# Introducing regularization to above models to constrain large estimates of averages
##########################################################
lambda <- 3 # trying tuning parameter value = 3

regularized_movie_avgs <- train_set %>% 
  group_by(movieId) %>% 
  summarize(regularized_movie_avg = sum(rating - avg_rating)/(n() + lambda)) # Regularization on movie specific effects

regularized_user_avgs <- train_set %>% 
  left_join(regularized_movie_avgs, by="movieId") %>%
  group_by(userId) %>%
  summarize(regularized_user_avg = sum(rating - regularized_movie_avg - avg_rating)/(n() + lambda)) # Regularization on user specific effects

model_3_predictions <- test_set %>% 
  left_join(regularized_movie_avgs, by = "movieId") %>%
  left_join(regularized_user_avgs, by = "userId") %>%
  mutate(prediction = avg_rating + regularized_movie_avg + regularized_user_avg) %>% .$prediction

model_3_rmse <- RMSE(model_3_predictions, test_set$rating, na.rm=TRUE)

rmse_results <- bind_rows(rmse_results,
                          data_frame(Method="Regularized Model (lambda = 3)",  
                                     RMSE = model_3_rmse ))

##########################################################
# Testing out range of values for lambda
##########################################################
lambdas <- seq(0, 10, 0.2)

RMSEs <- sapply(lambdas, function(l){
  
  regularized_movie_avgs <- train_set %>% 
    group_by(movieId) %>% 
    summarize(regularized_movie_avg = sum(rating - avg_rating)/(n() + l))
  
  regularized_user_avgs <- train_set %>% 
    left_join(regularized_movie_avgs, by="movieId") %>%
    group_by(userId) %>%
    summarize(regularized_user_avg = sum(rating - regularized_movie_avg - avg_rating)/(n() + l))
  
  model_4_predictions <- test_set %>% 
    left_join(regularized_movie_avgs, by = "movieId") %>%
    left_join(regularized_user_avgs, by = "userId") %>%
    mutate(prediction = avg_rating + regularized_movie_avg + regularized_user_avg) %>% .$prediction
  
  return(RMSE(model_4_predictions, test_set$rating, na.rm = TRUE))
})

qplot(lambdas, RMSEs, xlab = "Lambda", ylab = "RMSE")

min_lambda <- lambdas[which.min(RMSEs)]

rmse_results <- bind_rows(rmse_results,
                          data_frame(Method="Regularized Model (using optimum lambda)",  
                                     RMSE = min(RMSEs)))
rmse_results %>% knitr::kable()

##########################################################
# Making final prediction using best lambda
##########################################################
final_movie_avgs <- train_set %>% 
  group_by(movieId) %>% 
  summarize(final_movie_avg = sum(rating - avg_rating)/(n() + min_lambda))

final_user_avgs <- train_set %>% 
  left_join(final_movie_avgs, by="movieId") %>%
  group_by(userId) %>%
  summarize(final_user_avg = sum(rating - final_movie_avg - avg_rating)/(n() + min_lambda))

final_predictions <- validation %>% 
  left_join(final_movie_avgs, by = "movieId") %>%
  left_join(final_user_avgs, by = "userId") %>%
  mutate(prediction = avg_rating + final_movie_avg + final_user_avg) %>% .$prediction

final_rmse <- RMSE(validation$rating, final_predictions, na.rm = TRUE)
final_rmse