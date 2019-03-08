# read data into memory
data_set <- read.csv("hw01_data_set.csv")

# get training data set
train.x <- data_set$X[1:160]
train.y <- data_set$y[1:160]

# get validation data set
test.x <- data_set$X[161:200]
test.y <- data_set$y[161:200]

# plot the training data to see the trend
plot(test.x, test.y, type="p", col = "red", xlab = "x", ylab = "y")
points(train.x, train.y, type="p", col = "blue", xlab = "x", ylab = "y")

K <- 5
N <- length(train.x)
N_TEST <- length(test.x)

linear_regression <- function(x, y) {
  X <- matrix(0, N, K)

  X[, 1] <- 1
  X[, 2] <- x
  X[, 3] <- x^3
  X[, 4] <- exp(x)
  X[, 5] <- sin(x)
  
  # Transpose of X
  t_X <- t(X)
  
  R <- matrix(0, N, 1)
  R[, 1] <- y
  
  w <- chol2inv(chol(t_X %*% X)) %*% t_X %*% R
  
  return(w)
}

# Learnt parameters
w <- linear_regression(train.x, train.y)

# Model function
y_i <- function(x) {
  return(w[1] + w[2]*x + w[3]*(x^3) + w[4]*exp(x) + w[5]*sin(x))
}

# Find rmse
rmse <- sqrt(sum((test.y - y_i(test.x))^2) / N_TEST)

# Data interval
data_interval <- seq(from = -1, to = +6, by = 0.01)

# plot the training and test data points and fitted curve for the data interval
plot(test.x, test.y, type="p", col = "red", xlab = "x", ylab = "y")
points(train.x, train.y, type="p", col = "blue", xlab = "x", ylab = "y")
points(data_interval, y_i(data_interval), type="l", col="black", lwd=2)

