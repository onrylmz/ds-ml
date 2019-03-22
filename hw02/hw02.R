# define the sigmoid function
sigmoid <- function(X, w, w0) {
  scores <- cbind(X, 1) %*% rbind(W, w0)
  scores <- exp(-scores)
  
  scores <- (1/(1 + scores))
  
  return (scores)
}

# read data into memory
data("iris")

x <- as.matrix(iris[,1:4])
y <- as.numeric(iris[, 5])

# get training data set
train.x <- rbind(rbind(x[1:25,], x[51:75,]), x[101:125,])
train.y <- c(c(y[1:25], y[51:75]), y[101:125])

# get validation data set
test.x <- rbind(rbind(x[26:50,], x[76:100,]), x[126:150,])
test.y <- c(c(y[26:50], y[76:100]), y[126:150])

# get number of classes and number of samples
K <- max(train.y)
N <- length(train.y)

# one-of-K-encoding
Y_truth <- matrix(0, N, K)
Y_truth[cbind(1:N, train.y)] <- 1

# set learning parameters
eta <- 0.01
epsilon <- 1e-3

# randomly initalize w and w0
set.seed(521)

# define the gradient functions
gradient_W <- function(X, Y_truth, Y_predicted) {
  return (-sapply(X = 1:ncol(Y_truth), function(c) colSums(matrix(((Y_truth[,c] - Y_predicted[,c]) * Y_predicted[,c] * (1 - Y_predicted[,c])), nrow = nrow(X), ncol = ncol(X), byrow = FALSE) * X)))
}

gradient_w0 <- function(Y_truth, Y_predicted) {
  return (-colSums(((Y_truth - Y_predicted) * Y_predicted * (1 - Y_predicted))))
}

W <- matrix(runif(ncol(train.x) * K, min = -0.01, max = 0.01), ncol(train.x), K)
w0 <- runif(K, min = -0.01, max = 0.01)

# learn W and w0 using gradient descent
iteration <- 1
objective_values <- c()
while (1) {
  Y_predicted <- sigmoid(train.x, W, w0)
  
  objective_values <- c(objective_values, sum((Y_truth - Y_predicted)^2)/2)
  
  W_old <- W
  w0_old <- w0
  
  W <- W - eta * gradient_W(train.x, Y_truth, Y_predicted)
  w0 <- w0 - eta * gradient_w0(Y_truth, Y_predicted)
  
  if (sqrt(sum((w0 - w0_old)^2) + sum((W - W_old)^2)) < epsilon) {
    break
  }
  
  iteration <- iteration + 1
}

# plot objective function during iterations
plot(1:iteration, objective_values,
     type = "l", lwd = 2, las = 1,
     xlab = "Iteration", ylab = "Error")

y_predicted <- apply(Y_predicted, 1, which.max)
confusion_matrix <- table(y_predicted, train.y)
print(confusion_matrix)

Y_test <- sigmoid(test.x, W, w0)
y_test <- apply(Y_test, 1, which.max)
confusion_matrix_test <- table(y_test, test.y)
print(confusion_matrix_test)
