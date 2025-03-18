# regression
library(MASS)
library(mgcv)
library(splines)
library(glmnet)
library(dplyr)
library(Matrix)
library(caret)


X.reg = read.table("a24_reg_app.txt")
X<-subset(X.reg,select = -c(y))
preproc <- preProcess(X, method = c("center", "scale"))
X.scaled <- predict(preproc, X)
data <- data.frame(y=X.reg$y,X.scaled)
set.seed(123)  

outer_folds <- createFolds(data$y, k = 10, list = TRUE)  
results <- list()

for (i in seq_along(outer_folds)) {
  cat(sprintf("Processing outer fold %d\n", i))
  test_indices <- outer_folds[[i]]
  train_data <- data[-test_indices, ]
  test_data <- data[test_indices, ]
  train_control <- trainControl(
    method = "cv",                  
    number = 3,                        
    search = "grid"                 
  )
  tune_grid <- expand.grid(
    alpha = seq(0, 1, by=0.005),    
    lambda = 10^seq(-4, 0, length.out = 20)
  )
  model <- train(
    y ~ .,                            
    data = train_data,
    method = "glmnet",
    tuneGrid = tune_grid,
    trControl = train_control,
    metric= "RMSE"
  )
  
  best_alpha <- model$bestTune$alpha
  best_lambda <- model$bestTune$lambda
  
  final_model <- glmnet(
    subset(train_data,select=-c(y)),
    train_data$y,
    alpha = best_alpha,
    lambda = best_lambda
  )
  
  predictions <- predict(final_model, newx=as.matrix(subset(test_data,select=-c(y))), s = best_lambda)
  
  mse <- mean((test_data$y - predictions)^2)
  
  results[[i]] <- list(
    best_alpha = best_alpha,
    best_lambda = best_lambda,
    mse = mse
  )
}

mean_mse <- mean(sapply(results, function(res) res$mse))

best_alpha_global <- mean(sapply(results, function(res) res$best_alpha))
best_lambda_global <- mean(sapply(results, function(res) res$best_lambda))

final_model <- glmnet(
  subset(data,select=-c(y)),
  data$y,
  alpha = best_alpha_global,
  lambda = best_lambda_global
)

x <- as.matrix(subset(data,select=-c(y)))
y<-data$y
lasso_model <- glmnet(x, y, alpha = best_alpha_global, lambda = best_lambda_global)
coef_elnet <- as.matrix(coef(lasso_model, s = best_lambda_global))
selected_vars <- rownames(coef_elnet)[coef_elnet != 0]
selected_vars <- selected_vars[selected_vars != "(Intercept)"]
X.lasso <- data[, c("y", selected_vars)]

data <- X.lasso

numeric_vars <- names(data)[sapply(data, is.numeric)]
variables <- setdiff(numeric_vars, "y")

formula_text <- paste("y ~", paste(sapply(variables, function(var) paste("s(", var, ",bs='ts',k=10)", sep = "")), collapse = " + "))
formula <- as.formula(formula_text)

reg <- bam(formula, data = data,family = gaussian(),select = TRUE)


regresseur <- function(test_set) {
  library(MASS)
  library(mgcv)
  library(splines)
  library(glmnet)
  library(dplyr)
  library(caret)
  X.scaled<-predict(preproc,test_set)
  X_selected <- X.scaled[, selected_vars, drop = FALSE]
  test_set <- data.frame(X_selected)
  predict(reg, test_set)
}

# classification
library(caret)
library(glmnet)
install.packages("https://cran.r-project.org/src/contrib/themis_1.0.2.tar.gz", repos = NULL, method="libcurl")
library(dplyr)
library(MASS)

X.clas <- read.table("a24_clas_app.txt", header = TRUE)

X<-subset(X.clas,select = -c(y))
preproc <- preProcess(X, method = c("center", "scale"))
X.scaled <- predict(preproc, X)
data<-data.frame(y=X.clas$y,X.scaled)

lda.class<-lda(y~. ,data=data)
U<-lda.class$scaling
X<-as.matrix(subset(data,select = -c(y)))
Z<-X %*% U
fda<-data.frame(y=X.clas$y,Z)

fda$y<-as.factor(fda$y)



data<-fda

set.seed(123)

X_train_fin<-subset(data,select = -c(y))
y_train_fin<-data$y
  
  train_control <- trainControl(
    method = "cv", 
    number = 10, 
    search = "grid",
    allowParallel = TRUE,
    classProbs = TRUE,
    sampling = 'smote'
  )
  
  tune_grid<-expand.grid(
    mtry=c(1, 2)
  )

final_model <- train(
  X_train_fin,y_train_fin,
  method = "rf",
  tuneGrid = tune_grid,
  metric = "Accuracy",
  trControl = train_control
)

classifieur <- function(test_set) {
  library(caret)
  library(glmnet)
  library(dplyr)
  library(MASS)
  library(Matrix)
  Z<-as.matrix(test_set)%*%U
  predict(final_model, Z)
  
}

save("classifieur", "regresseur", "final_model", "preproc", "reg","U", "selected_vars", file = "env.Rdata")