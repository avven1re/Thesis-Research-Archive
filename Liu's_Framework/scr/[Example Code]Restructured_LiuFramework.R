# Restructured Liu Fraemwork EXAMPLE CODE
library(sampling)  
library(lubridate)
library(progress)
library(openxlsx)
library(dplyr)
library(naivebayes)
library(caret) #Confusion matrix
library(MASS)
library(party)
library(rpart.plot)
library(WeightIt)
library(BART)
library(dbarts)
library(e1071)
library(parallel)
library(xgboost)
#num_cores <- detectCores()
library(foreach)
library(doParallel)
#library(doSNOW)
#library(doMPI)
#cl <- makeCluster(num_cores)
#registerDoParallel(cl)
#?UPrandomsystematic


load("pop.RData")  

# draw a smaller random sample as a testing population
set.seed(44)
pop = pop[sample(1:nrow(pop), size = 1e5), ]
colnames(pop) = c('x1', 'x2', 'x3', 'x4', 'y') 
# Function
Restructured_Liu_Framework <- function(P, NP, data, Nreplication = 500, seed = 1234, Logistic = F, NB = F, XGB = F, SVM = F, SVM_P = F, RF = F, BART = F){
  num_cores <- parallel::detectCores()
  cl <- parallel::makeCluster(num_cores)
  doParallel::registerDoParallel(cl)
  set.seed(seed)
  
  #Ensure P and NP are vector of probability
  if (length(P) < sum(P) || length(NP) < sum(NP)){
    stop("NP or P must be a vector of probability, between 1.0 and 0.0")
  } else if (max(P) > 1 || max(NP) > 1){
    stop("NP or P must be a vector of probability, between 1.0 and 0.0")
  }
  
  #Check which method is available
  method_index <- 1
  method_names <- c("n_NP", "n_P", "RB_Raw", "RMSE_Raw")
  
  if (Logistic == T) {
    method_names <- c(method_names, "RB_Log", "RMSE_Log")
    Logistic_index <- method_index + 1
    method_index <- method_index + 1
    getLogistic <- function(m){
      sample.NP <- sample_NP_sets[[m]]
      glmO <- glm(Z ~ x1+x2+x4, data = Bsets[[m]], family = "binomial")
      Log_O <- exp(predict(glmO, newdata = sample.NP))
      Log_ind <- 1+(Weight_NP_sets[[m]]-1)/Log_O
      result <- sum(sample.NP$y*Log_ind)/sum(Log_ind)
      return(result)
    }
  }
  
  if (NB == T) {
    method_names <- c(method_names, "RB_NB", "RMSE_NB")
    NB_index <- method_index + 1
    method_index <- method_index + 1
    getNB <- function(m){
      library(naivebayes)
      sample.NP <- sample_NP_sets[[m]]
      NB_model <- naive_bayes(factor(Z) ~ (x1)+(x2)+x4, 
                             data = Bsets[[m]], usekernel = F, poisson = F)
      NB_ps <- predict(NB_model, newdata = sample.NP, type = "prob")[, 2]
      NB_O <- NB_ps / (1 - NB_ps)
      NB_ind <- 1+(Weight_NP_sets[[m]]-1)/NB_O
      result <- sum(sample.NP$y*NB_ind)/sum(NB_ind)
      return(result)
    }
  }
  
  if (XGB == T) {
    method_names <- c(method_names, "RB_XGB", "RMSE_XGB")
    XGB_index <- method_index + 1
    method_index <- method_index + 1
    getXGB <- function(m){
      library(xgboost)
      sample.NP <- sample_NP_sets[[m]]
      B <- Bsets[[m]]
      B_IV <- B[, c(1, 2, 4)]
      B_DV <- B$Z
      XGB_model <- xgboost(data = as.matrix(B_IV), label = B_DV, objective = "binary:logistic", nrounds = 100, max_depth = 10, eta = 0.3, verbose = 0)
      XGB_ps <- predict(XGB_model,  newdata = as.matrix(sample.NP[, c(1, 2, 4)]))
      XGB_O <- XGB_ps / (1 - XGB_ps)
      XGB_ind = 1+(Weight_NP_sets[[m]]-1)/XGB_O
      result <- sum(sample.NP$y*XGB_ind)/sum(XGB_ind)
      return(result)
    }
  }
  
  if (SVM == T) {
    method_names <- c(method_names, "RB_SVM", "RMSE_SVM")
    SVM_index <- method_index + 1
    method_index <- method_index + 1
    getSVM <- function(m){
      library(e1071)
      B <- Bsets[[m]]
      sample.NP <- sample_NP_sets[[m]]
      
      SVM_model <- svm(factor(Z) ~ (x1)+(x2)+x4, data = B, type = 'C-classification', 
                       probability = T, kernel = "linear")
      SVM_pred <- predict(SVM_model, newdata = sample.NP, probability = T)
      SVM_ps <- attr(SVM_pred, "probabilities")[, 1]
      SVM_O <- SVM_ps / (1 - SVM_ps)
      SVM_ind <- 1+(Weight_NP_sets[[m]]-1)/SVM_O
      result <- sum(sample.NP$y*SVM_ind)/sum(SVM_ind)
      return(result)
    }
  }
  
  if (SVM_P == T) {
    method_names <- c(method_names, "RB_SVM_P", "RMSE_SVM_P")
    SVM_P_index <- method_index + 1
    method_index <- method_index + 1
    getSVM_P <- function(m){
      library(e1071)
      B <- Bsets[[m]]
      sample.NP <- sample_NP_sets[[m]]
      
      SVM_model <- svm(factor(Z) ~ (x1)+(x2)+x4, data = B, type = 'C-classification', 
                       probability = T, kernel = "polynomial")
      SVM_pred <- predict(SVM_model, newdata = sample.NP, probability = T)
      SVM_ps <- attr(SVM_pred, "probabilities")[, 1]
      SVM_O <- SVM_ps / (1 - SVM_ps)
      SVM_ind <- 1+(Weight_NP_sets[[m]]-1)/SVM_O
      result <- sum(sample.NP$y*SVM_ind)/sum(SVM_ind)
      return(result)
    }
  }
  
  
  if (RF == T) {
    method_names <- c(method_names, "RB_RF", "RMSE_RF")
    RF_index <- method_index + 1
    method_index <- method_index + 1
    getRF <- function(m){
      library(randomForest)
      B <- Bsets[[m]]
      sample.NP <- sample_NP_sets[[m]]
      
      # estimate O
      RF_model = randomForest::randomForest(factor(Z) ~ (x1)+(x2)+x4, 
                                       data = B, mtry = 3, ntree = 500)
      RF_ps = predict(RF_model, newdata = sample.NP, type = "prob")[, 2]
      
      for (z in 1:length(RF_ps)) {
        
        if (RF_ps[z] == 1){
          RF_ps[z] <- 499/500
        }
        if (RF_ps[z] == 0){
          RF_ps[z] <- 1/500
        }
      }
      
      RF_O = RF_ps / (1 - RF_ps)
      
      RF_ind <- 1+(Weight_NP_sets[[m]]-1)/RF_O
      result <- sum(sample.NP$y*RF_ind)/sum(RF_ind)
      return(result)
    }
  }
  
  if(BART == T){
    method_names <- c(method_names, "RB_BART", "RMSE_BART")
    BART_index <- method_index + 1
    method_index <- method_index + 1
    
    get_BART <- function(m){
      B <- Bsets[[m]]
      sample.NP <- sample_NP_sets[[m]]
      
      #estimate O
      BART_model <- dbarts::bart2(Z ~ x1 + x2 + x4, data = B, keepTrees = T, seed = m, n.threads = 8, n.chains = 2, verbose = F)
      mcmc.out = predict(BART_model, newdata = sample.NP)
      BART_pscore = colMeans(mcmc.out)
      BART_O <- BART_pscore / (1 - BART_pscore)
      
      BART_ind <- 1+(Weight_NP_sets[[m]]-1)/BART_O
      result <- sum(sample.NP$y*BART_ind)/sum(BART_ind)
      return(result)
    }
  }
  
  N = nrow(data)
  num_methods = 1 + Logistic + NB + XGB + SVM + SVM_P + RF + BART
  
  #Define the scenario
  scenario <- expand.grid(P, NP)
  NumberOfScenario <- nrow(scenario)
  
  #Create empty matrix for result
  result_numcol <- 2 + 2 + 2 * (num_methods - 1)
  Classification_result = matrix(NaN, NumberOfScenario, result_numcol)
  iter_y_matrix <- list()
  
  # Calculate the predicted mean in each scenario and replications
  #foreach (i = 1 : NumberOfScenario) %do% {
  for (i in 1 : NumberOfScenario) {
    InclusionFraction_P = scenario[i, 1] ; InclusionFraction_NP = scenario[i, 2]
    set.seed(12345)
    ybarm_matrix <- matrix(NaN, nrow = Nreplication, ncol = num_methods)
    
    Bsets = list()
    Weight_NP_sets = list()
    sample_NP_sets = list()
    sample_P_sets = list()
    
    #foreach (m = 1:Nreplication) %do% {
    for (m in 1 : Nreplication) {
      set.seed(m)
      # probability sample 
      x = data$x1/30
      f = function(theta) sum(exp(theta + x) / (1 + exp(theta + x))) - InclusionFraction_P*N
      theta = uniroot(f, c(-100, 100))$root
      includP = exp(theta + x) / (1 + exp(theta + x))
      S = as.logical(UPrandomsystematic(includP))
      sample.P = data[S,-5]
      
      # nonprobability sample 
      x = data$x2 - data$x4/20 
      
      f = function(theta) sum(exp(theta + x) / (1 + exp(theta + x))) - InclusionFraction_NP*N
      theta = uniroot(f, c(-100, 100))$root
      includNP = exp(theta + x) / (1 + exp(theta + x))
      Sstar = as.logical(UPrandomsystematic(includNP))
      sample.NP = data[Sstar,]
      
      # design weight
      #W.P  = 1/includP[S]
      W.NP = 1/includP[Sstar]
      
      # B set
      B = data[Sstar+S == 1, ]
      B$Z = Sstar[Sstar+S == 1]
      
      Bsets[[m]] <- B
      Weight_NP_sets[[m]] <- W.NP
      sample_NP_sets[[m]] <- sample.NP
      sample_P_sets[[m]] <- sample.P
      
      # Raw result
      ybarm_matrix[m, 1] = mean(sample.NP$y)
    }
    
    # Logistic Regression
    if (Logistic){
    Log_res <- foreach(m = 1 : Nreplication, .combine = rbind) %dopar% {
    #for (m in 1 : Nreplication) {
      set.seed(m)
      getLogistic(m)
    }
    ybarm_matrix[, Logistic_index] <- Log_res
    }
    #Naive Bayes
    if(NB){
    NB_res <- foreach(m = 1: Nreplication, .combine = rbind) %dopar% {
    #for (m in 1 : Nreplication) {
      set.seed(m)
      getNB(m)
    }
    ybarm_matrix[, NB_index] <- NB_res
    }
    #XGBoost
    if(XGB){
    XGB_res <- foreach(m = 1 : Nreplication, .combine = rbind) %dopar% {
    #for (m in 1 : Nreplication) {
      set.seed(m)
      getXGB(m)
    }
    ybarm_matrix[, XGB_index] <- XGB_res
    }
    #LIBSVM
    if(SVM){
    SVM_res <- foreach(m = 1 : Nreplication, .combine = rbind) %dopar% {
    #for (m in 1 : Nreplication) {
      set.seed(m)
      
      getSVM(m)
    }
    ybarm_matrix[, SVM_index] <- SVM_res
    }
    
    #LIBSVM_polynomial
    if(SVM_P){
      SVM_P_res <- foreach(m = 1 : Nreplication, .combine = rbind) %dopar% {
        #for (m in 1 : Nreplication) {
        set.seed(m)
        
        getSVM_P(m)
      }
      ybarm_matrix[, SVM_P_index] <- SVM_P_res
    }
    
    #RandomForest
    if(RF){
      RF_res <- foreach(m = 1 : Nreplication, .combine = rbind) %dopar% {
        #for (m in 1 : Nreplication) {
        set.seed(m)
        
        getRF(m)
      }
      ybarm_matrix[, RF_index] <- RF_res
    }
    
    #BART
    if(BART){
      BART_res <- foreach(m = 1 : Nreplication, .combine = rbind) %dopar% {
        #for (m in 1 : Nreplication) {
        set.seed(m)
        
        get_BART(m)
      }
      ybarm_matrix[, BART_index] <- BART_res
    }
    
    Relative_Bias_Raw <- (sum(ybarm_matrix[, 1] - mean(data$y)) / mean(data$y)) / nrow(ybarm_matrix) * 100
    RMSE_Raw <- (sum((ybarm_matrix[, 1] - mean(data$y)) ^2) / nrow(ybarm_matrix))^(1/2)
    
    Classification_result[i, 1] <- InclusionFraction_NP
    Classification_result[i, 2] <- InclusionFraction_P
    Classification_result[i, 3] <- Relative_Bias_Raw
    Classification_result[i, 4] <- RMSE_Raw
    
    if (Logistic) {
      Relative_Bias_Log <- (sum(ybarm_matrix[, Logistic_index] - mean(data$y)) / mean(data$y)) / nrow(ybarm_matrix) * 100
      RMSE_Log <- (sum((ybarm_matrix[, Logistic_index] - mean(data$y)) ^2) / nrow(ybarm_matrix))^(1/2)
      Classification_result[i, 2*Logistic_index + 1] = Relative_Bias_Log
      Classification_result[i, 2*Logistic_index + 2] = RMSE_Log
    }
    
    if (NB) {
      Relative_Bias_NB <- (sum(ybarm_matrix[, NB_index] - mean(data$y)) / mean(data$y)) / nrow(ybarm_matrix) * 100
      RMSE_NB <- (sum((ybarm_matrix[, NB_index] - mean(data$y)) ^2) / nrow(ybarm_matrix))^(1/2)
      Classification_result[i, 2*NB_index + 1] = Relative_Bias_NB
      Classification_result[i, 2*NB_index + 2] = RMSE_NB
    }
    
    if (XGB) {
      Relative_Bias_XGB <- (sum(ybarm_matrix[, XGB_index] - mean(data$y)) / mean(data$y)) / nrow(ybarm_matrix) * 100
      RMSE_XGB <- (sum((ybarm_matrix[, XGB_index] - mean(data$y)) ^2) / nrow(ybarm_matrix))^(1/2)
      Classification_result[i, 2*XGB_index + 1] = Relative_Bias_XGB
      Classification_result[i, 2*XGB_index + 2] = RMSE_XGB
    }
    
    if (SVM) {
      Relative_Bias_SVM <- (sum(ybarm_matrix[, SVM_index] - mean(data$y)) / mean(data$y)) / nrow(ybarm_matrix) * 100
      RMSE_SVM <- (sum((ybarm_matrix[, SVM_index] - mean(data$y)) ^2) / nrow(ybarm_matrix))^(1/2)
      Classification_result[i, 2*SVM_index + 1] = Relative_Bias_SVM
      Classification_result[i, 2*SVM_index + 2] = RMSE_SVM
    }
    
    if (SVM_P) {
      Relative_Bias_SVM_P <- (sum(ybarm_matrix[, SVM_P_index] - mean(data$y)) / mean(data$y)) / nrow(ybarm_matrix) * 100
      RMSE_SVM_P <- (sum((ybarm_matrix[, SVM_P_index] - mean(data$y)) ^2) / nrow(ybarm_matrix))^(1/2)
      Classification_result[i, 2*SVM_P_index + 1] = Relative_Bias_SVM_P
      Classification_result[i, 2*SVM_P_index + 2] = RMSE_SVM_P
    }
    
    if (RF) {
      Relative_Bias_RF <- (sum(ybarm_matrix[, RF_index] - mean(data$y)) / mean(data$y)) / nrow(ybarm_matrix) * 100
      RMSE_RF <- (sum((ybarm_matrix[, RF_index] - mean(data$y)) ^2) / nrow(ybarm_matrix))^(1/2)
      Classification_result[i, 2*RF_index + 1] = Relative_Bias_RF
      Classification_result[i, 2*RF_index + 2] = RMSE_RF
    }
    
    if (BART) {
      Relative_Bias_BART <- (sum(ybarm_matrix[, BART_index] - mean(data$y)) / mean(data$y)) / nrow(ybarm_matrix) * 100
      RMSE_BART <- (sum((ybarm_matrix[, BART_index] - mean(data$y)) ^2) / nrow(ybarm_matrix))^(1/2)
      Classification_result[i, 2*BART_index + 1] = Relative_Bias_BART
      Classification_result[i, 2*BART_index + 2] = RMSE_BART
    }
    
    iter_y_matrix[[i]] = ybarm_matrix
  }
  Classification_result = as.data.frame(Classification_result)
  names(Classification_result) <- method_names
  parallel::stopCluster(cl)
  return(list(Classification_result = Classification_result, iter_y_matrix = iter_y_matrix))
}
res1 <- Restructured_Liu_Framework(P = c(0.01, 0.1), NP = c(0.05), Nreplication = 500, data = pop, Logistic = T, NB = T, XGB = T, SVM = T, SVM_P = T, RF = T, BART = T)
write.xlsx(res1$Classification_result, file = "Liu's_Framework\\output\\result1.xlsx", rowNames = F)

res2 <- Restructured_Liu_Framework(P = c(0.01, 0.1), NP = c(0.3), Nreplication = 500, data = pop, Logistic = T, NB = T, XGB = T, SVM = T, SVM_P = T, RF = T, BART = T)
write.xlsx(res2$Classification_result, file = "Liu's_Framework\\output\\result2.xlsx", rowNames = F)

res3 <- Restructured_Liu_Framework(P = c(0.01, 0.1), NP = c(0.5), Nreplication = 500, data = pop, Logistic = T, NB = T, XGB = T, SVM = T, SVM_P = T, RF = T, BART = T)
write.xlsx(res3$Classification_result, file = "Liu's_Framework\\output\\result3.xlsx", rowNames = F)
