# Doubly Robust EXAMPLE CODE
# Restructured Liu Framework
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

regulized <- function(vec){
  res <- (vec) / (sum(vec))
  return(res)
}


load("pop.RData")  

# draw a smaller random sample as a testing population
set.seed(44)
pop = pop[sample(1:nrow(pop), size = 1e5), ]
colnames(pop) = c('x1', 'x2', 'x3', 'x4', 'y') 
# Function
Restructured_Liu_Framework_DR <- function(P, NP, data, Nreplication = 500, seed = 1234, Logistic = F, NB = F, BART = F, DR_LM = F){
  num_cores <- parallel::detectCores()
  cl <- parallel::makeCluster(num_cores)
  doParallel::registerDoParallel(cl)
  set.seed(seed)
  regulized <- function(vec){
    res <- (vec) / (sum(vec))
    return(res)
  }
  
  #Ensure P and NP are vector of probability
  if (length(P) < sum(P) || length(NP) < sum(NP)){
    stop("NP or P must be a vector of probability, between 1.0 and 0.0")
  } else if (max(P) > 1 || max(NP) > 1){
    stop("NP or P must be a vector of probability, between 1.0 and 0.0")
  }
  
  #Check which method is available
  method_index <- 1
  method_names <- c("n_NP", "n_P", "RB_Raw", "RMSE_Raw")
  num_methods = 1 
  
  if (DR_LM == T){
    P_fitted = list()
    NP_fitted = list()
  }
  
  if (Logistic == T) {
    method_names <- c(method_names, "RB_Log", "RMSE_Log")
    Logistic_index <- method_index + 1
    method_index <- method_index + 1
    num_methods = num_methods + 1
    getLogistic <- function(m){
      sample.NP <- sample_NP_sets[[m]]
      glmO <- glm(Z ~ x1+x2+x4, data = Bsets[[m]], family = "binomial")
      Log_O <- exp(predict(glmO, newdata = sample.NP))
      Log_ind <- 1+(Weight_NP_sets[[m]]-1)/Log_O
      Log_ind <- regulized(Log_ind)
      Weight.P <- regulized(Weight_P_sets[[m]])
      
      result <- sum(Weight.P * P_fitted[[m]]) + sum(Log_ind * (sample.NP$y - NP_fitted[[m]]))
      return(result)
    }
    
  }
  
  if (NB == T) {
    method_names <- c(method_names, "RB_NB", "RMSE_NB")
    NB_index <- method_index + 1
    method_index <- method_index + 1
    num_methods = num_methods + 1
    getNB <- function(m){
      library(naivebayes)
      sample.NP <- sample_NP_sets[[m]]
      NB_model <- naive_bayes(factor(Z) ~ (x1)+(x2)+x4, 
                              data = Bsets[[m]], usekernel = F, poisson = F)
      NB_ps <- predict(NB_model, newdata = sample.NP, type = "prob")[, 2]
      NB_O <- NB_ps / (1 - NB_ps)
      NB_ind <- 1+(Weight_NP_sets[[m]]-1)/NB_O
      NB_ind <- regulized(NB_ind)
      Weight.P <- regulized(Weight_P_sets[[m]])
      
      result <- sum(Weight.P * P_fitted[[m]]) + sum(NB_ind * (sample.NP$y - NP_fitted[[m]]))
      return(result)
    }
    
  }
  
  if(BART == T){
    method_names <- c(method_names, "RB_BART", "RMSE_BART")
    BART_index <- method_index + 1
    method_index <- method_index + 1
    num_methods = num_methods + 1
    get_BART <- function(m){
      B <- Bsets[[m]]
      sample.NP <- sample_NP_sets[[m]]
      
      #estimate O
      BART_model <- dbarts::bart2(Z ~ x1 + x2 + x4, data = B, keepTrees = T, seed = m, n.threads = 8, n.chains = 2, verbose = F)
      mcmc.out = predict(BART_model, newdata = sample.NP)
      BART_pscore = colMeans(mcmc.out)
      BART_O <- BART_pscore / (1 - BART_pscore)
      
      BART_ind <- 1+(Weight_NP_sets[[m]]-1)/BART_O
      Weight.P <- regulized(Weight_P_sets[[m]])
      BART_ind <- regulized(BART_ind)
      result <- sum(Weight.P * P_fitted[[m]]) + sum(BART_ind * (sample.NP$y - NP_fitted[[m]]))
      return(result)
    }
    
  }
  
  N = nrow(data)
  
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
    Weight_P_sets = list()
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
      W.P  = 1/includP[S]
      W.NP = 1/includP[Sstar]
      
      # B set
      B = data[Sstar+S == 1, ]
      B$Z = Sstar[Sstar+S == 1]
      
      # Doubly Robust Fitted Value
      l_model = lm(y ~ x1 + x2 + x4, data = sample.NP)
      P_fitted[[m]] = predict(l_model, sample.P)
      NP_fitted[[m]] = predict(l_model, sample.NP)
      
      Bsets[[m]] <- B
      Weight_P_sets[[m]] <- W.P
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

res_DR1 <- Restructured_Liu_Framework_DR(P = c(0.01, 0.1), NP = c(0.05), Nreplication = 500, data = pop, Logistic = T, NB = T, BART = T, DR_LM = T)
write.xlsx(res_DR1$Classification_result, file = "Liu's_Framework\\output\\result_DR1.xlsx", rowNames = F)

res_DR2 <- Restructured_Liu_Framework_DR(P = c(0.01, 0.1), NP = c(0.3), Nreplication = 500, data = pop, Logistic = T, NB = T, BART = T, DR_LM = T)
write.xlsx(res_DR2$Classification_result, file = "Liu's_Framework\\output\\result_DR2.xlsx", rowNames = F)

res_DR3 <- Restructured_Liu_Framework_DR(P = c(0.01, 0.1), NP = c(0.5), Nreplication = 500, data = pop, Logistic = T, NB = T, BART = T, DR_LM = T)
write.xlsx(res_DR3$Classification_result, file = "Liu's_Framework\\output\\result_DR3.xlsx", rowNames = F)