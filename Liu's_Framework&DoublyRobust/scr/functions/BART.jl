using DataFrames, RCall
R"""
library(dbarts)
"""

BART_fun = function (Formula, train, test, m)
    f = string(Formula)
    @rput train
    @rput test
    @rput m
    R"""
    B <- train
    sample_NP <- test
    R_formula <- update(as.formula($f), . ~ .)
    BART_model <- dbarts::bart2(R_formula, data = B, keepTrees = T, seed = m, n.threads = 12, n.chains = 2, verbose = F)
    mcmc.out <- predict(BART_model, newdata = sample_NP)
    BART_pscore <- colMeans(mcmc.out)
    BART_O <- BART_pscore / (1 - BART_pscore)
    """
    BART_O = @rget BART_O
    return BART_O
end
