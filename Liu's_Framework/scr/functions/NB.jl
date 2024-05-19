
using NaiveBayes, DataFrames, RCall
R"""
library(naivebayes)
"""

NBayes = function (Formula, train, test)
    feature_index = collect(Symbol.(Formula.rhs))
    model = GaussianNB(train[!, Symbol.(Formula.lhs)], length(feature_index))
    fit(model, Matrix(permutedims(train[!, feature_index])), train[!, Symbol.(Formula.lhs)])
    #pred = NaiveBayes.predict_logprobs(model, restructure_matrix(Matrix(permutedims(test[:, feature_index]))), Dict{Symbol, Vector{Int}}())
    pred = NaiveBayes.predict_logprobs(model, (Matrix(permutedims(test[:, feature_index]))))
    ps = exp.(pred[2])

    ps_ratio = ps[2, :] ./ ps[1, :]
    return ps_ratio
end

NBayes2 = function (Formula, train, test)
    feature_index = collect(Symbol.(Formula.rhs))
    model = GaussianNB(train[!, Symbol.(Formula.lhs)], length(feature_index))
    fit(model, Matrix(permutedims(train[!, feature_index])), train[!, Symbol.(Formula.lhs)])
    #pred = NaiveBayes.predict_logprobs(model, restructure_matrix(Matrix(permutedims(test[:, feature_index]))), Dict{Symbol, Vector{Int}}())
    pred = NaiveBayes.predict_logprobs(model, (Matrix(permutedims(test[:, feature_index]))))[2]
    ps = exp.(pred[1, :]) ./ (exp.(pred[1, :]) .+ exp.(pred[2, :]))

    
    return ps
end

NBayesKDE = function (Formula, train, test)
    feature_index = collect(Symbol.(Formula.rhs))
    model = HybridNB(train[!, Symbol.(Formula.lhs)])
    fit(model, Matrix(permutedims(train[!, feature_index])), train[!, Symbol.(Formula.lhs)])
    pred = NaiveBayes.predict_logprobs(model, restructure_matrix(Matrix(permutedims(test[:, feature_index]))), Dict{Symbol, Vector{Int}}())
    #pred = NaiveBayes.predict_logprobs(model, (Matrix(permutedims(test[:, feature_index]))))
    ps = exp.(pred)

    ps_ratio = ps[2, :] ./ ps[1, :]
    return ps_ratio
end

NBayesKDE2 = function (Formula, train, test)
    feature_index = collect(Symbol.(Formula.rhs))
    model = HybridNB(train[!, Symbol.(Formula.lhs)])
    fit(model, Matrix(permutedims(train[!, feature_index])), train[!, Symbol.(Formula.lhs)])
    pred = NaiveBayes.predict_logprobs(model, restructure_matrix(Matrix(permutedims(test[:, feature_index]))), Dict{Symbol, Vector{Int}}())
    #pred = NaiveBayes.predict_logprobs(model, (Matrix(permutedims(test[:, feature_index]))))
    ps = exp.(pred[1, :])

    return ps
end

R_NBayes_KDE = function (Formula, train, test)
    f = string(Formula)
    rtrain = robject(train)
    rtest = robject(test)
    R"""
    R_formula <- update(as.formula($f), factor(.) ~ .)
    nbO = naive_bayes(R_formula, data = $rtrain, usekernel = F, poisson = F);
    psO = predict(nbO, newdata = $rtest, type = "prob")[, 2];
    O = psO / (1 - psO);
    """
    @rget O
    return O
end

R_NBayes_KDE2 = function (Formula, train, test)
    f = string(Formula)
    rtrain = robject(train)
    rtest = robject(test)
    R"""
    R_formula <- update(as.formula($f), factor(.) ~ .)
    nbO = naive_bayes(R_formula, data = $rtrain, usekernel = F, poisson = F);
    L = 1 - predict(nbO, newdata = $rtest, type = "prob")[, 2];
    """
    @rget L
    return L
end