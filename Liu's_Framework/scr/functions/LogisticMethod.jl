# Import Packages
using CSV, Statistics, DataFrames, DelimitedFiles, RDatasets, ProgressBars, GLM, StatsModels, Fetch
using Distributions, Roots# like uniroot in R
using Tables, Plots, ThreadSafeDicts, ProgressMeter, Random

LogisticMethod = function (Dict_Prob_Sample, Dict_NonProb_Sample, Dict_B, Formula; Weight_NP = Weight_NP, Weight_P = Weight_P)
    # Calculate O by Logistic Regression
    Logistic_f_O = Term(:Z) ~ sum(Formula.rhs[1:5]) 
    glmO = glm(Logistic_f_O, Dict_B, Binomial(), LogitLink())
    Logistic_psO = predict(glmO, DataFrame(Dict_NonProb_Sample))
        
    Logistic_Oᵢ = Logistic_psO ./ (1 .- Logistic_psO)

    # Calculate L by Logistic Regrerssion
    Dict_Prob_Sample.S★ .= Int.(S★[S .== 1])
    Logistic_f_L = Term(:S★) ~ sum(Formula.rhs[1:5])
    glmL = glm(Logistic_f_L, Dict_Prob_Sample, Binomial(), LogitLink())
    Logistic_Lᵢ = 1 .- predict(glmL, Dict_NonProb_Sample)


    #formula (11) Dep
    fdep = (Weight_NP .- 1) ./ (Logistic_Oᵢ .* Logistic_Lᵢ) 

    # formula (18) Ind
    find = 1 .+ (Weight_NP .- 1) ./ Logistic_Oᵢ

    return find, fdep
end