# Import Packages
using CSV, Statistics, DataFrames, DelimitedFiles, RDatasets, ProgressBars, GLM, StatsModels, Fetch, RCall, LIBSVM
using Distributions, Roots# like uniroot in R
using Tables, ThreadSafeDicts, ProgressMeter, Random, XGBoost, DecisionTree, Plots
import MLJ#, MLJNaiveBayesInterface
import NaiveBayes


include("functions/UPrandomsystematic.jl")  #* The inclusion mechanism of Porbabiliy Sampling form liu et al.
include("functions/Bisection.jl")           
include("../Data/DataGeneration.jl")      #* Import the simulated data by me.
include("functions/NB.jl")
include("functions/SRS.jl")
include("functions/BART.jl")
include("functions/normalized.jl")

df = Simulate_Data()
#df = Simulate_Data2()

Restructured_Liu_framework_DR = function (P::Vector, NP::Vector, data::DataFrame, Formula::FormulaTerm; Nreplication = 500,
    seed = 1234, Logistic = false, NB = false, BART = false, DR_LM = false)
    Random.seed!(seed) 
    # Check P and NP is vector of probability 
    if (length(P) < sum(P) || length(NP) < sum(NP))
        return error("NP or P must be a vector of probabilty, between 1.0 and 0.0")
    elseif (sort(P)[end] > 1 || sort(NP)[end] > 1)
        return error("NP or P must be a vector of probabilty, between 1.0 and 0.0")
    end
    
    # Check which method is available
    num_methods = 1 
    method_index = 1
    method_names = ["n_NP", "n_P", "RB_Raw", "RMSE_Raw"]

    if DR_LM == true
        NP_fitted = ThreadSafeDict()
        P_fitted = ThreadSafeDict()
    end

    if Logistic == true
        push!(method_names, "RB_Log")
        push!(method_names, "RMSE_Log")
        Logistic_index = method_index + 1
        method_index = method_index + 1
        num_methods = num_methods + 1
        
        if DR_LM == true
            push!(method_names, "RB_Log_DR_lm")
            push!(method_names, "RMSE_Log_DR_lm")
            Logistic_DR_lm_index = method_index + 1
            method_index = method_index + 1
            num_methods = num_methods + 1
        end
    end

    if NB == true
        push!(method_names, "RB_NB")
        push!(method_names, "RMSE_NB")
        NB_index = method_index + 1
        method_index = method_index + 1
        num_methods = num_methods + 1

        if DR_LM == true
            push!(method_names, "RB_NB_DR_lm")
            push!(method_names, "RMSE_NB_DR_lm")
            NB_DR_lm_index = method_index + 1
            method_index = method_index + 1
            num_methods = num_methods + 1
        end
    end

    if BART == true
        push!(method_names, "RB_BART")
        push!(method_names, "RMSE_BART")
        BART_index = method_index + 1
        method_index = method_index + 1
        num_methods = num_methods + 1

        if DR_LM == true
            push!(method_names, "RB_BART_DR_lm")
            push!(method_names, "RMSE_BART_DR_lm")
            BART_DR_lm_index = method_index + 1
            method_index = method_index + 1
            num_methods = num_methods + 1
        end
    end    

    N = nrow(data)
    IV_index = collect(Symbol.(Formula.rhs))
    DV_index = (Symbol.(Formula.lhs))

    # Define the scenario
    scenario = collect(Base.product(NP, P))
    NumberOfScenario = length(scenario)

    #Create empty matrix for result
    result_numcol = 2 + 2 + 2 * (num_methods - 1)
    Classification_result = Array{Float64}(undef, NumberOfScenario, result_numcol) #! The size of the results!
    iter_y_matrix = ThreadSafeDict()

    # Calculate the predicted mean in each scenario and replications
    for i in ProgressBar(1 : NumberOfScenario)
        InclusionFraction_NP, InclusionFraction_P = scenario[i]
        #scenario_seed = 100000 * InclusionFraction_NP * InclusionFraction_P
        Random.seed!(12345)
        ȳₘ_matrix = Array{Float64}(undef, Nreplication, num_methods)      #! The size of ȳₘ matrix
        P_size = InclusionFraction_P * N

        formula_O = Term(:Z) ~ sum(Formula.rhs[1:end])
        formula_DR = @formula(y ~ x1 + x5 + x7 + x9 + x11)
        Z_index = Symbol(formula_O.lhs)
        Bsets = ThreadSafeDict()
        Weight_NP_sets = ThreadSafeDict()
        Weight_P_sets = ThreadSafeDict()
        sample_NP_sets = ThreadSafeDict()
        sample_P_sets = ThreadSafeDict()
        ##Calculate the inclusion probability for the dataset
        #Probability Sample
        includeP = repeat([InclusionFraction_P], N)

        #Non-probability sample
        x = data[!, DV_index] ./ 100 #! y  , devided by 100 for df; 10 for df2
        #x = data[!, DV_index] ./ 10000
        theta = find_zero(theta -> sum(exp.(theta .+ x) ./ (1 .+ exp.(theta .+ x))) - InclusionFraction_NP * N, [-30, 10])  #![-30, 10] for df; [-300, 100] for df2
        includeNP = exp.(theta .+ x) ./ (1 .+ exp.(theta .+ x))

        # Sampling
        Threads.@threads for m in 1 : Nreplication
            Random.seed!(m)
            S = SRS(P_size, N)
            sample_P = data[S, :]

            S_star = Bool.(round.(UPrandomsystematic(includeNP), digits = 0))
            sample_NP = data[S_star, :]

            #Design Weight
            Weight_P  = 1 ./ includeP[S]
            Weight_NP = 1 ./ includeP[S_star]

            # B set
            B = data[S_star .+ S .== 1, :]
            B.Z .= Int.(S_star[S_star .+ S .== 1])

            Bsets[m] = B
            Weight_NP_sets[m] = Weight_NP
            Weight_P_sets[m] = Weight_P
            sample_NP_sets[m] = sample_NP
            sample_P_sets[m] = sample_P
            
            #Raw result
            ȳₘ_matrix[m, 1] = mean(sample_NP[!, DV_index]) 
        end

        # Doubly Robust with Linear model
        Threads.@threads for m in 1 : Nreplication

            if DR_LM != true
                break
            end

            Random.seed!(m)
            l_model =  lm(formula_DR, DataFrame(sample_NP_sets[m])) # Formula or formula_DR
            P_fitted[m] = GLM.predict(l_model, DataFrame(sample_P_sets[m]))
            NP_fitted[m] = GLM.predict(l_model, DataFrame(sample_NP_sets[m]))
        end

        # Logistic Regression
        Threads.@threads for m in 1 : Nreplication
            Random.seed!(m)
            if Logistic != true
                break
            end
            glmO = glm(formula_O, Bsets[m], Binomial(), LogitLink())
            sample_NP = sample_NP_sets[m]
            Logistic_psO = GLM.predict(glmO, DataFrame(sample_NP))
            Logistic_Oᵢ = Logistic_psO ./ (1 .- Logistic_psO)
            #Calculate the psuedo weight
            Log_find = 1 .+ (Weight_NP_sets[m] .- 1) ./ Logistic_Oᵢ
            ȳₘ_matrix[m, Logistic_index] = sum(sample_NP[!, DV_index] .* Log_find) / sum(Log_find)

            if DR_LM == true
                ȳₘ_matrix[m, Logistic_DR_lm_index] = sum(normalized(Weight_P_sets[m]) .* P_fitted[m]) + sum(normalized(Log_find) .* (sample_NP[!, DV_index] .- NP_fitted[m]))
            end
        end

        #Naive Bayes
        Threads.@threads for m in 1 : Nreplication
            Random.seed!(m)
            if NB != true
                break
            end
            sample_NP = sample_NP_sets[m]
            NB_Oᵢ = NBayes(formula_O, Bsets[m], sample_NP)
            #NB_Oᵢ = R_NBayes_KDE(formula_O, Bsets[m], sample_NP)

            #Calculate the psuedo weight
            NB_find = 1 .+ (Weight_NP_sets[m] .- 1) ./ NB_Oᵢ
            ȳₘ_matrix[m, NB_index] = sum(sample_NP[!, DV_index] .* NB_find) / sum(NB_find)

            if DR_LM == true
                ȳₘ_matrix[m, NB_DR_lm_index] = sum(normalized(Weight_P_sets[m]) .* P_fitted[m]) + sum(normalized(NB_find) .* (sample_NP[!, DV_index] .- NP_fitted[m]))
            end
        end

        # BART
        for m in ProgressBar(1 : Nreplication)
            Random.seed!(m)
            if BART != true
                break
            end
            B = Bsets[m]
            sample_NP = sample_NP_sets[m]
            
            BART_Oᵢ = BART_fun(formula_O, B, sample_NP, m)

            BART_find = 1 .+ (Weight_NP_sets[m] .- 1) ./ BART_Oᵢ
            ȳₘ_matrix[m, BART_index] = sum(sample_NP[:, 1] .* BART_find) / sum(BART_find)
            
            if DR_LM == true
                ȳₘ_matrix[m, BART_DR_lm_index] = sum(normalized(Weight_P_sets[m]) .* P_fitted[m]) + sum(normalized(BART_find) .* (sample_NP[!, DV_index] .- NP_fitted[m]))
            end
        end
        
        Relative_Bias_Raw = (sum(ȳₘ_matrix[:, 1] .- mean(data[:, DV_index])) / mean(data[:, DV_index])) / size(ȳₘ_matrix)[1] * 100
        RMSE_Raw = (sum((ȳₘ_matrix[:, 1] .- mean(data[:, DV_index])) .^ 2) / size(ȳₘ_matrix)[1]) ^ (1/2)

        Classification_result[i, 1] = InclusionFraction_NP
        Classification_result[i, 2] = InclusionFraction_P
        Classification_result[i, 3] = Relative_Bias_Raw
        Classification_result[i, 4] = RMSE_Raw

        if Logistic == true
            Relative_Bias_Ind_Log = (sum(ȳₘ_matrix[:, Logistic_index] .- mean(data[:, DV_index])) / mean(data[:, DV_index])) / size(ȳₘ_matrix)[1] * 100
            RMSE_Ind_Log = (sum((ȳₘ_matrix[:, Logistic_index] .- mean(data[:, DV_index])) .^ 2) / size(ȳₘ_matrix)[1]) ^ (1/2)
            Classification_result[i, 2*Logistic_index + 1] = Relative_Bias_Ind_Log
            Classification_result[i, 2*Logistic_index + 2] = RMSE_Ind_Log

            if DR_LM == true
                Relative_Bias_Ind_Log_DR_lm = (sum(ȳₘ_matrix[:, Logistic_DR_lm_index] .- mean(data[:, DV_index])) / mean(data[:, DV_index])) / size(ȳₘ_matrix)[1] * 100
                RMSE_Ind_Log_DR_lm = (sum((ȳₘ_matrix[:, Logistic_DR_lm_index] .- mean(data[:, DV_index])) .^ 2) / size(ȳₘ_matrix)[1]) ^ (1/2)
                Classification_result[i, 2*Logistic_DR_lm_index + 1] = Relative_Bias_Ind_Log_DR_lm
                Classification_result[i, 2*Logistic_DR_lm_index + 2] = RMSE_Ind_Log_DR_lm
            end
        end

        if NB == true
            Relative_Bias_Ind_NB = (sum(ȳₘ_matrix[:, NB_index] .- mean(data[:, DV_index])) / mean(data[:, DV_index])) / size(ȳₘ_matrix)[1] * 100
            RMSE_Ind_NB = (sum((ȳₘ_matrix[:, NB_index] .- mean(data[:, DV_index])) .^ 2) / size(ȳₘ_matrix)[1]) ^ (1/2)
            Classification_result[i, 2*NB_index + 1] = Relative_Bias_Ind_NB
            Classification_result[i, 2*NB_index + 2] = RMSE_Ind_NB

            if DR_LM == true
                Relative_Bias_Ind_NB_DR_lm = (sum(ȳₘ_matrix[:, NB_DR_lm_index] .- mean(data[:, DV_index])) / mean(data[:, DV_index])) / size(ȳₘ_matrix)[1] * 100
                RMSE_Ind_NB_DR_lm = (sum((ȳₘ_matrix[:, NB_DR_lm_index] .- mean(data[:, DV_index])) .^ 2) / size(ȳₘ_matrix)[1]) ^ (1/2)
                Classification_result[i, 2*NB_DR_lm_index + 1] = Relative_Bias_Ind_NB_DR_lm
                Classification_result[i, 2*NB_DR_lm_index + 2] = RMSE_Ind_NB_DR_lm
            end
        end

        if BART == true
            Relative_Bias_Ind_BART = (sum(ȳₘ_matrix[:, BART_index] .- mean(data[:, DV_index])) / mean(data[:, DV_index])) / size(ȳₘ_matrix)[1] * 100
            RMSE_Ind_BART = (sum((ȳₘ_matrix[:, BART_index] .- mean(data[:, DV_index])) .^ 2) / size(ȳₘ_matrix)[1]) ^ (1/2)
            Classification_result[i, 2*BART_index + 1] = Relative_Bias_Ind_BART
            Classification_result[i, 2*BART_index + 2] = RMSE_Ind_BART

            if DR_LM == true
                Relative_Bias_Ind_BART_DR_lm = (sum(ȳₘ_matrix[:, BART_DR_lm_index] .- mean(data[:, DV_index])) / mean(data[:, DV_index])) / size(ȳₘ_matrix)[1] * 100
                RMSE_Ind_BART_DR_lm = (sum((ȳₘ_matrix[:, BART_DR_lm_index] .- mean(data[:, DV_index])) .^ 2) / size(ȳₘ_matrix)[1]) ^ (1/2)
                Classification_result[i, 2*BART_DR_lm_index + 1] = Relative_Bias_Ind_BART_DR_lm
                Classification_result[i, 2*BART_DR_lm_index + 2] = RMSE_Ind_BART_DR_lm
            end
        end

        iter_y_matrix[i] = ȳₘ_matrix
    end
        Classification_result = DataFrame(Classification_result, :auto)
        result_names = Symbol.(method_names)
        rename!(Classification_result, result_names)  
        

        return Classification_result, iter_y_matrix
end

ff = @formula(y ~ x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10 + x11 + x12 + x13 + x14 + x15)
lmodel = lm(ff, df)
r2(lmodel) # R-squared = 0.99

# Doubly Robust Implementation with R-squared = 0.99; Logistic model, Naive Bayes, and BART in six scenarios
res_DR1, rep_y1 = Restructured_Liu_framework_DR([0.01], [0.05], df, ff; Nreplication = 500, seed = 1234, Logistic = true, NB = true, BART = true, DR_LM = true)
res_DR2, rep_y2 = Restructured_Liu_framework_DR([0.01], [0.3], df, ff; Nreplication = 500, seed = 1234, Logistic = true, NB = true, BART = true, DR_LM = true)
res_DR31, rep_y31 = Restructured_Liu_framework_DR([0.01], [0.5], df, ff; Nreplication = 500, seed = 1234, Logistic = true, NB = true, DR_LM = true)
res_DR32, rep_y32 = Restructured_Liu_framework_DR([0.01], [0.5], df, ff; Nreplication = 500, seed = 1234, BART= true, DR_LM = true)
CSV.write("Liu's_Framework&DoublyRobust\\output\\nor_DoublyRobust1.csv", res_DR1)
CSV.write("Liu's_Framework&DoublyRobust\\output\\nor_DoublyRobust2.csv", res_DR2)
CSV.write("Liu's_Framework&DoublyRobust\\output\\nor_DoublyRobust3_1.csv", res_DR31)
CSV.write("Liu's_Framework&DoublyRobust\\output\\nor_DoublyRobust3_2.csv", res_DR32)

res_DR41, rep_y41 = Restructured_Liu_framework_DR([0.1], [0.05], df, ff; Nreplication = 500, seed = 1234, Logistic = true, NB = true, DR_LM = true)
res_DR42, rep_y42 = Restructured_Liu_framework_DR([0.1], [0.05], df, ff; Nreplication = 500, seed = 1234, BART = true, DR_LM = true)
CSV.write("Liu's_Framework&DoublyRobust\\output\\nor_DoublyRobust4_2.csv", res_DR42)
CSV.write("Liu's_Framework&DoublyRobust\\output\\nor_DoublyRobust4_1.csv", res_DR41)

res_DR51, rep_y51 = Restructured_Liu_framework_DR([0.1], [0.3], df, ff; Nreplication = 500, seed = 1234, Logistic = true, NB = true, DR_LM = true)
res_DR52, rep_y52 = Restructured_Liu_framework_DR([0.1], [0.3], df, ff; Nreplication = 500, seed = 1234, BART = true, DR_LM = true)
CSV.write("Liu's_Framework&DoublyRobust\\output\\nor_DoublyRobust5_1.csv", res_DR51)
CSV.write("Liu's_Framework&DoublyRobust\\output\\nor_DoublyRobust5_2.csv", res_DR52)

res_DR61, rep_y61 = Restructured_Liu_framework_DR([0.1], [0.5], df, ff; Nreplication = 500, seed = 1234, Logistic = true, NB = true, DR_LM = true)
res_DR62, rep_y62 = Restructured_Liu_framework_DR([0.1], [0.5], df, ff; Nreplication = 500, seed = 1234, BART = true, DR_LM = true)
CSV.write("Liu's_Framework&DoublyRobust\\output\\nor_DoublyRobust6_1.csv", res_DR61)
CSV.write("Liu's_Framework&DoublyRobust\\output\\nor_DoublyRobust6_2.csv", res_DR62)




# Prediction model with R-squared = 0.634
ff2 = @formula(y ~ x1 + x5 + x7 + x9+ x11)
lmodel2 = lm(ff2, df)
r2(lmodel2) # R-squared = 0.634

# Doubly Robust Implementation with R-squared = 0.634; Logistic model, Naive Bayes, and BART in six scenarios
lr2_res_DR1, rep_y1 = Restructured_Liu_framework_DR([0.01], [0.05], df, ff2; Nreplication = 500, seed = 1234, Logistic = true, NB = true, BART = true, DR_LM = true)
lr2_res_DR2, rep_y2 = Restructured_Liu_framework_DR([0.01], [0.3], df, ff2; Nreplication = 500, seed = 1234, Logistic = true, NB = true, BART = true, DR_LM = true)
lr2_res_DR31, rep_y31 = Restructured_Liu_framework_DR([0.01], [0.5], df, ff2; Nreplication = 500, seed = 1234, Logistic = true, NB = true, DR_LM = true)
lr2_res_DR32, rep_y32 = Restructured_Liu_framework_DR([0.01], [0.5], df, ff2; Nreplication = 500, seed = 1234, BART= true, DR_LM = true)
CSV.write("Liu's_Framework&DoublyRobust\\output\\lr2_nor_DoublyRobust1.csv", lr2_res_DR1)
CSV.write("Liu's_Framework&DoublyRobust\\output\\lr2_nor_DoublyRobust2.csv", lr2_res_DR2)
CSV.write("Liu's_Framework&DoublyRobust\\output\\lr2_nor_DoublyRobust3_1.csv", lr2_res_DR31)
CSV.write("Liu's_Framework&DoublyRobust\\output\\lr2_nor_DoublyRobust3_2.csv", lr2_res_DR32)

lr2_res_DR41, rep_y41 = Restructured_Liu_framework_DR([0.1], [0.05], df, ff2; Nreplication = 500, seed = 1234, Logistic = true, NB = true, DR_LM = true)
lr2_res_DR42, rep_y42 = Restructured_Liu_framework_DR([0.1], [0.05], df, ff2; Nreplication = 500, seed = 1234, BART = true, DR_LM = true)
CSV.write("Liu's_Framework&DoublyRobust\\output\\lr2_nor_DoublyRobust4_2.csv", lr2_res_DR42)
CSV.write("Liu's_Framework&DoublyRobust\\output\\lr2_nor_DoublyRobust4_1.csv", lr2_res_DR41)

lr2_res_DR51, rep_y51 = Restructured_Liu_framework_DR([0.1], [0.3], df, ff2; Nreplication = 500, seed = 1234, Logistic = true, NB = true, DR_LM = true)
lr2_res_DR52, rep_y52 = Restructured_Liu_framework_DR([0.1], [0.3], df, ff2; Nreplication = 500, seed = 1234, BART = true, DR_LM = true)
CSV.write("Liu's_Framework&DoublyRobust\\output\\lr2_nor_DoublyRobust5_1.csv", lr2_res_DR51)
CSV.write("Liu's_Framework&DoublyRobust\\output\\lr2_nor_DoublyRobust5_2.csv", lr2_res_DR52)

lr2_res_DR61, rep_y61 = Restructured_Liu_framework_DR([0.1], [0.5], df, ff2; Nreplication = 500, seed = 1234, Logistic = true, NB = true, DR_LM = true)
lr2_res_DR62, rep_y62 = Restructured_Liu_framework_DR([0.1], [0.5], df, ff2; Nreplication = 500, seed = 1234, BART = true, DR_LM = true)
CSV.write("Liu's_Framework&DoublyRobust\\output\\lr2_nor_DoublyRobust6_1.csv", lr2_res_DR61)
CSV.write("Liu's_Framework&DoublyRobust\\output\\lr2_nor_DoublyRobust6_2.csv", lr2_res_DR62)
