############################## [Thesis] Correcting Selection Bias by Machine Learning Techniques ##############################

# Import Packages
using CSV, Statistics, DataFrames, DelimitedFiles, RDatasets, ProgressBars, GLM, StatsModels, Fetch, RCall, LIBSVM
using Distributions, Roots# like uniroot in R
using Tables, ThreadSafeDicts, ProgressMeter, Random, XGBoost, DecisionTree, Plots
import NaiveBayes


include("functions/UPrandomsystematic.jl")  #* The inclusion mechanism of Porbabiliy Sampling form liu et al.
include("functions/Bisection.jl")           
include("../Data/DataGeneration.jl")      #* Import the simulated data by me.
include("functions/NB.jl")
include("functions/SRS.jl")
include("functions/BART.jl")


df = Simulate_Data()
#
Restructured_Liu_framework = function (P::Vector, NP::Vector, data::DataFrame, Formula::FormulaTerm; Nreplication = 500, seed = 1234, scenarios = 1:6, Logistic = false, NB = false,
                        XGB = false, SVM = false, SVM_P = false, RF = false, BART = false)
    Random.seed!(seed) 
    # Check P and NP is vector of probability 
    if (length(P) < sum(P) || length(NP) < sum(NP))
        return error("NP or P must be a vector of probabilty, between 1.0 and 0.0")
    elseif (sort(P)[end] > 1 || sort(NP)[end] > 1)
        return error("NP or P must be a vector of probabilty, between 1.0 and 0.0")
    end
    
    # Check which method is available
    method_index = 1
    method_names = ["n_NP", "n_P", "RB_Raw", "RMSE_Raw"]
    if Logistic == true
        push!(method_names, "RB_Log")
        push!(method_names, "RMSE_Log")
        Logistic_index = method_index + 1
        method_index = method_index + 1
    end

    if NB == true
        push!(method_names, "RB_NB")
        push!(method_names, "RMSE_NB")
        NB_index = method_index + 1
        method_index = method_index + 1
    end

    if XGB == true
        push!(method_names, "RB_XGB")
        push!(method_names, "RMSE_XGB")
        XGB_index = method_index + 1
        method_index = method_index + 1
    end

    if SVM == true
        push!(method_names, "RB_SVM")
        push!(method_names, "RMSE_SVM")
        SVM_index = method_index + 1
        method_index = method_index + 1
    end

    if SVM_P == true
        push!(method_names, "RB_SVM_P")
        push!(method_names, "RMSE_SVM_P")
        SVM_P_index = method_index + 1
        method_index = method_index + 1
    end

    if RF == true
        push!(method_names, "RB_RF")
        push!(method_names, "RMSE_RF")
        RF_index = method_index + 1
        method_index = method_index + 1
    end

    if BART == true
        push!(method_names, "RB_BART")
        push!(method_names, "RMSE_BART")
        BART_index = method_index + 1
        method_index = method_index + 1
    end    

    N = nrow(data)
    IV_index = collect(Symbol.(Formula.rhs))
    DV_index = (Symbol.(Formula.lhs))
    num_methods = 1 + Logistic + NB + XGB + SVM + SVM_P + RF + BART

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
        Random.seed!(12345)
        ȳₘ_matrix = Array{Float64}(undef, Nreplication, num_methods)      #! The size of ȳₘ matrix
        P_size = InclusionFraction_P * N

        formula_O = Term(:Z) ~ sum(Formula.rhs[1:end])
        Z_index = Symbol(formula_O.lhs)
        Bsets = ThreadSafeDict()
        Weight_NP_sets = ThreadSafeDict()
        sample_NP_sets = ThreadSafeDict()
        sample_P_sets = ThreadSafeDict()
        ##Calculate the inclusion probability for the dataset
        #Probability Sample
        includeP = repeat([InclusionFraction_P], N)

        #Non-probability sample
        x = data[!, DV_index] ./ 100
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
            #Weight_P  = 1 ./ includeP[S]
            Weight_NP = 1 ./ includeP[S_star]

            # B set
            B = data[S_star .+ S .== 1, :]
            B.Z .= Int.(S_star[S_star .+ S .== 1])

            Bsets[m] = B
            Weight_NP_sets[m] = Weight_NP
            sample_NP_sets[m] = sample_NP
            sample_P_sets[m] = sample_P
            
            #Raw result
            ȳₘ_matrix[m, 1] = mean(sample_NP[!, DV_index]) 
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
        end

        #XGBoosting method
        Threads.@threads for m in 1 : Nreplication
            Random.seed!(m)
            if XGB != true
                break
            end
            B = Bsets[m]
            sample_NP = sample_NP_sets[m] 
            Indp_var = B[:, IV_index]
            Dep_var = B[:, Z_index]

            if InclusionFraction_NP > InclusionFraction_P
                η = 1 + InclusionFraction_P * InclusionFraction_NP
                else η = 1 + InclusionFraction_NP * InclusionFraction_P
            end
            

            XGB_res = xgboost((Indp_var, Dep_var) , num_round = 100, max_depth = 10, objective = "binary:logistic", eta = 0.3, watchlist = (;))
            XGB_psO = XGBoost.predict(XGB_res, DataFrame(sample_NP[:, IV_index]))
            XGB_Oᵢ = XGB_psO ./ (1 .- XGB_psO)
            

            XGB_find = 1 .+ (Weight_NP_sets[m] .- 1) ./ XGB_Oᵢ
            ȳₘ_matrix[m, XGB_index] = sum(sample_NP[!, DV_index] .* XGB_find) / sum(XGB_find)
        end

        # LIBSVM
        Threads.@threads for m in 1 : Nreplication
            Random.seed!(m)
            if SVM != true
                break
            end
            B = Bsets[m]
            sample_NP = Matrix(sample_NP_sets[m])
            Indp_var = Matrix(B[:, IV_index])
            Dep_var = B[:, Z_index]

            SVM_res = svmtrain(Indp_var', Dep_var, probability = true, kernel=LIBSVM.Kernel.Linear)
            SVM_label, SVM_decisionvalue = svmpredict(SVM_res, sample_NP[:, Not(1)]')
            SVM_Oᵢ = SVM_decisionvalue[1, :] ./ SVM_decisionvalue[2, :]

            SVM_find = 1 .+ (Weight_NP_sets[m] .- 1) ./ SVM_Oᵢ
            ȳₘ_matrix[m, SVM_index] = sum(sample_NP[:, 1] .* SVM_find) / sum(SVM_find)
        end

        Threads.@threads for m in 1 : Nreplication
            Random.seed!(m)
            if SVM_P != true
                break
            end
            B = Bsets[m]
            sample_NP = Matrix(sample_NP_sets[m])
            Indp_var = Matrix(B[:, IV_index])
            Dep_var = B[:, Z_index]

            SVM_res = svmtrain(Indp_var', Dep_var, probability = true, kernel=LIBSVM.Kernel.Polynomial)
            SVM_label, SVM_decisionvalue = svmpredict(SVM_res, sample_NP[:, Not(1)]')
            SVM_Oᵢ = SVM_decisionvalue[1, :] ./ SVM_decisionvalue[2, :]

            SVM_P_find = 1 .+ (Weight_NP_sets[m] .- 1) ./ SVM_Oᵢ
            ȳₘ_matrix[m, SVM_P_index] = sum(sample_NP[:, 1] .* SVM_P_find) / sum(SVM_P_find)
        end


        # Random Forest
        Threads.@threads for m in 1 : Nreplication
            Random.seed!(m)
            if RF != true
                break
            end
            B = Bsets[m]
            sample_NP = sample_NP_sets[m]
            Indp_var = Matrix(B[:, IV_index])
            Dep_var = string.(B[:, Z_index])

            model = build_forest(Dep_var, Indp_var, 15, 500) # 15 subfeatures and 500 trees
            RF_res = apply_forest_proba(model, Matrix(sample_NP[:, Not(:y)]), ["1", "0"])
            RF_Oᵢ = RF_res[:, 1] ./ RF_res[:, 2]

            for z in 1 : length(RF_Oᵢ)
                if RF_Oᵢ[z] == Inf
                    RF_Oᵢ[z] = 499/1
                end

                if RF_Oᵢ[z] == 0
                    RF_Oᵢ[z] = 1/499
                end
            end

            RF_find = 1 .+ (Weight_NP_sets[m] .- 1) ./ RF_Oᵢ
            ȳₘ_matrix[m, RF_index] = sum(sample_NP[:, 1] .* RF_find) / sum(RF_find)
            println(m)
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
            #println(m)
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
        end

        if NB == true
            Relative_Bias_Ind_NB = (sum(ȳₘ_matrix[:, NB_index] .- mean(data[:, DV_index])) / mean(data[:, DV_index])) / size(ȳₘ_matrix)[1] * 100
            RMSE_Ind_NB = (sum((ȳₘ_matrix[:, NB_index] .- mean(data[:, DV_index])) .^ 2) / size(ȳₘ_matrix)[1]) ^ (1/2)
            Classification_result[i, 2*NB_index + 1] = Relative_Bias_Ind_NB
            Classification_result[i, 2*NB_index + 2] = RMSE_Ind_NB
        end
        

        if XGB == true
            Relative_Bias_Ind_XGB = (sum(ȳₘ_matrix[:, XGB_index] .- mean(data[:, DV_index])) / mean(data[:, DV_index])) / size(ȳₘ_matrix)[1] * 100
            RMSE_Ind_XGB = (sum((ȳₘ_matrix[:, XGB_index] .- mean(data[:, DV_index])) .^ 2) / size(ȳₘ_matrix)[1]) ^ (1/2)
            Classification_result[i, 2*XGB_index + 1] = Relative_Bias_Ind_XGB
            Classification_result[i, 2*XGB_index + 2] = RMSE_Ind_XGB
        end
        
        if SVM == true
            Relative_Bias_Ind_SVM = (sum(ȳₘ_matrix[:, SVM_index] .- mean(data[:, DV_index])) / mean(data[:, DV_index])) / size(ȳₘ_matrix)[1] * 100
            RMSE_Ind_SVM = (sum((ȳₘ_matrix[:, SVM_index] .- mean(data[:, DV_index])) .^ 2) / size(ȳₘ_matrix)[1]) ^ (1/2)
            Classification_result[i, 2*SVM_index + 1] = Relative_Bias_Ind_SVM
            Classification_result[i, 2*SVM_index + 2] = RMSE_Ind_SVM
        end

        if SVM_P == true
            Relative_Bias_Ind_SVM_P = (sum(ȳₘ_matrix[:, SVM_P_index] .- mean(data[:, DV_index])) / mean(data[:, DV_index])) / size(ȳₘ_matrix)[1] * 100
            RMSE_Ind_SVM_P = (sum((ȳₘ_matrix[:, SVM_P_index] .- mean(data[:, DV_index])) .^ 2) / size(ȳₘ_matrix)[1]) ^ (1/2)
            Classification_result[i, 2*SVM_P_index + 1] = Relative_Bias_Ind_SVM_P 
            Classification_result[i, 2*SVM_P_index + 2] = RMSE_Ind_SVM_P
        end

        if RF == true
            Relative_Bias_Ind_RF = (sum(ȳₘ_matrix[:, RF_index] .- mean(data[:, DV_index])) / mean(data[:, DV_index])) / size(ȳₘ_matrix)[1] * 100
            RMSE_Ind_RF = (sum((ȳₘ_matrix[:, RF_index] .- mean(data[:, DV_index])) .^ 2) / size(ȳₘ_matrix)[1]) ^ (1/2)
            Classification_result[i, 2*RF_index + 1] = Relative_Bias_Ind_RF
            Classification_result[i, 2*RF_index + 2] = RMSE_Ind_RF
        end

        if BART == true
            Relative_Bias_Ind_BART = (sum(ȳₘ_matrix[:, BART_index] .- mean(data[:, DV_index])) / mean(data[:, DV_index])) / size(ȳₘ_matrix)[1] * 100
            RMSE_Ind_BART = (sum((ȳₘ_matrix[:, BART_index] .- mean(data[:, DV_index])) .^ 2) / size(ȳₘ_matrix)[1]) ^ (1/2)
            Classification_result[i, 2*BART_index + 1] = Relative_Bias_Ind_BART
            Classification_result[i, 2*BART_index + 2] = RMSE_Ind_BART
        end

        iter_y_matrix[i] = ȳₘ_matrix
    end
        Classification_result = DataFrame(Classification_result, :auto)
        result_names = Symbol.(method_names)
        rename!(Classification_result, result_names)  
        

        return Classification_result, iter_y_matrix
end

ff = @formula(y ~ x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10 + x11 + x12 + x13 + x14 + x15)

# Results of Logistic model, Naive Bayes in f_P = [0.01, 0.1], f_NP = [0.05, 0.3]
res1, rep_y = Restructured_Liu_framework([0.01, 0.1], [0.05], df, ff; Nreplication = 500, seed = 1234, Logistic = true, NB = true)
CSV.write("Liu's_Framework\\output\\Log_NB_results1.csv", res1)
res2, rep_y = Restructured_Liu_framework([0.01, 0.1], [0.3], df, ff; Nreplication = 500, seed = 1234, Logistic = true, NB = true)
CSV.write("Liu's_Framework\\output\\Log_NB_results2.csv", res2)
# Results of Logistic model in f_P = [0.01, 0.1], f_NP = [0.5]
res3_1, rep_y = Restructured_Liu_framework([0.01], [0.5], df, ff; Nreplication = 500, seed = 1234, Logistic = true)
CSV.write("Liu's_Framework\\output\\Log_results3_1.csv", res3_1)
res3_2, rep_y = Restructured_Liu_framework([0.1], [0.5], df, ff; Nreplication = 500, seed = 1234, Logistic = true)
CSV.write("Liu's_Framework\\output\\Log_results3_2.csv", res3_2)
# Results of Naive Bayes in f_P = [0.01, 0.1], f_NP = [0.5]
res4_1, rep_y = Restructured_Liu_framework([0.01], [0.5], df, ff; Nreplication = 500, seed = 1234, NB = true)
CSV.write("Liu's_Framework\\output\\NB_results4_1.csv", res4_1)
res4_2, rep_y = Restructured_Liu_framework([0.1], [0.5], df, ff; Nreplication = 500, seed = 1234, NB = true)
CSV.write("Liu's_Framework\\output\\NB_results4_2.csv", res4_2)

# Results of XGB in six scenarios
res_XGB1, rep_y = Restructured_Liu_framework([0.01], [0.05], df, ff; Nreplication = 500, seed = 1234, XGB = true)
CSV.write("Liu's_Framework\\output\\true_XGB_results1.csv", res_XGB1)
res_XGB2, rep_y = Restructured_Liu_framework([0.01], [0.3], df, ff; Nreplication = 500, seed = 1234, XGB = true)
CSV.write("Liu's_Framework\\output\\true_XGB_results2.csv", res_XGB2)
res_XGB3, rep_y = Restructured_Liu_framework([0.01], [0.5], df, ff; Nreplication = 500, seed = 1234, XGB = true)
CSV.write("Liu's_Framework\\output\\true_XGB_results3.csv", res_XGB3)
res_XGB4, rep_y = Restructured_Liu_framework([0.1], [0.05], df, ff; Nreplication = 500, seed = 1234, XGB = true)
CSV.write("Liu's_Framework\\output\\true_XGB_results4.csv", res_XGB4)
res_XGB5, rep_y = Restructured_Liu_framework([0.1], [0.3], df, ff; Nreplication = 500, seed = 1234, XGB = true)
CSV.write("Liu's_Framework\\output\\true_XGB_results5.csv", res_XGB5)
res_XGB6, rep_y = Restructured_Liu_framework([0.1], [0.5], df, ff; Nreplication = 500, seed = 1234, XGB = true)
CSV.write("Liu's_Framework\\output\\true_XGB_results6.csv", res_XGB6)

# Results of BART in six scenarios
res_BART1, rep_y = Restructured_Liu_framework([0.01], [0.05], df, ff; Nreplication = 500, seed = 1234, BART = true)
CSV.write("Liu's_Framework\\output\\res_BART1.csv", res_BART1)
res_BART2, rep_y = Restructured_Liu_framework([0.01], [0.3], df, ff; Nreplication = 500, seed = 1234, BART = true)
CSV.write("Liu's_Framework\\output\\res_BART2.csv", res_BART2)
res_BART3, rep_y = Restructured_Liu_framework([0.01], [0.5], df, ff; Nreplication = 500, seed = 1234, BART = true)
CSV.write("Liu's_Framework\\output\\res_BART3.csv", res_BART3)
res_BART4, rep_y = Restructured_Liu_framework([0.1], [0.05], df, ff; Nreplication = 500, seed = 1234, BART = true)
CSV.write("Liu's_Framework\\output\\res_BART4.csv", res_BART4)
res_BART5, rep_y = Restructured_Liu_framework([0.1], [0.3], df, ff; Nreplication = 500, seed = 1234, BART = true)
CSV.write("Liu's_Framework\\output\\res_BART5.csv", res_BART5)
res_BART6, rep_y = Restructured_Liu_framework([0.1], [0.5], df, ff; Nreplication = 500, seed = 1234, BART = true)
CSV.write("Liu's_Framework\\output\\res_BART6.csv", res_BART6)

# Results of Random Forest in six scenarios
res_RF1, rep_y = Restructured_Liu_framework([0.01], [0.05], df, ff; Nreplication = 500, seed = 1234, RF = true)
CSV.write("Liu's_Framework\\output\\res_RF1.csv", res_RF1)

res_RF2, rep_y = Restructured_Liu_framework([0.01], [0.3], df, ff; Nreplication = 500, seed = 1234, NB = false, Logistic = false, XGB = false, SVM = false, SVM_P = false, RF = true)
CSV.write("Liu's_Framework\\output\\res_RF2.csv", res_RF2)

res_RF3, rep_y = Restructured_Liu_framework([0.01], [0.5], df, ff; Nreplication = 500, seed = 1234, NB = false, Logistic = false, XGB = false, SVM = false, SVM_P = false, RF = true)
CSV.write("Liu's_Framework\\output\\res_RF3.csv", res_RF3)

res_RF4, rep_y = Restructured_Liu_framework([0.1], [0.05], df, ff; Nreplication = 500, seed = 1234, NB = false, Logistic = false, XGB = false, SVM = false, SVM_P = false, RF = true)
CSV.write("Liu's_Framework\\output\\res_RF4.csv", res_RF4)

res_RF5, rep_y = Restructured_Liu_framework([0.1], [0.3], df, ff; Nreplication = 500, seed = 1234, NB = false, Logistic = false, XGB = false, SVM = false, SVM_P = false, RF = true)
CSV.write("Liu's_Framework\\output\\res_RF5.csv", res_RF5)

res_RF6, rep_y = Restructured_Liu_framework([0.1], [0.5], df, ff; Nreplication = 500, seed = 1234, NB = false, Logistic = false, XGB = false, SVM = false, SVM_P = false, RF = true)
CSV.write("Liu's_Framework\\output\\res_RF6.csv", res_RF6)

# Results of SVM_linear in six scenarios
res3, rep_y2 = Restructured_Liu_framework([0.01], [0.05], df, ff; Nreplication = 500, seed = 1234, NB = false, Logistic = false, XGB = false, SVM = true)
CSV.write("Liu's_Framework\\output\\res_SVM_l1.csv", res3)
res4, rep_y2 = Restructured_Liu_framework([0.01], [0.3], df, ff; Nreplication = 500, seed = 1234, NB = false, Logistic = false, XGB = false, SVM = true)
CSV.write("Liu's_Framework\\output\\res_SVM_l2.csv", res4)
res5, rep_y2 = Restructured_Liu_framework([0.01], [0.5], df, ff; Nreplication = 500, seed = 1234, NB = false, Logistic = false, XGB = false, SVM = true)
CSV.write("Liu's_Framework\\output\\res_SVM_l3.csv", res5)
res6, rep_y2 = Restructured_Liu_framework([0.1], [0.05], df, ff; Nreplication = 500, seed = 1234, NB = false, Logistic = false, XGB = false, SVM = true)
CSV.write("Liu's_Framework\\output\\res_SVM_l4.csv", res6)
res7, rep_y2 = Restructured_Liu_framework([0.1], [0.3], df, ff; Nreplication = 500, seed = 1234, NB = false, Logistic = false, XGB = false, SVM = true)
CSV.write("Liu's_Framework\\output\\res_SVM_l5.csv", res7)
res8, rep_y2 = Restructured_Liu_framework([0.1], [0.5], df, ff; Nreplication = 500, seed = 1234, NB = false, Logistic = false, XGB = false, SVM = true)
CSV.write("Liu's_Framework\\output\\res_SVM_l6.csv", res8)

# Results of SVM_polynomial in six scenarios
res9, rep_y = Restructured_Liu_framework([0.01], [0.05], df, ff; Nreplication = 500, seed = 1234, NB = false, Logistic = false, XGB = false, SVM_P = true)
CSV.write("Liu's_Framework\\output\\res_SVM_P1.csv", res9)
res10, rep_y = Restructured_Liu_framework([0.01], [0.3], df, ff; Nreplication = 500, seed = 1234, NB = false, Logistic = false, XGB = false, SVM_P = true)
CSV.write("Liu's_Framework\\output\\res_SVM_P2.csv", res10)
res11, rep_y = Restructured_Liu_framework([0.01], [0.5], df, ff; Nreplication = 500, seed = 1234, NB = false, Logistic = false, XGB = false, SVM_P = true)
CSV.write("Liu's_Framework\\output\\res_SVM_P3.csv", res11)
res12, rep_y = Restructured_Liu_framework([0.1], [0.05], df, ff; Nreplication = 500, seed = 1234, NB = false, Logistic = false, XGB = false, SVM_P = true)
CSV.write("Liu's_Framework\\output\\res_SVM_P4.csv", res12)
res13, rep_y = Restructured_Liu_framework([0.1], [0.3], df, ff; Nreplication = 500, seed = 1234, NB = false, Logistic = false, XGB = false, SVM_P = true)
CSV.write("Liu's_Framework\\output\\res_SVM_P5.csv", res13)
res14, rep_y = Restructured_Liu_framework([0.1], [0.5], df, ff; Nreplication = 500, seed = 1234, NB = false, Logistic = false, XGB = false, SVM_P = true)
CSV.write("Liu's_Framework\\output\\res_SVM_P6.csv", res14)


