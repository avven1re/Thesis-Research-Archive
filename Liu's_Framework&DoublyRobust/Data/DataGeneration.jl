using CSV, Statistics, DataFrames, DelimitedFiles, RDatasets, ProgressBars, Random, StatsBase
using Distributions

# size of the data
#N = 100000

# Column names
colnames = ["y", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", 
            "x10", "x11", "x12", "x13", "x14", "x15"]

# Set seed

Simulate_Data = function (N = 100000; seed = 123)

    Random.seed!(seed)    
    x1 = rand(Normal(50, 5), N)             # μ = 50 ; σ = 5
    x2 = rand(Beta(2, 5), N)                # α = 2 ; Β = 5
    x3 = rand(Exponential(2), N)            # Θ = 2
    x4 = rand(Uniform(), N)                 # a = 0, b = 1
    ϵ1 = rand(Normal(), N)
    x5 = 3.2 .* x1 .- 2 .* x2 .+ 5 .* x3 .- 10 .* x4 .+ ϵ1

    x6 = rand(Poisson(5), N)                   # λ = 5
    x7 = rand(Binomial(10, 0.55), N)           # n = 10 ; p = 0.55
    x8 = rand(NegativeBinomial(8, 0.4), N)        # r = 8 ; p = 0.4

    year = 1990 : 2023
    x9 = sample(year, N, replace = true)

    x10 = rand(Poisson(15), N)  

    ϵ2 = rand(Normal(), N)
    ϵ3 = rand(Normal(), N)
    ϵ4 = rand(Normal(), N)
    ϵ5 = rand(Normal(), N)
    ϵ6 = rand(Normal(), N)
    ϵ7 = rand(Normal(), N)

    x11 = 3 .* x1 .- 4.5 .* x4 .+ 6.7 .* x7 .+ ϵ2
    x12 = 7 .* x2 .+ 2.5 .* x3 .+ 2 .* x6 .+ ϵ3
    x13 = 0.3 .* x9 .- 0.7 .* x5 .+ ϵ4
    x14 = 0.2 .* x9 .- 2 .* x6 .+ 1.3 .* x3 .+ ϵ5
    x15 = 1.5 .* x10 .+ 3 .* x2 .- x8 .+ ϵ6

    y = 2.3 .* x1 .+ 3.3 .* x11 .- 0.7 .* x13 .+ x9 .* x4 .- 4.7 .* x15 .+ ϵ7
    #y = x1 .* x10 .+ x4 .* x8 .+ 79 .* x13 .* x13 .+ ϵ7 + sin.(x14) + cos.(x9)
    
    df = DataFrame(y = y, x1 = x1, x2 = x2, x3 = x3, x4 = x4, x5 = x5,
                    x6 = x6, x7 = x7, x8 = x8, x9 = x9, x10 = x10, x11 = x11,
                    x12 = x12, x13 = x13, x14 = x14, x15 = x15)
    return df

end

df = Simulate_Data

df()