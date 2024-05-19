# Uses the systematic method to select a sample of units (unequal probabilities, without replacement, fixed sample size).
#function (pik, eps = 1e-06) 
#    {
#        if (any(is.na(pik))) 
#            stop("there are missing values in the pik vector")
#        list = pik > eps & pik < 1 - eps
#        pik1 = pik[list]
#        N = length(pik1)
#        a = (c(0, cumsum(pik1)) - runif(1, 0, 1))%%1
#        s1 = as.integer(a[1:N] > a[2:(N + 1)])
#        s = pik
#        s[list] = s1
#        s
#    }
#    <bytecode: 0x00000227a56c28b0>
#    <environment: namespace:sampling>
################################################################################################
using Distributions

function UPsystematic(πₖ, ϵ = 1.0e-6)
    if any(ismissing.(πₖ))
        error("there are missing values in the πₖ vector")
        return
    end

    list = πₖ .> ϵ .&& πₖ .< 1 - ϵ
    πₖ₁ = πₖ[list]
    N = length(πₖ₁)
    a = mod.(([0; cumsum(πₖ₁)] .- rand(Uniform(), 1)), 1)
    s1 = Int.(a[1:N] .> a[2:(N+1)])
    s = πₖ
    s[list] = s1
    s
end

