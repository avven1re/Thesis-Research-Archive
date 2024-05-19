# A function of Random systematic sampling from the package "sampling" in R

#function (pik, eps = 1e-06) 
#    {
#        if (any(is.na(pik))) 
#            stop("there are missing values in the pik vector")
#        N = length(pik)
#        v = sample.int(N, N)
#        s = numeric(N)
#        s[v] = UPsystematic(pik[v], eps)
#        s
#    }
#    <bytecode: 0x00000227a788afb8>
#    <environment: namespace:sampling>
############################################################################################################################
import StatsBase
include("UPsystematic.jl")
#any(ismissing.([1 2 missing 4]))

function UPrandomsystematic(πₖ, ϵ = 1.0e-6)
    if any(ismissing.(πₖ))
        error("there are missing values in the πₖ vector")
        return
    end
    N = length(πₖ)
    v = StatsBase.sample(1:N, N, replace = false)
    s = zeros(N)
    s[v] = UPsystematic(πₖ[v], ϵ)
    s
end
##

