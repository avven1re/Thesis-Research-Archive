using Random, StatsBase

SRS = function (n, pop_size)
    size = Int(n)
    inclusion_indicator = falses(pop_size)
    included_indices = sample(1:pop_size, size, replace = false)
    inclusion_indicator[included_indices] .= true
    return inclusion_indicator
end

