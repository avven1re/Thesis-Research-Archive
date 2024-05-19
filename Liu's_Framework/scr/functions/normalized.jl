normalized = function (vec::Vector)
    res = vec ./ sum(vec)
    return res    
end