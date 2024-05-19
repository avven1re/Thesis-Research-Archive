function bisection(f, a, b, tol)
    
    if sign(f(a)) == sign(f(b))
        error("function has the same sign at given endpoints")
    end

    mid = (a + b)/2

    while abs(f(mid)) > tol

        sign(f(mid)) == sign(f(a)) ? a=mid : b=mid
            
        mid = (a + b)/2

    end

    return mid
    
end