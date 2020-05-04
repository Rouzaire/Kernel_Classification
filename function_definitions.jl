## Some basic functions
@everywhere dist(x,y) = norm(x-y) # euclidien distance

## Generate x components of the whole dataset, uniformly distributed on a hypersphere of dimension d=D-1.
    # for instance, in d=1, X = unit circle in R^2
    # for instance, in d=2, X = unit sphere in R^3
@everywhere function generate_X(Ptrain,Ptest,dimension,Δ0=0.0)
    @assert isinteger(dimension) ; @assert dimension > 0 ; @assert Δ0 ≥ 0 # Δ0 = margin separating decision boundaries
    N = Ptrain + Ptest
    if Δ0 == 0
        X = rand(MvNormal(zeros(dimension+1),I(dimension+1)),N)
        normX = [norm(X[:,i]) for i in 1:N]
        X_normalized = [(X[:,i] ./ normX[i]) for i in 1:N]
    else
        X = rand(MvNormal(zeros(dimension+1),I(dimension+1)),Int(ceil(N/(1-SpecialFunctions.erf(Δ0/2))))) # generate more data than necessary
        M = size(X)[2]
        normX = [norm(X[:,i]) for i in 1:M]
        X_normalized = [(X[:,i] ./ normX[i]) for i in 1:M]
        X_normalized = X_normalized[[abs(X_normalized[i][1]) ≥ Δ0/2 for i in 1:M]] # Keep only the points out-of-margin and hope that there is at least N of them
        X_normalized = X_normalized[1:N] # keep only the N first datapoint
    end
    return X_normalized
end

@everywhere function generate_Y(X)
    labels = zeros
end
