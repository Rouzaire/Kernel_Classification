## Some basic functions
@everywhere dist(x,y) = norm(x-y) # euclidien distance
@everywhere testerr(pred,Y) = mean(pred .!= Y)

## Generate x components of the whole dataset, uniformly distributed on a hypersphere of dimension d=D-1.
    # for instance, in d=1, X = unit circle in R^2
    # for instance, in d=2, X = unit sphere in R^3
@everywhere function generate_X(Ptrain,Ptest,dimension,Δ0=0.0)
    @assert isinteger(dimension) ; @assert dimension > 0 ; @assert Δ0 ≥ 0 # Δ0 = margin separating decision boundaries
    N = Int(Ptrain + Ptest)
    if Δ0 == 0
        X = rand(MvNormal(zeros(dimension+1),I(dimension+1)),N)
        normX = [norm(X[:,i]) for i in 1:N]
        X_normalized = [(X[:,i] ./ normX[i]) for i in 1:N]
    else
        Ntilde = Int(ceil(2N/(1-SpecialFunctions.erf(Δ0/2)))) # generate more data than necessary
        X = rand(MvNormal(zeros(dimension+1),I(dimension+1)),Ntilde)
        M = size(X)[2]
        normX = [norm(X[:,i]) for i in 1:M]
        X_normalized = [(X[:,i] ./ normX[i]) for i in 1:M]
        X_normalized = X_normalized[[abs(X_normalized[i][1]) ≥ Δ0/2 for i in 1:M]] # Keep only the points out-of-margin and hope that there is at least N of them
        Nkept = length(X_normalized)
        X_normalized = X_normalized[1:N] # keep only the N first datapoint
    end
    # println("N = $N ,  Actually generated = $Ntilde , Thrown away = $(Ntilde-Nkept)")
    return X_normalized
end

@everywhere function generate_Y(X,Δ0=0.0)
    labels = zeros(length(X))
    for i in eachindex(X)
        if X[i][1] ≥ Δ0/2 labels[i] = + 1
        else              labels[i] = - 1
        end
    end
    return labels
end

# The way these functions are defined ensure the non-overlapping of the testing and training sets
@everywhere function extract_TrainSet(X,Y,Ptrain)
    dimension = length(X[1])
    Xtrain = zeros(Ptrain,dimension) ; Ytrain = zeros(Ptrain)
    for i in 1:Ptrain
        Xtrain[i,:] = X[i]
        Ytrain[i] = Y[i]
    end
    return Xtrain,Ytrain
    # return Xtrain,categorical(Ytrain)
end

@everywhere function extract_TestSet(X,Y,Ptest)
    dimension = length(X[1])
    Xtest = zeros(Ptest,dimension) ; Ytest = zeros(Ptest)
    for i in 1:Ptest
        Xtest[i,:] = X[end-i+1]
        Ytest[i] = Y[end-i+1]
    end
    return Xtest,Ytest
    # return Xtest,categorical(Ytest)
end

@everywhere function rc(sv) ## returns the mean minimum distance separating support vectors (SV)
    svy = Bool.((generate_Y(sv) .+ 1)/2)
    sv_plus  = sv[svy]
    sv_minus = sv[.!svy]
    rc_plus  = compute_rc(sv_plus)
    rc_minus = compute_rc(sv_minus)
    rc = vcat(rc_plus,rc_minus)
    return mean(rc),std(rc)
end

@everywhere function compute_rc(sv) # auxillary function that returns a list of the minimal distance to all other SV for each SV
    rc  = zeros(length(sv))
    for i in eachindex(sv)
        dist_to_all_other_SV = [norm(sv[i]-sv[j]) for j in eachindex(sv)]
        if length(dist_to_all_other_SV) > 1
            rc[i] = sort(dist_to_all_other_SV)[2]
        else
            rc[i] = -1 ## impossible value, meaning that there is only one (or zero) SV, hence the distance to nearest SV is not defined
        end
    end
    return rc
end
## Matérn Covariance functions
@everywhere function Matérn(h,ν,σ=1,ρ=100) # Matérn kernel
    ## h is the euclidian distance in real space R² (actual subspace = unit circle)
    ## ν (nu) the smoothness of the Matérn covariance function (here, usually ν = 0.5 to get a Laplace Kernel)
    ## ρ is the length scale. Here I impose a very large length scale
    ## σ² is the amplitude/sill of the function
    if     ν == 1/2 return σ^2*exp(-h/ρ)
    elseif ν == 3/2 return σ^2*(1+sqrt(3)*h/ρ)*exp(-sqrt(3)*h/ρ)
    elseif ν == 5/2 return σ^2*(1+sqrt(5)*h/ρ + 5/3*h^2/ρ^2)*exp(-sqrt(5)*h/ρ)
    else            return σ^2 * 2.0^(1-ν) / SpecialFunctions.gamma(ν) * (sqrt(2ν)h/ρ)^ν * SpecialFunctions.besselk(ν,Float64(sqrt(2ν)h/ρ))
    end
end

@everywhere function Laplace_Kernel(X,Y)
    ρ = 100 ; σ = 1
    Px = size(X)[1] ; Py = size(Y)[1]
    K = ones(Px,Py)
    for i in 1:Px
        for j in 1:Py
            K[i,j] = σ^2*exp(-norm(X[i,:]-Y[j,:])/ρ)
        end
    end
    return K' ## the adjoint is necessary to match the desired format of sklearn
end

@everywhere function Run_fixed_dimension(PP,Δ,d,M=1) ## d is a integer passed in argument and the scan is over M and the vectors PP, Δ0
    ## 3D Matrix to store data // dim 1 : PP //  dim 2 : margin // dim 3 : Realisations
    misclassification_error_matrix  = NaN*zeros(length(PP),length(Δ),M)
    alpha_mean_matrix  = NaN*zeros(length(PP),length(Δ),M)
    alpha_std_matrix   = NaN*zeros(length(PP),length(Δ),M)
    rc_mean_matrix     = NaN*zeros(length(PP),length(Δ),M)
    rc_std_matrix      = NaN*zeros(length(PP),length(Δ),M)

    for i in eachindex(PP)
        Ptrain = PP[i]

        for j in eachindex(Δ)
            Δ0 = Δ[j]
            low = 1E3 ; high = 1E6 ; pow = 1 + 4Δ0
            Ptest = Int(round((min(high,max(10*Ptrain^pow,low))))) # enforce low ≤ Ptest ≤ high
            N = Ptrain + Ptest

            println("SVM for P = $Ptrain , Ptest = 1E$(Int(round(log10(Ptest)))) , Δ = $Δ0 , Time : "*string(Dates.hour(now()))*"h"*string(Dates.minute(now()))*" [d = $d]")
            for m in 1:M

                X             = generate_X(Ptrain,Ptest,d,Δ0)
                Y             = generate_Y(X,Δ0)
                Xtest,Ytest   = extract_TestSet(X,Y,Ptest)
                Xtrain,Ytrain = extract_TrainSet(X,Y,Ptrain)

                clf = SV.SVC(C=1E10,kernel="precomputed",cache_size=800) # 800 MB allocated cache
                GramTrain = Laplace_Kernel(Xtrain,Xtrain)
                clf.fit(GramTrain, Ytrain)
                GramTest = Laplace_Kernel(Xtrain,Xtest)

                # Test Error
                    misclassification_error_matrix[i,j,m] = testerr(clf.predict(GramTest),Ytest)
                # α
                    alpha_mean_matrix[i,j,m] = mean(abs.(clf.dual_coef_))
                    alpha_std_matrix[i,j,m]  = std(abs.(clf.dual_coef_))
                # r_c
                    sv = X[clf.support_ .+ 1]
                    rc_mean_matrix[i,j,m],rc_std_matrix[i,j,m] = rc(sv)
            end # Realisations
        end # Δ0
    end # Ptrain

    ## Save Data for later analysis
    str = "D_"*string(d)*"_"*string(Dates.day(now()))
    JLD.save("Data\\"*str*".jld", "error", misclassification_error_matrix,"alpha_mean_matrix",alpha_mean_matrix,"alpha_std_matrix",alpha_std_matrix,"rc_mean_matrix",rc_mean_matrix,"rc_std_matrix",rc_std_matrix, "PP", PP, "Δ", Δ, "d", d, "M", M)
end

@everywhere function Run_fixed_delta(PP,Δ0,dim,M=1) ## Δ0 is a scalar passed in argument and the scan is over M and the vectors PP, dim
    ## 3D Matrix to store data // dim 1 : PP //  dim 2 : dimensions // dim 3 : Realisations
    misclassification_error_matrix  = NaN*zeros(length(PP),length(dim),M)
    alpha_mean_matrix  = NaN*zeros(length(PP),length(dim),M)
    alpha_std_matrix   = NaN*zeros(length(PP),length(dim),M)
    rc_mean_matrix     = NaN*zeros(length(PP),length(dim),M)
    rc_std_matrix      = NaN*zeros(length(PP),length(dim),M)

    for i in eachindex(PP)
        Ptrain = PP[i]

        for j in eachindex(dim)
            d = dim[j]
            low = 1E3 ; high = 1E6 ; pow = 1 + 4Δ0
            Ptest = Int(round((min(high,max(10*Ptrain^pow,low))))) # enforce low ≤ Ptest ≤ high
            N = Ptrain + Ptest



            println("SVM for P = $Ptrain , Ptest = 1E$(Int(round(log10(Ptest)))) , Δ = $Δ0 , Time : "*string(Dates.hour(now()))*"h"*string(Dates.minute(now()))*" [d = $d]")
            for m in 1:M

                X             = generate_X(Ptrain,Ptest,d,Δ0)
                Y             = generate_Y(X,Δ0)
                Xtest,Ytest   = extract_TestSet(X,Y,Ptest)
                Xtrain,Ytrain = extract_TrainSet(X,Y,Ptrain)

                clf = SV.SVC(C=1E10,kernel="precomputed",cache_size=800) # 800 MB allocated cache
                GramTrain = Laplace_Kernel(Xtrain,Xtrain)
                clf.fit(GramTrain, Ytrain)
                GramTest = Laplace_Kernel(Xtrain,Xtest)

                # Test Error
                    misclassification_error_matrix[i,j,m] = testerr(clf.predict(GramTest),Ytest)
                # α
                    alpha_mean_matrix[i,j,m] = mean(abs.(clf.dual_coef_))
                    alpha_std_matrix[i,j,m]  = std(abs.(clf.dual_coef_))
                # r_c
                    sv = X[clf.support_ .+ 1]
                    rc_mean_matrix[i,j,m],rc_std_matrix[i,j,m] = rc(sv)
            end # Realisations
        end # Δ0
    end # Ptrain

    ## Save Data for later analysis
    str = "Δ_"*string(Δ0)*"_"*string(Dates.day(now()))
    JLD.save("Data\\"*str*".jld", "error", misclassification_error_matrix,"alpha_mean_matrix",alpha_mean_matrix,"alpha_std_matrix",alpha_std_matrix,"rc_mean_matrix",rc_mean_matrix,"rc_std_matrix",rc_std_matrix, "PP", PP, "Δ", Δ0, "d", dim, "M", M)
end

@everywhere function Run(parallelized_over,args...)
    @assert parallelized_over in ["d","Δ"] println("Choose the function name among \"PP\",\"d\",\"Δ\".")
    if     parallelized_over == "d" pmap(Run_fixed_dimension,args...)
    elseif parallelized_over == "Δ" pmap(Run_fixed_delta,args...)
    end
end
