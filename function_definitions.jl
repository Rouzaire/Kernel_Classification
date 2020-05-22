## Some basic functions
@everywhere dist(x,y) = norm(x-y) # euclidien distance
@everywhere testerr(pred,Y) = mean(pred .!= Y)

## Generation of data, uniformly on the d-dimensional unit-hypersphere (d=1 -> X lies on unit circle in R^2 // d=2 -> X lies on unit sphere in R^3)
    # Notation : Δ0 is the gap size bewteen 2 interfaces and SVband is the SV bandwidth

@everywhere function generate_Y(X,Δ0)
    labels = zeros(length(X))
    for i in eachindex(X)
        if X[i][1] ≥ Δ0/2 labels[i] = + 1
        else              labels[i] = - 1
        end
    end
    return labels
end

@everywhere function generate_TrainSet(Ptrain,d,Δ0)
    @assert isinteger(d) ; @assert d > 0 ; @assert Δ0 ≥ 0 # Δ0 = margin separating decision boundaries

    M = Int(ceil(2*Ptrain/(1-SpecialFunctions.erf(Δ0/2)))) # generate more data than necessary
    X = rand(MvNormal(zeros(d+1),I(d+1)),M)
    normX = [norm(X[:,i]) for i in 1:M]
    X_normalized = [(X[:,i] ./ normX[i]) for i in 1:M]
    X_normalized = X_normalized[[Δ0/2 ≤ abs(X_normalized[i][1]) for i in 1:M]] # Keep only the points out-of-margin and hope that there is at least N of them
    @assert length(X_normalized) ≥ Ptrain
    X_normalized = X_normalized[1:Ptrain]

    return X_normalized , generate_Y(X_normalized,Δ0)
end

@everywhere function generate_TestSet(Ptest,d,Δ0;ξ=1)
    # 0<ξ<2 is the exponent governing the cusp (ξ=1 for Laplace kernel, ξ=2 for RBF/Gaussian kernel)
    # Note that ξ = min(2ν,2) for a kernel in the Matérn family
    @assert isinteger(d) ; @assert d > 0 ; @assert Δ0 ≥ 0 # Δ0 = margin separating decision boundaries

    # Points will only be generated in this band, because beyond one can be sure that
        # they will be correctly classified. It has to be wide enough to be sure that
        # all misclassified points are contained and thin enough to save memory later
        # in the code
    exponent = (d-1+ξ)/(3d-3+ξ)
    SVband = Ptest^(-exponent) ## upperbound, according to the paper "How isotropic kernels learn simple invariants" de Jonas and my own benchmarks
    SVband = 0.2
    weight_band = 2SVband/(π - Δ0) # fraction of the surface occupied by this band (to weight the final result)

    M = Int(ceil(2*Ptest/weight_band/(1-SpecialFunctions.erf(Δ0/2)))) # generate more data than necessary

    X = rand(MvNormal(zeros(d+1),I(d+1)),M)
    normX = [norm(X[:,i]) for i in 1:M]
    X_normalized = [(X[:,i] ./ normX[i]) for i in 1:M]
    X_normalized = X_normalized[[Δ0/2 ≤ abs(X_normalized[i][1]) ≤ Δ0/2 + SVband for i in 1:M]] # Keep only the points out-of-margin and hope that there is at least N of them
    Nkept = length(X_normalized)
    @assert Nkept ≥ Ptest
    X_normalized = X_normalized[1:Ptest]

    # println("N = $Ptest ,  Actually generated = $M , Wasted = $((Nkept-Ptest))")
    return X_normalized , generate_Y(X_normalized,Δ0) , weight_band
end

@everywhere function rc(sv,Δ0) ## returns the mean minimum distance separating support vectors (SV)
    svy = Bool.((generate_Y(sv,Δ0) .+ 1)/2)
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

@everywhere function Run_fixed_dimension(PP,Δ,d,M=1) ## d is a integer passed in argument and the scan is over M and the vectors PP, Δ (gaps between interfaces)
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
            # Ptest = Int(round((min(high,max(5*Ptrain^pow,low))))) # enforce low ≤ Ptest ≤ high
            if Δ0 == 0 Ptest = 1000 else Ptest = 1000 end

            println("SVM for P = $Ptrain , Ptest = 1E$(round(log10(Ptest),digits=1)) , Δ = $Δ0 , Time : "*string(Dates.hour(now()))*"h"*string(Dates.minute(now()))*" [d = $d]")
            for m in 1:M

                Xtrain,Ytrain = generate_TrainSet(Ptrain,d,Δ0)
                Xtest,Ytest,weight_band = generate_TestSet(Ptest,d,Δ0,ξ=1)

                clf = SV.SVC(C=1E10,kernel="precomputed",cache_size=1000) # 800 MB allocated cache
                GramTrain = Laplace_Kernel(Xtrain,Xtrain)
                clf.fit(GramTrain, Ytrain)
                GramTest = Laplace_Kernel(Xtrain,Xtest)

                # Test Error
                    misclassification_error_matrix[i,j,m] = testerr(clf.predict(GramTest),Ytest)*weight_band
                # α
                    alpha_mean_matrix[i,j,m] = mean(abs.(clf.dual_coef_))
                    alpha_std_matrix[i,j,m]  = std(abs.(clf.dual_coef_))
                # r_c
                    sv = Xtrain[clf.support_ .+ 1]
                    rc_mean_matrix[i,j,m],rc_std_matrix[i,j,m] = rc(sv,Δ0)
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
            # Ptest = Int(round((min(high,max(10*Ptrain^pow,low))))) # enforce low ≤ Ptest ≤ high
            Ptest = 1000

            println("SVM for P = $Ptrain , Ptest = 1E$(Int(round(log10(Ptest)))) , Δ = $Δ0 , Time : "*string(Dates.hour(now()))*"h"*string(Dates.minute(now()))*" [d = $d]")

            for m in 1:M

                Xtrain,Ytrain = generate_TrainSet(Ptrain,d,Δ0)
                Xtest,Ytest,weight_band = generate_TestSet(Ptest,d,Δ0,ξ=1)

                ## Laplace Kernel
                # clf = SV.SVC(C=1E10,kernel="precomputed",cache_size=1000) # allocated cache in MB
                # GramTrain = Laplace_Kernel(Xtrain,Xtrain)
                # clf.fit(GramTrain, Ytrain)

                ## Gaussian Kernel
                clf = SV.SVC(C=1E10,cache_size=1000) # 800 MB allocated cache
                GramTrain = Laplace_Kernel(Xtrain,Xtrain)
                clf.fit(GramTrain, Ytrain)


                # Test Error
                    GramTest = Laplace_Kernel(Xtrain,Xtest)
                    misclassification_error_matrix[i,j,m] = testerr(clf.predict(GramTest),Ytest)*weight_band
                # α
                    alpha_mean_matrix[i,j,m] = mean(abs.(clf.dual_coef_))
                    alpha_std_matrix[i,j,m]  = std(abs.(clf.dual_coef_))
                # r_c
                    sv = Xtrain[clf.support_ .+ 1]
                    rc_mean_matrix[i,j,m],rc_std_matrix[i,j,m] = rc(sv,Δ0)
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

## Functions used for data analysis
@everywhere function smooth(X) ## for smoother plots
    tmp = copy(X)
    coeff = [1,2,1]
    coeff = coeff./sum(coeff)
    @assert isodd(length(coeff))
    s = Int(floor(length(coeff)/2))
    for i in 1+s:length(tmp)-s
        tmp[i] = X[i-s:i+s]'*coeff
    end
    return tmp
end

@everywhere function cut_zeros(error_avg) # to cut away data that is zeros
    length_PP  = size(error_avg)[1]
    length_gap = size(error_avg)[2]
    length_dim = size(error_avg)[3]
    s = zeros(length_gap,length_dim)
    for i in 1:length_gap
        for j in 1:length_dim
            if findfirst(iszero,error_avg[:,i,j,1]) == nothing
                s[i,j] = length_PP
            else
                s[i,j] = findfirst(iszero,error_avg[:,i,j,1]) - 1
            end
        end
    end
    return Int.(s)
end
