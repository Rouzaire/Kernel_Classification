## Some basic functions
# @everywhere dist(x,y) = norm(x-y) # euclidien distance
@everywhere testerr(pred,Y) = mean(pred .!= Y)
@everywhere NaNmean(x) = mean(filter(!isnan,x))
@everywhere NaNstd(x)  = std(filter(!isnan,x))


## Generation of data, uniformly on the d-dimensional unit-hypersphere (d=1 -> X lies on unit circle in R^2 // d=2 -> X lies on unit sphere in R^3)
    # Notation : Δ0 is the gap size bewteen 2 interfaces and SVband is the SV bandwidth

@everywhere function generate_Y(X::Array{Float64,2},Δ0::Float64)
    M = size(X)[2]
    labels = zeros(Int,M)
    for i in 1:M
        if X[1,i] ≥ Δ0/2 labels[i] = + 1 # label function : y(x) = sgn(x1)
        else             labels[i] = - 1
        end
    end
    return labels
end

@everywhere function generate_TrainSet(Ptrain::Int,d::Int,Δ0::Float64)
    # @assert isinteger(d) ; @assert d > 0 ; @assert Δ0 ≥ 0 # Δ0 = margin separating decision boundaries

    M = Int(ceil(2*Ptrain/(1-SpecialFunctions.erf(Δ0/2)))) # generate more data than necessary
    X = rand(MvNormal(zeros(d+1),I(d+1)),M)
    for m in 1:M
        X[:,m] = X[:,m]./norm(X[:,m]) ## normalizing it on the unit sphere
    end
    X = X[:,[Δ0/2 ≤ abs(X[1,i]) for i in 1:M]] # Keep only the points out-of-margin and hope that there is at least N of them
    @assert length(X) ≥ Ptrain
    X = X[:,1:Ptrain]

    return X , generate_Y(X,Δ0)
end

@everywhere function generate_TestSet(Ptest::Int,Ptrain::Int,d::Int,Δ0::Float64)
    # 0<ξ<2 is the exponent governing the cusp (ξ=1 for Laplace kernel, ξ=2 for RBF/Gaussian kernel)
    # Note that ξ = min(2ν,2) for a kernel in the Matérn family
    # @assert isinteger(d) ; @assert d > 0 ; @assert Δ0 ≥ 0 # Δ0 = margin separating decision boundaries

    # Points will only be generated in this band, because beyond one can be sure that
        # they will be correctly classified. It has to be wide enough to be sure that
        # all misclassified points are contained and thin enough to save memory later
        # in the code
    SVband = min(0.2,2*Ptrain^(-d/(3d-2)))
    weight_band = 2SVband/(π - Δ0) # fraction of the surface occupied by this band (to weight the final result)

    M = Int(ceil(2*Ptest/weight_band/(1-SpecialFunctions.erf(Δ0/2)))) # generate more data than necessary
    X = rand(MvNormal(zeros(d+1),I(d+1)),M)
    for m in 1:M
        X[:,m] = X[:,m]./norm(X[:,m]) ## normalizing it on the unit sphere
    end
    X = X[:,[Δ0/2 ≤ abs(X[1,i]) ≤ Δ0/2 + SVband for i in 1:M]] # Keep only the points out-of-margin and hope that there is at least N of them
    @assert length(X) ≥ Ptest
    X = X[:,1:Ptest]

    # println("N = $Ptest ,  Actually generated = $M , Wasted = $((Nkept-Ptest))")
    return X , generate_Y(X,Δ0) , weight_band
end

@everywhere function rc(sv::Array{Float64,2},Δ0) ## returns the mean minimum distance separating support vectors (SV)
    ## sv is a (dimension x numberSV) matrix
    svy = Bool.((generate_Y(sv,Δ0) .+ 1)/2)
    sv_plus  = sv[:,svy]
    sv_minus = sv[:,.!svy]
    rc_plus  = compute_rc(sv_plus)
    rc_minus = compute_rc(sv_minus)
    rc = vcat(rc_plus,rc_minus)
    return NaNmean(rc),NaNstd(rc) # filters out NaN values
end

@everywhere function compute_rc(sv::Array{Float64,2}) # auxillary function that returns a list of the minimal distance to all other SV for each SV
    rc  = NaN*zeros(size(sv)[2])
    for i in eachindex(rc)
        dist_to_all_other_SV = [norm(sv[:,i]-sv[:,j]) for j in eachindex(rc)]
        if length(dist_to_all_other_SV)>1
            rc[i] = sort(dist_to_all_other_SV)[2] # the minimum will alway be zero (distance to itself). Small arrays so no need for O(n) algo
        end
    end
    return rc
end
## Matérn Covariance functions
@everywhere function Matérn(h,ν::Float64,σ::Float64=1.0,ρ::Float64=100.0) # Matérn kernel
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

@everywhere function GaussianKernel(h,σ::Float64=1.0,ρ::Float64=100)
    return σ^2*exp(-h^2/2/ρ^2)
end

@everywhere function Kernel_Matrix(X::Array{Float64,2},Y::Array{Float64,2})::Array{Float64,2} ## general case
    # ρ = 100.0 ; σ = 1.0
    Px = size(X)[2] ; Py = size(Y)[2]
    K = ones(Float64,Px,Py)
    for i in 1:Px , j in 1:Py
        K[i,j] = Matérn(norm(X[:,i]-Y[:,j]),1/2)
    end
    return K'
end

@everywhere function Kernel_Matrix(X::Array{Float64,2})::Array{Float64,2} ## special case when X=Y (Kernel_Matrix for trainset, evaluated in (Xtrain,Xtrain))
    # ρ = 100.0 ; σ = 1.0
    Px = size(X)[2]
    K = ones(Float64,Px,Px)
    for i in 1:Px , j in i+1:Px
        K[i,j] = K[j,i] = Matérn(norm(X[:,i]-X[:,j]),1/2)
    end
    return K
end


@everywhere function Run_fixed_dimension(PP,Δ,d,M=1) ## d=dimension is a integer passed in argument and the scan is over M and the vectors PP, Δ (gaps between interfaces)
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
            low = 1E2 ; high = 1E3
            Ptest = Int(round((min(high,max(Ptrain,low))))) # enforce low ≤ Ptest = Ptrain ≤ high
            # Ptest = 1000

            println("SVM for P = $Ptrain, Δ = $Δ0 , Time : "*string(Dates.hour(now()))*"h"*string(Dates.minute(now()))*" [d = $d]")
            for m in 1:M
                ## If Kernel = Laplace
                    # Xtrain,Ytrain = generate_TrainSet(Ptrain,d,Δ0)
                    # Xtest,Ytest,weight_band = generate_TestSet(Ptest,d,Δ0)
                    #
                    # clf = SV.SVC(C=1E10,kernel="precomputed",cache_size=1000) # allocated cache (in MB)
                    # GramTrain = Kernel_Matrix(Xtrain,Xtrain)
                    # clf.fit(GramTrain, Ytrain)
                    #
                    # GramTest = Kernel_Matrix(Xtrain,Xtest)
                    # misclassification_error_matrix[i,j,m] = testerr(clf.predict(GramTest),Ytest)*weight_band
                    # global str = "Laplace_Kernel\\D_"*string(d)*"_"*string(Dates.day(now())) # where to store data, at the end of the function

                ## If Kernel = Gaussian (default kernel of the Python SVC machine)
                    # Xtrain,Ytrain = generate_TrainSet(Ptrain,d,Δ0)
                    # Xtest,Ytest,weight_band = generate_TestSet(Ptest,d,Δ0)
                    #
                    # ρ = 100
                    # clf = SV.SVC(C=1E10,gamma=1.0/(2*ρ^2),cache_size=1000) # allocated cache (in MB)
                    # clf.fit(Xtrain', Ytrain)
                    #
                    # misclassification_error_matrix[i,j,m] = testerr(clf.predict(Xtest'),Ytest)*weight_band
                    # global str = "Gaussian_Kernel\\D_"*string(d)*"_"*string(Dates.day(now())) # where to store data, at the end of the function



        ## The following is the same for any kernel
                # α
                tmp = abs.(clf.dual_coef_)
                alpha_mean_matrix[i,j,m] = mean(tmp)
                alpha_std_matrix[i,j,m]  = std(tmp)
                # r_c
                sv = Xtrain[:,clf.support_ .+ 1]
                rc_mean_matrix[i,j,m],rc_std_matrix[i,j,m] = rc(sv,Δ0)
            end # Realisations
        end # Δ0
    end # Ptrain

    ## Save Data for later analysis
    JLD.save("Data\\"*str*".jld", "error", misclassification_error_matrix,"alpha_mean_matrix",alpha_mean_matrix,"alpha_std_matrix",alpha_std_matrix,"rc_mean_matrix",rc_mean_matrix,"rc_std_matrix",rc_std_matrix, "PP", PP, "Δ", Δ, "d", d, "M", M)
end

@everywhere function Run_fixed_delta(PP,Δ0,dim,M=1) ## Δ0 is a scalar passed in argument and the scan is over M and the vectors PP, dim
    ## 3D Matrix to store data // dim 1 : PP //  dim 2 : dimensions // dim 3 : Realisations
    misclassification_error_matrix  = NaN*zeros(length(PP),length(dim),M)
    alpha_mean_matrix  = NaN*zeros(length(PP),length(dim),M)
    alpha_std_matrix   = NaN*zeros(length(PP),length(dim),M)
    rc_mean_matrix     = NaN*zeros(length(PP),length(dim),M)
    rc_std_matrix      = NaN*zeros(length(PP),length(dim),M)
    delta_mean_matrix  = NaN*zeros(length(PP),length(dim),M)
    delta_std_matrix   = NaN*zeros(length(PP),length(dim),M)

    for i in eachindex(PP)
        Ptrain = PP[i]

        for j in eachindex(dim)
            d = dim[j]
            low = 1E2 ; high = 5E3
            Ptest = Int(round((min(high,max(Ptrain,low))))) # enforce low ≤ Ptest ≤ high
            # Ptest = 1000
            println("SVM for P = $Ptrain , Δ = $Δ0 , Time : "*string(Dates.hour(now()))*"h"*string(Dates.minute(now()))*" [d = $d]")

            for m in 1:M
                ## If Kernel = Laplace
                    Xtrain,Ytrain = generate_TrainSet(Ptrain,d,Δ0)
                    Xtest,Ytest,weight_band = generate_TestSet(Ptest,Ptrain,d,Δ0)

                    clf = SV.SVC(C=1E10,kernel="precomputed",cache_size=1000) # allocated cache (in MB)
                    GramTrain = Kernel_Matrix(Xtrain,Xtrain)
                    clf.fit(GramTrain, Ytrain)

                    GramTest = Kernel_Matrix(Xtrain,Xtest)
                    misclassification_error_matrix[i,j,m] = testerr(clf.predict(GramTest),Ytest)*weight_band
                    global str = "Laplace_Kernel\\Δ_"*string(Δ0)*"_"*string(Dates.day(now())) # where to store data, at the end of the function

                ## If Kernel = Gaussian (default kernel of the Python SVC machine)
                    # Xtrain,Ytrain = generate_TrainSet(Ptrain,d,Δ0)
                    # Xtest,Ytest,weight_band = generate_TestSet(Ptest,d,Δ0)
                    #
                    # ρ = 100 # scale >> typical width = 1 (variance of the distrib of data for Normal(Hypercube) of Unif(Hypersphere))
                    # clf = SV.SVC(C=1E10,gamma=1.0/(2*ρ^2),cache_size=1000) # allocated cache (in MB)
                    # clf.fit(Xtrain',Ytrain)
                    #
                    # misclassification_error_matrix[i,j,m] = testerr(clf.predict(Xtest'),Ytest)*weight_band
                    # global str = "Gaussian_Kernel\\Δ_"*string(Δ0)*"_"*string(Dates.day(now())) # where to store data, at the end of the function

        ## The following is the same for any kernel
                # α
                    tmp = abs.(clf.dual_coef_)
                    alpha_mean_matrix[i,j,m] = mean(tmp)
                    alpha_std_matrix[i,j,m]  = std(tmp)
                # r_c
                    sv = Xtrain[:,clf.support_ .+ 1]
                    rc_mean_matrix[i,j,m],rc_std_matrix[i,j,m] = rc(sv,Δ0)
                # \Delta
                    tmp = abs.(Xtrain[:,clf.support_ .+ 1][1,:])
                    delta_mean_matrix[i,j,m] = mean(tmp)
                    delta_std_matrix[i,j,m] = std(tmp)
            end # Realisations
        end # Δ0
    end # Ptrain

    ## Save Data for later analysis
    JLD.save("Data\\"*str*".jld", "error", misclassification_error_matrix,"alpha_mean_matrix",alpha_mean_matrix,"alpha_std_matrix",alpha_std_matrix,"rc_mean_matrix",rc_mean_matrix,"rc_std_matrix",rc_std_matrix,"delta_mean_matrix",delta_mean_matrix,"delta_std_matrix",delta_std_matrix, "PP", PP, "Δ", Δ0, "d", dim, "M", M)
end

@everywhere function Run(parallelized_over,args...)
    @assert parallelized_over in ["d","Δ"] println("Choose the function name among \"PP\",\"d\",\"Δ\".")
    if     parallelized_over == "d" pmap(Run_fixed_dimension,args...)
    elseif parallelized_over == "Δ" pmap(Run_fixed_delta,args...)
    end
end
# end of functions used in main.jl

## Functions used in decision_boundary.jl to generate the meshgrid and to compute the subspace where f(x,y) = 0
@everywhere function grid_2D_plane(Δ0,a,h)
    nx = Int((2a + Δ0)/h)+1 ; ny = nh
    grid = Array{Array{Float64,1},2}(undef,ny,nx) # [x,y] coord. of every point on the grid
    for i=1:ny , j=1:nx
        grid[i,j] = [h*(j-1) - a - Δ0/2,1-h*i] ## one only constructs points such that -a < x < a and -1 < y < 1
    end
    Xtest = Array{Float64,2}(undef,2,length(grid)) # necessary format to be accepted by Kernel_Matrix(.,.)
    for i in eachindex(grid)
        Xtest[:,i] = grid[i]
    end
    return Xtest
end

@everywhere function grid_2D_sphere(a,N=Int(1E6)) ## implements the Fibonacci (usual 2D) unit sphere restreigned -a < x < a   N = number of points
    indices = 1:N
    theta = acos.(1 .- 2*indices/N)
    phi = pi * (1 + sqrt(5)) * indices
    x, y, z = cos.(phi) .* sin.(theta), sin.(phi) .* sin.(theta), cos.(theta)

    Xtest = Array{Float64,2}(undef,3,N) # necessary format to be accepted by Kernel_Matrix(.,.)
    Xtest[1,:] = x ; Xtest[2,:] = y ; Xtest[3,:] = z

    Xtest = Xtest[:,sortperm(Xtest[1,:])] ## sort points in order of increasing x1
    ind_min = findfirst(x -> x>-a,Xtest[1,:]) + 1
    ind_max = findfirst(x -> x>+a,Xtest[1,:]) - 1
    Xtest = Xtest[:,ind_min:ind_max] ## keep only points in -a < x < a for efficiency

    return Xtest ## Expectation[length(Xtest)] = aN and it turns out the Variance is ridiculously low
end

@everywhere function extrapolate_root(X::Array{Float64,1},h::Float64,xmin::Float64)
    if X[1]*X[end] > 0 return NaN
    else
        ind = 1
        while X[ind]*X[ind+1] > 0
            ind = ind + 1
        end ## ind is now the index of the last negative value
        v1 = X[ind]
        v2 = X[ind+1]
        # @assert v1 < 0 ; @assert v2 > 0 ;
        return xmin + (ind-1)*h -h*(v2+v1)/(v2-v1) # linear extrapolation of the root (where the decision boudary would have been =0)
    end
end

@everywhere function extrapolate_root_sphere_aux(clf::PyObject, Xtrain::Matrix{Float64},Xtest::Matrix{Float64},y::Float64,z::Float64,tol::Float64=4E-2)
    ## Next line of code will construct a thin (width = tol) layer : it will play the role of the regularly spaced layer in the 2D plane case.
        ## To do so, only keep data points such that |y* - y| < tol/2. If one only did that, one would end up with data whose z corrdinate can be
        ## both positive and negative. Since we do not want that in order to construct the vecteor f0 in the correct ordering, we select data that
        ## have the same signe as z. Here,         ".*" plays the role of an AND boolean function.
    X    = Xtest[:,(abs.(Xtest[2,:] .- y) .< tol/2) .* Bool.((1 .+ sign.(z*Xtest[3,:]))/2) ]
    Gram = Kernel_Matrix(Xtrain,X)
    f    = clf.decision_function(Gram) # if this line throws an error, it is probably because X is empty -> not enough training point AND/OR tol too low

    # starts looking for the location of the change of sign of the decision_function = change in prediction
    if f[1]*f[end] > 0  return NaN # decision function does not change sign (and it cannot change twice)
    else
        ind = 1
        while f[ind]*f[ind+1] > 0
            ind = ind + 1
        end ## ind is now the index of the last negative value of decision_function
        v1 = f[ind]
        v2 = f[ind+1]
        h  = X[1,ind+1] - X[1,ind] # distance in the x direction between the last negative value and the first positive value
        # @assert v1 < 0 ; @assert v2 > 0  # for benchmarks
        return X[1,ind] -h*(v2+v1)/(v2-v1) # X[1,ind] is the x value of the last negative value of decision_function
    end
end


@everywhere function extrapolate_root_sphere(clf::PyObject, Xtrain::Matrix{Float64},Xtest::Matrix{Float64})::Vector{Float64}
    # Arbitrary tolerance (= width of the slices.)
        # If too big, result will be less precise.
        # If too small, runtime increases (linearly in 1/tol).
        # If extremely small, risk of have slices with no datapoints.
    tol = 1E-2
    f0 = zeros(length(1-tol:-tol:tol-1) + length(2tol-1:tol:1-2tol))
    i = 1 # index unsed in the following for loops to fill up f0
    ## We place ourselves in the y-z plane.
    ## Since the sphere is periodic, we arbitrarily choose to construct the upper hemisphere

    ## We start with the z>0 hemisphere
    for y in 1-tol:-tol:tol-1
        z = sqrt(1-y^2) ## actually, only the sign of z matters
        f0[i] = extrapolate_root_sphere_aux(clf,Xtrain,Xtest,y,z,tol)
        i = i + 1
    end

    ## We finish by the z<0 hemisphere
    for y in 2tol-1:tol:1-2tol
        z = -sqrt(1-y^2) ## actually, only the sign of z matters
        f0[i] = extrapolate_root_sphere_aux(clf,Xtrain,Xtest,y,z,tol)
        i = i + 1
    end
    # @assert i-1 == length(f0) # for benchmarks
    return f0
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
