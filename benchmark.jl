cd("C:\\Users\\Ylann Rouzaire\\.julia\\environments\\ML_env")
using Pkg; Pkg.activate("."); Pkg.instantiate();
cd("D:\\Documents\\Ecole\\EPFL\\Internship_2019_ML\\Kernel Classification SVM")
using Plots, SpecialFunctions, JLD, Dates,Distributed, LinearAlgebra, Distributions
pyplot() ; plot()
using SpecialFunctions,Distributed,LinearAlgebra,Distributions,PyCall,BenchmarkTools
SV = pyimport("sklearn.svm")

include("function_definitions.jl")

Δ0 = 0.5
Ptrain = Int(5E3)
Ptest = Int(1E3)
dimension = 2
δ = Ptrain^(-1/dimension)

Xtrain,Ytrain = generate_TrainSet(Ptrain,dimension,Δ0)
mean(Ytrain)
Xtest,Ytest,w = generate_TestSet(Ptest,dimension,Δ0)
mean(Ytest)

scatter([Xtrain[1,i] for i in 1:Ptrain],[Xtrain[2,i] for i in 1:Ptrain],[Xtrain[3,i] for i in 1:Ptrain],legend=nothing,camera=(30,70))
title!("TrainSet with Margin Δ0 = $Δ0")
scatter([Xtest[1,i] for i in 1:Ptest],[Xtest[2,i] for i in 1:Ptest],[Xtest[3,i] for i in 1:Ptest],legend=nothing,camera=(30,70))
title!("TestSet with Margin Δ0 = $Δ0 and SVband = 0.2")

xlabel!("x")
xlims!((-1,1))
ylabel!("y")
savefig("Figures\\distrib_XX.svg")
savefig("Figures\\distrib_XX.pdf")

##
cd("C:\\Users\\Ylann Rouzaire\\.julia\\environments\\ML_env")
using Pkg; Pkg.activate("."); Pkg.instantiate();
cd("D:\\Documents\\Ecole\\EPFL\\Internship_2019_ML\\Kernel Classification SVM")

using SpecialFunctions,LinearAlgebra,Distributions,PyCall,Dates,JLD,Distributed
include("function_definitions.jl")
SV = pyimport("sklearn.svm")

PP = [10,20,30,50,100,200,300,500] ; Ptest = 1000
PP = 5000 ; Ptest = 10000
Δ = 0.04
d = 2
M = 10
misclassification_error_matrix = zeros(length(PP))
# coeff_SV = []
for i in eachindex(PP)
    tmp = zeros(M)
    for m in 1:M
    Ptrain = PP[i]
    X,weight_band = generate_X(Ptrain,Ptest,d,Δ)
    Y             = generate_Y(X,Δ)

    clf = SV.SVC(C=1E10,kernel="precomputed",cache_size=800) # 800 MB allocated cache
    GramTrain = Laplace_Kernel(Xtrain,Xtrain)
    clf.fit(GramTrain, Ytrain)
    # alpha_std[i] = mean(std.(clf.dual_coef_))
    # if i == length(PP)
    # append!(coeff_SV,clf.dual_coef_)
    # println("#dual coeff min : ",minimum(abs.(clf.dual_coef_)))
    # # display(histogram(log10.(abs.(clf.dual_coef_')),yaxis=(:log10)))
    # display(histogram(abs.(clf.dual_coef_'),axis=(:log10)))
    # end
    # println("SV : ")
    # println("#alpha bar : ",mean(abs.(clf.dual_coef_))," ± ",std(abs.(clf.dual_coef_)))
    # println("#SVind : ",minimum([X[clf.support_ .+ 1][i][1] for i in 1:length(X[clf.support_ .+ 1])]))
    # println("#SVind : ",maximum([X[clf.support_ .+ 1][i][1] for i in 1:length(X[clf.support_ .+ 1])]))
    GramTest = Laplace_Kernel(Xtrain,Xtest)
    tmp[m] = testerr(clf.predict(GramTest),Ytest)*weight_band
    end

    misclassification_error_matrix[i] = mean(tmp)
    # alpha_avg[i] = mean(abs.(clf.dual_coef_))

end # Ptrain


## Compute Delta
PP = unique(Int.(round.(10.0 .^range(log10(100),stop=log10(1E3),length=50))))
dimensions = [2] ; ξ=1; β = -(dimensions .- 1 .+ ξ)./(3dimensions .- 3 .+ ξ)
M = 50
Delta0 = [0.0,0.02,0.05,0.1]
delta_max_matrix = zeros(length(PP),length(Delta0),length(dimensions),M)
delta_mean_matrix = zeros(length(PP),length(Delta0),length(dimensions),M)
@time for i in eachindex(PP)
    Ptrain = PP[i]
    for j in eachindex(Delta0)
        Δ0 = Delta0[j]
        for k in eachindex(dimensions)
            d = dimensions[k]
            println("P = $Ptrain , Δ = $Δ0 , d = $d")
            for m in 1:M
                Xtrain,Ytrain = generate_TrainSet(Ptrain,d,Δ0)
                clf = SV.SVC(C=1E10,kernel="precomputed",cache_size=1000) # allocated cache (in MB)
                GramTrain = Kernel_Matrix(Xtrain,Xtrain)
                clf.fit(GramTrain,Ytrain)
                tmp = abs.(Xtrain[:,clf.support_ .+ 1][1,:])
                delta_max_matrix[i,j,k,m] = maximum(tmp)
                delta_mean_matrix[i,j,k,m] = mean(tmp)
            end
        end
    end
end
delta_max_matrix_mean = mean(delta_max_matrix,dims=4)
delta_max_matrix_std = std(delta_max_matrix,dims=4)
delta_mean_matrix_mean = mean(delta_mean_matrix,dims=4)
delta_mean_matrix_std = std(delta_mean_matrix,dims=4)
for j in eachindex(dimensions)
    p = plot(box=true,legend=:bottomleft,xlabel="P",ylabel="Δ avg. over $M realisations",title="Departure from powerlaw regime [d=$(dimensions[j])]")
    for i in eachindex(Delta0)
        plot!(PP,smooth(delta_max_matrix_mean[:,i,j,1]).-Delta0[i],ribbon=0.25*smooth(delta_max_matrix_std[:,i,j,1]),axis=:log,color=i,label="Δ0 = $(Delta0[i])")
    end
    plot!(PP,3*PP .^ β[1],axis=:log,color=:black,label="Slope β")
    # display(p)
    savefig("Figures\\Laplace_Kernel\\testdelta")
end

# # α
# alpha_mean_matrix = mean(abs.(clf.dual_coef_))
# alpha_std_matrix  = std(abs.(clf.dual_coef_))
# # r_c
# sv = Xtrain[:,clf.support_ .+ 1]
# rc_mean_matrix,rc_std_matrix= rc(sv,Δ0)

##
