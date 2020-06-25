cd("C:\\Users\\Ylann Rouzaire\\.julia\\environments\\ML_env")
using Pkg; Pkg.activate("."); Pkg.instantiate();
cd("D:\\Documents\\Ecole\\EPFL\\Internship_2019_ML\\Kernel Classification SVM")
using Plots, SpecialFunctions, JLD, Dates,Distributed, LinearAlgebra, Distributions,ColorSchemes,PyCall
pyplot() ; default(:palette,ColorSchemes.tab10.colors[1:10]); default(:box,true) ; default(:legend,:best) ; default(:grid,false) ; plot()
SV = pyimport("sklearn.svm")
# np = pyimport("numpy")

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
PP = unique(Int.(round.(10.0 .^range(log10(50),stop=log10(1E4),length=10))))
dimensions = [10] ; ξ=1; β = -2(dimensions .- 1 .+ ξ)./(5dimensions .- 5 .+ 2ξ)
M = 30
Delta0 = [0.0,0.05,0.1,0.5]
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
                GramTrain = Kernel_Matrix(Xtrain)
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
    p = plot(box=true,legend=:bottomleft,xlabel="P",ylabel="Δ avg. over $M realisations")
    for i in eachindex(Delta0)
        plot!(PP,smooth(delta_mean_matrix_mean[:,i,j,1]).-Delta0[i]/2,ribbon=smooth(delta_mean_matrix_std[:,i,j,1]),axis=:log,color=i,label="Δ0 = $(Delta0[i])")
        if Delta0[i] > 0 plot!(PP,fill(Delta0[i],length(PP)),color=i,ls=:dash,axis=:log,label="") end
    end
    plot!(PP,0.9*PP .^ -0.5,axis=:log,color=:black,label="Slope 1/2")
    # display(p)
    savefig("Figures\\Report\\scalingdelta10D.pdf")
end
JLD.save("Data\\Laplace_Kernel\\delta10D.jld","delta_mean_matrix_mean",delta_mean_matrix_mean,"delta_mean_matrix_std",delta_mean_matrix_std,"PP",PP,"Delta0",Delta0,"dimensions", dimensions, "M",M)
# data = JLD.load("Data\\Laplace_Kernel\\delta10D.jld")
# PP = data["PP"]
# delta_mean_matrix_mean = data["delta_mean_matrix_mean"]
# delta_mean_matrix_std = data["delta_mean_matrix_std"]
# dimensions = data["dimensions"]
# Delta0 = data["Delta0"]


## where are the SV ?
d = 2
Δ0 = 0.5
Ptrain = 10000
Ptest = Ptrain
Xtrain,Ytrain = generate_TrainSet(Ptrain,d,Δ0)

clf = SV.SVC(C=1E10,kernel="precomputed",cache_size=1000) # allocated cache (in MB)
GramTrain = Kernel_Matrix(Xtrain,Xtrain)
clf.fit(GramTrain, Ytrain)

sv = Xtrain[:,clf.support_ .+ 1]
nsv = sum(clf.n_support_)
svm = sv[:,1:clf.n_support_[1]]
svp = sv[:,clf.n_support_[1]+1:end]

# plot()
# scatter!([Xtrain[1,i] for i in 1:Ptrain],[Xtrain[2,i] for i in 1:Ptrain],[Xtrain[3,i] for i in 1:Ptrain],label="Training Set",color=1,camera=(30,70),zlabel="x3")
# # scatter!(svm[1,:],svm[2,:],svm[3,:],color=:red,label="Support Vectors Class  -1",camera=(30,70))
# # scatter!(svp[1,:],svp[2,:],svp[3,:],color=:green,label="Support Vectors Class +1",camera=(30,70))
# scatter!(sv[1,:],sv[2,:],sv[3,:],color=:red,label="Support Vectors Class  -1",camera=(30,70))
# scatter!(svp[1,:],svp[2,:],svp[3,:],color=:green,label="Support Vectors Class  +1",camera=(30,70))
# xlabel!("x1")
# ylabel!("x2")
# savefig("Figures\\whereSV.pdf")
#
#
# scatter([svm[2,i] for i in 1:clf.n_support_[1]],[svm[3,i] for i in 1:clf.n_support_[1]],color=:red,label="Support Vectors Class  -1",camera=(30,70))
# scatter!([svp[2,i] for i in 1:clf.n_support_[2]],[svp[3,i] for i in 1:clf.n_support_[2]],color=:green,label="Support Vectors Class +1",camera=(30,70))
# xlabel!("x2")
# ylabel!("x3")
# savefig("Figures\\proj.png")
