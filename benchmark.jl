cd("C:\\Users\\Ylann Rouzaire\\.julia\\environments\\ML_env")
using Pkg; Pkg.activate("."); Pkg.instantiate();
using Plots, SpecialFunctions, JLD, Dates,Distributed, LinearAlgebra, Distributions
pyplot() ; plot()
using SpecialFunctions,Distributed,LinearAlgebra,Distributions,PyCall,BenchmarkTools
SV = pyimport("sklearn.svm")

include("function_definitions.jl")

Δ0 = 0.5
Ptrain = Int(1E3)
Ptest = Int(1E3)
dimension = 1
δ = Ptrain^(-1/dimension)

Xtrain,Ytrain = generate_TrainSet(Ptrain,dimension,Δ0)
mean(Ytrain)
Xtest,Ytest,w = generate_TestSet(Ptest,dimension,Δ0)
mean(Ytest)

scatter([Xtrain[1,i] for i in 1:Ptrain],[Xtrain[2,i] for i in 1:Ptrain],legend=nothing,camera=(30,70))
scatter([Xtest[1,i] for i in 1:Ptest],[Xtest[2,i] for i in 1:Ptest],legend=nothing,camera=(30,70))
# xlabel!("x")
# ylabel!("y")
# title!("Distribution with Margin Δ0 = $Δ0")
savefig("distrib_test.pdf")

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
    Xtest,Ytest   = extract_TestSet(X,Y,Ptest)
    Xtrain,Ytrain = extract_TrainSet(X,Y,Ptrain)

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
misclassification_error_matrix
println(length(coeff_SV))
plot(box=true,xlabel="log10(SV coefficients)",ylabel="Count")
histogram!(log10.(abs.(coeff_SV)),yaxis=(:log10),label="")
savefig("Figures\\CoeffSV.pdf")

## Compute rc
sv = X[clf.support_ .+ 1]
svy = Bool.((generate_Y(sv) .+ 1)/2)
sv_plus = sv[svy]
sv_minus = sv[.!svy]

rc_plus  = [sort([norm(sv_plus[i]-sv_plus[j]) for j in eachindex(sv_plus)])[2] for i in eachindex(sv_plus)]
rc_minus = [sort([norm(sv_minus[i]-sv_minus[j]) for j in eachindex(sv_minus)])[2] for i in eachindex(sv_minus)]
rc = vcat(rc_plus,rc_minus)
println("+ : ",mean(rc_plus)," ± ",std(rc_plus))
println("- : ",mean(rc_minus)," ± ",std(rc_minus))
println("Both : ",mean(rc)," ± ",std(rc))
    
