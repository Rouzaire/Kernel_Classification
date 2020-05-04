using Plots, SpecialFunctions, JLD, Dates,Distributed, LinearAlgebra, Distributions
pyplot()

dimension = 2
Δ0 = 1/2
Ptrain = 10000
Ptest = 0
N = Ptrain+ Ptest

X = generate_X(Ptrain,Ptest,dimension,Δ0)
scatter([X[i][1] for i in 1:N],[X[i][2] for i in 1:N],[X[i][3] for i in 1:N],label=nothing,camera=(30,70))
savefig("distrib_X.pdf")
