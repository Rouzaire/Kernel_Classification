cd("C:\\Users\\Ylann Rouzaire\\.julia\\environments\\ML_env")
using Pkg; Pkg.activate("."); Pkg.instantiate();
cd("D:\\Documents\\Ecole\\EPFL\\Internship_2019_ML\\Kernel Classification SVM")

PP = [10,20,30,50,100,200,300,500,1000]
Δ = [0,0.01,0.05,0.1]
dimensions = [1,2,3,5,10]
M = 1

using Distributed
addprocs(length(dimensions))
@everywhere include("function_definitions.jl")
@everywhere using SpecialFunctions,LinearAlgebra,Distributions,PyCall,Dates,JLD
@everywhere SV = pyimport("sklearn.svm")

@time pmap(Run,[PP for i in eachindex(dimensions)],[Δ for i in eachindex(dimensions)],dimensions)
JLD.save("Data\\parameters_"*string(Dates.day(now()))*".jld","PP", PP, "Δ", Δ, "dimensions", dimensions, "M", M)
# rmprocs(workers())
