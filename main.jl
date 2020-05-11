cd("C:\\Users\\Ylann Rouzaire\\.julia\\environments\\ML_env")
using Pkg; Pkg.activate("."); Pkg.instantiate();
cd("D:\\Documents\\Ecole\\EPFL\\Internship_2019_ML\\Kernel Classification SVM")

parallelized_over = "Δ"
Δ = [el for el in 0:0.1:0.5]
PP = Int.(round.(10.0 .^range(1,stop=3,length=30)))
dimensions = [3]
M = 50

using Distributed
addprocs(length(Δ))
@everywhere include("function_definitions.jl")
@everywhere using SpecialFunctions,LinearAlgebra,Distributions,PyCall,Dates,JLD
@everywhere SV = pyimport("sklearn.svm")

@time Run(parallelized_over,[PP for i in eachindex(Δ)],Δ,[dimensions for i in eachindex(Δ)],[M for i in eachindex(Δ)])
JLD.save("Data\\parameters_"*string(Dates.day(now()))*".jld","PP", PP, "Δ", Δ, "dimensions", dimensions, "M", M,"parallelized_over",parallelized_over)
rmprocs(workers())
