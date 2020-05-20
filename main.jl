cd("C:\\Users\\Ylann Rouzaire\\.julia\\environments\\ML_env")
using Pkg; Pkg.activate("."); Pkg.instantiate();
cd("D:\\Documents\\Ecole\\EPFL\\Internship_2019_ML\\Kernel Classification SVM")

parallelized_over = "d" # change accordingly [PP for i in eachindex(...)] and addprocs
# Δ = 10.0 .^range(-4,stop=log10(0.01),length=5)
Δ = [0,1E-3,5E-3,1E-2,5E-2]
PP = unique(Int.(round.(10.0 .^range(log10(50),stop=log10(1E4),length=30))))
dimensions = [2,3,10]
M = 25

using Distributed
addprocs(length(dimensions))
@everywhere include("function_definitions.jl")
@everywhere using SpecialFunctions,LinearAlgebra,Distributions,PyCall,Dates,JLD
@everywhere SV = pyimport("sklearn.svm")

# @time Run(parallelized_over,[PP for i in eachindex(Δ)],Δ,[dimensions for i in eachindex(Δ)],[M for i in eachindex(Δ)]) ## parallelized_over = "Δ"
@time Run(parallelized_over,[PP for i in eachindex(dimensions)],[Δ for i in eachindex(dimensions)],dimensions,[M for i in eachindex(dimensions)]) ## parallelized_over = "dimensions"
JLD.save("Data\\parameters_"*string(Dates.day(now()))*".jld","PP", PP, "Δ", Δ, "dimensions", dimensions, "M", M,"parallelized_over",parallelized_over)
rmprocs(workers())
