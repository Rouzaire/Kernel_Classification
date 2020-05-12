cd("C:\\Users\\Ylann Rouzaire\\.julia\\environments\\ML_env")
using Pkg; Pkg.activate("."); Pkg.instantiate();
cd("D:\\Documents\\Ecole\\EPFL\\Internship_2019_ML\\Kernel Classification SVM")

parallelized_over = "d" # change accordingly [PP for i in eachindex(...)] and addprocs
Δ = [0]
PP = Int.(round.(10.0 .^range(1,stop=log10(500),length=30)))
dimensions = [1]
M = 20

using Distributed
addprocs(length(dimensions))
@everywhere include("function_definitions.jl")
@everywhere using SpecialFunctions,LinearAlgebra,Distributions,PyCall,Dates,JLD
@everywhere SV = pyimport("sklearn.svm")

@time Run(parallelized_over,[PP for i in eachindex(dimensions)],[Δ for i in eachindex(dimensions)],dimensions,[M for i in eachindex(dimensions)])
JLD.save("Data\\parameters_"*string(Dates.day(now()))*".jld","PP", PP, "Δ", Δ, "dimensions", dimensions, "M", M,"parallelized_over",parallelized_over)
rmprocs(workers())
