using Plots,JLD,Distributed,Statistics,LaTeXStrings
pyplot()
include("function_definitions.jl")

##
dayy = "5" ; param = load("Data\\parameters_"*dayy*".jld")
PP   = param["PP"] ; Δ = param["Δ"] ; dimensions = param["dimensions"] ; M = param["M"]

## 4D Matrix to store data // dim 1 : PP //  dim 2 : Δ // dim 3 : Realisations  // dim 4 : dimension
misclassification_error_matrix  = NaN*zeros(length(PP),length(Δ),M,length(dimensions))
for i in eachindex(dimensions)
    misclassification_error_matrix[:,:,:,i] = load("Data\\"*string(dimensions[i])*"D_"*dayy*".jld")["error"]
end
error_avg  = mean(misclassification_error_matrix,dims=3)
error_std  = std(misclassification_error_matrix,dims=3)

ξ = 1 # 1 = Laplace # 2 = Gaussian=RBF
β = (dimensions .- 1 .+ ξ)./(3dimensions .- 3 .+ ξ)

##
for i in 1:length(Δ)
    plot(box=true)
    for j in 1:length(dimensions)
        plot!(PP,error_avg[:,i,1,j],label="d = $(dimensions[j])",color=j)
        plot!(PP,PP .^ β[j],line=:dash,color=j)
    end
    xlabel!("P")
    ylabel!("Test Error")
    title!("Δ_0 = $(Δ[i])")
    savefig("Figures\\test_error_margin=$(Δ[i]).pdf")
end
