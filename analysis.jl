cd("D:\\Documents\\Ecole\\EPFL\\Internship_2019_ML\\Kernel Classification SVM")
using Plots,JLD,Distributed,Statistics
pyplot()
include("function_definitions.jl")

##
dayy = "5" ; param = load("Data\\parameters_"*dayy*".jld")
PP   = param["PP"] ; Δ = param["Δ"] ; dimensions = param["dimensions"] ; M = param["M"]

## 4D Matrix to store data // dim 1 : PP //  dim 2 : Δ // dim 3 : Realisations  // dim 4 : dimension
misclassification_error_matrix  = zeros(length(PP),length(Δ),M,length(dimensions))
for i in eachindex(dimensions)
    misclassification_error_matrix[:,:,:,i] = load("Data\\"*string(dimensions[i])*"D_"*dayy*".jld")["error"]
end
error_avg  = mean(misclassification_error_matrix,dims=3)
error_std  = std(misclassification_error_matrix,dims=3)

ξ = 1 # 1 = Laplace # 2 = Gaussian/RBF
β = (dimensions .- 1 .+ ξ)./(3dimensions .- 3 .+ ξ)
factor = 1
## Without margin
coeff = [0.5,0.4,0.3,0.5]
plot(box=true,yticks=nothing,legend=:topleft,xlabel="P",ylabel="Test Error",title="No Gap")
for j in 1:length(dimensions)
    plot!(PP,10^j * error_avg[:,1,1,j],ribbon=factor*error_std[:,1,1,j],axis=:log,color=j,label="d = $(dimensions[j]) , Slope $(round(β[j],digits=2))")
    plot!(PP,10^j * coeff[j]*PP .^ (-β[j]),line=:dash,axis=:log,color=j,label="")
end
savefig("Figures\\test_error_no_gap.pdf")

## With margin = Δ[2]
coeff = [0.5,0.4,0.3,0.5]
plot(box=true,yticks=nothing,legend=:topleft,xlabel="P",ylabel="Test Error",title="Gap Δ0 = $(Δ[2])")
for j in 1:length(dimensions)
    plot!(PP,10^j * error_avg[:,2,1,j],ribbon=factor*error_std[:,2,1,j],axis=:log,color=j,label="d = $(dimensions[j]) , Slope $(round(β[j],digits=2))")
    plot!(PP,10^j * coeff[j]*PP .^ (-β[j]),line=:dash,axis=:log,color=j,label="")
end
savefig("Figures\\test_error_gap.pdf")
