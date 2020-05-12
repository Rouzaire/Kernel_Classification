cd("D:\\Documents\\Ecole\\EPFL\\Internship_2019_ML\\Kernel Classification SVM")
using Plots,JLD,Distributed,Statistics
pyplot()
include("function_definitions.jl")

##
dayy = "12" ; param = load("Data\\parameters_"*dayy*".jld")
PP   = param["PP"] ; M = param["M"]
Δ = param["Δ"]
dimensions = param["dimensions"]
# parallelized_over = "d"
parallelized_over = param["parallelized_over"]

## 4D Matrix to store data // dim 1 : PP //  dim 2 : Δ // dim 3 : Dimensions  // dim 4 : Realisations
misclassification_error_matrix  = zeros(length(PP),length(Δ),length(dimensions),M)
if parallelized_over == "Δ"
    for i in eachindex(Δ)
        str = "Δ_"*string(Δ[i])*"_"*dayy
        misclassification_error_matrix[:,i,:,:] = load("Data\\"*str*".jld")["error"]
        # misclassification_error_matrix[:,:,:,i] = load("Data\\"*string(dimensions[i])*"D_"*dayy*".jld")["error"]
    end
elseif parallelized_over == "d"
    for i in eachindex(dimensions)
        str = "D_"*string(dimensions[i])*"_"*dayy
        println(size(load("Data\\"*str*".jld")["error"]))
        misclassification_error_matrix[:,i,:,:] = load("Data\\"*str*".jld")["error"]
        # misclassification_error_matrix[:,:,:,i] = load("Data\\"*string(dimensions[i])*"D_"*dayy*".jld")["error"]
    end
end
error_avg  = mean(misclassification_error_matrix,dims=4)
error_std  = std(misclassification_error_matrix,dims=4)

ξ = 1 # 1 = Laplace # 2 = Gaussian/RBF
β = (dimensions .- 1 .+ ξ)./(3dimensions .- 3 .+ ξ)

# ## Without margin
# coeff = [0.5,0.4,0.3,0.5]
# plot(box=true,yticks=nothing,legend=:topleft,xlabel="P",ylabel="Test Error avg. on $M realisations",title="No Gap")
# for j in 1:length(dimensions)
#     plot!(PP,10^j * error_avg[:,1,j,1],ribbon=10^j *error_std[:,1,j,1],axis=:log,color=j,label="d = $(dimensions[j]) , Slope $(round(β[j],digits=2))")
#     plot!(PP,10^j * coeff[j]*PP .^ (-β[j]),line=:dash,axis=:log,color=j,label="")
# end
# savefig("Figures\\test_error_no_gap.pdf")
#
# ## With margin = Δ[2]
# factor = 0.5
# coeff = [0.5,0.4,0.3,0.5]
# plot(box=true,yticks=nothing,legend=:topleft,xlabel="P",ylabel="Test Error avg. on $M realisations",title="Gap Δ0 = $(Δ[2])")
# for j in 1:length(dimensions)
#     plot!(PP,10^j * error_avg[:,2,j,1],ribbon=10^j * factor*error_std[:,2,j,1],axis=:log,color=j,label="d = $(dimensions[j])")
#     plot!([10,150],10^j * coeff[j]*[10,150] .^ (-β[j]),line=:dash,axis=:log,color=j,label="")
# end
# savefig("Figures\\test_error_gap.pdf")

## Investigate the departure from powerlaw
plot(box=true,legend=:topright,xlabel="P",ylabel="Test Error avg. on $M realisations",title="Departure from powerlaw regime")
for j in eachindex(dimensions)
    for i in eachindex(Δ)
        plot!(PP,error_avg[:,i,j,1]/error_avg[1,i,j,1],ribbon=error_std[:,i,j,1],axis=:log,color=i,label="Δ = $(Δ[i])")
    end
end
xlabel!("P")
# savefig("Figures\\departure.pdf")
