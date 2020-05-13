cd("D:\\Documents\\Ecole\\EPFL\\Internship_2019_ML\\Kernel Classification SVM")
using Plots,JLD,Distributed,Statistics
pyplot()
include("function_definitions.jl")

##
dayy = "13" ; param = load("Data\\parameters_"*dayy*".jld")
PP   = param["PP"] ; M = param["M"]
Δ = param["Δ"]
dimensions = param["dimensions"]
# parallelized_over = "d"
parallelized_over = param["parallelized_over"]

## 4D Matrix to store data // dim 1 : PP //  dim 2 : Δ // dim 3 : Dimensions  // dim 4 : Realisations
misclassification_error_matrix  = zeros(length(PP),length(Δ),length(dimensions),M)
alpha_mean_matrix               = zeros(length(PP),length(Δ),length(dimensions),M)
alpha_std_matrix                = zeros(length(PP),length(Δ),length(dimensions),M)
rc_mean_matrix                  = zeros(length(PP),length(Δ),length(dimensions),M)
rc_std_matrix                   = zeros(length(PP),length(Δ),length(dimensions),M)

if parallelized_over == "Δ"
    for i in eachindex(Δ)
        str = "Δ_"*string(Δ[i])*"_"*dayy
        misclassification_error_matrix[:,i,:,:] = load("Data\\"*str*".jld")["error"]
        alpha_mean_matrix[:,i,:,:] = load("Data\\"*str*".jld")["alpha_mean_matrix"] ; alpha_std_matrix[:,i,:,:] = load("Data\\"*str*".jld")["alpha_std_matrix"]
        rc_mean_matrix[:,i,:,:] = load("Data\\"*str*".jld")["rc_mean_matrix"] ; rc_std_matrix[:,i,:,:] = load("Data\\"*str*".jld")["rc_std_matrix"]
    end
elseif parallelized_over == "d"
    for i in eachindex(dimensions)
        str = "D_"*string(dimensions[i])*"_"*dayy
        misclassification_error_matrix[:,:,i,:] = load("Data\\"*str*".jld")["error"]
        alpha_mean_matrix[:,:,i,:] = load("Data\\"*str*".jld")["alpha_mean_matrix"] ; alpha_std_matrix[:,:,i,:] = load("Data\\"*str*".jld")["alpha_std_matrix"]
        rc_mean_matrix[:,:,i,:] = load("Data\\"*str*".jld")["rc_mean_matrix"] ; rc_std_matrix[:,:,i,:] = load("Data\\"*str*".jld")["rc_std_matrix"]
    end
end
error_avg       = mean(misclassification_error_matrix,dims=4)
error_std       = std(misclassification_error_matrix,dims=4)
alpha_mean_avg  = mean(alpha_mean_matrix,dims=4)
alpha_mean_std  = std(alpha_mean_matrix,dims=4)
rc_mean_avg     = mean(rc_mean_matrix,dims=4)
rc_mean_std     = std(rc_mean_matrix,dims=4)

ξ = 1 # 1 = Laplace # 2 = Gaussian/RBF

β = (dimensions .- 1 .+ ξ)./(3dimensions .- 3 .+ ξ)
pow_̄α = 2ξ./(3dimensions .- 3 .+ ξ)
pow_rc = -2 ./(3dimensions .- 3 .+ ξ)




dimensions = dimensions .- 1
βsphere = (2dimensions .- 1 .+ 2ξ)./(dimensions)/(2ξ+3)
pow_̄α_sphere = 2ξ./(3dimensions .- 3 .+ ξ)
pow_rc_sphere = -2 ./(3dimensions .- 3 .+ ξ)

## Without margin
coeff = [0.5,0.4,0.3,0.5,0.5]
plot(box=true,yticks=nothing,legend=:topright,xlabel="P",ylabel="Test Error avg. over $M realisations",title="No Gap")
for j in 1:length(dimensions)
    plot!(PP,10^j * error_avg[:,1,j,1],ribbon=10^j *0.5*error_std[:,1,j,1],axis=:log,color=j,label="d = $(dimensions[j]) , Slope $(round(β[j],digits=2))")
    plot!(PP,10^j * coeff[j]*PP .^ (-βsphere[j]),line=:dash,axis=:log,color=j,label="")
end
savefig("Figures\\test_error_no_gap.pdf")

plot(box=true,yticks=nothing,legend=:topright,xlabel="P",ylabel="̄α avg. over $M realisations",title="No Gap")
for j in 1:length(dimensions)
    plot!(PP,10^j * alpha_mean_avg[:,1,j,1],ribbon=10^j *0.5*alpha_mean_std[:,1,j,1],axis=:log,color=j,label="d = $(dimensions[j]) , Slope $(round(β[j],digits=2))")
    plot!(PP,10^j * coeff[j]*PP .^ (pow_αbar[j]),line=:dash,axis=:log,color=j,label="")
    # plot!(PP,10^j * coeff[j]*PP .^ (-β[j]),line=:dot,axis=:log,color=j,label="")
end
xlabel!("P")
savefig("Figures\\alphabar_no_gap.pdf")


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
# plot(box=true,legend=:topright,xlabel="P",ylabel="Test Error avg. on $M realisations",title="Departure from powerlaw regime")
# for j in eachindex(dimensions)
#     for i in eachindex(Δ)
#         plot!(PP,error_avg[:,i,j,1]/error_avg[1,i,j,1],ribbon=error_std[:,i,j,1],axis=:log,color=i,label="Δ = $(Δ[i])")
#     end
# end
# # savefig("Figures\\departure.pdf")
