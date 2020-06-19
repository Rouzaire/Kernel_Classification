cd("D:\\Documents\\Ecole\\EPFL\\Internship_2019_ML\\Kernel Classification SVM")
using Plots,JLD,Distributed,Statistics,ColorSchemes
pyplot() ; default(:palette,ColorSchemes.tab10.colors[1:10]) ; default(:box,true) ; plot()
include("function_definitions.jl")

##
dayy = "11" ; param = load("Data\\Laplace_Kernel\\Scan_gap\\parameters_"*dayy*".jld")
PP = param["PP"]
M  = param["M"]
Δ  = param["Δ"]
dimensions = param["dimensions"]
parallelized_over = param["parallelized_over"]
ξ = 1 # 1 = Laplace # 2 = Gaussian/RBF

## 4D Matrix to store data // dim 1 : PP //  dim 2 : Δ // dim 3 : Dimensions  // dim 4 : Realisations
misclassification_error_matrix  = zeros(length(PP),length(Δ),length(dimensions),M)
alpha_mean_matrix               = zeros(length(PP),length(Δ),length(dimensions),M)
alpha_std_matrix                = zeros(length(PP),length(Δ),length(dimensions),M)
rc_mean_matrix                  = zeros(length(PP),length(Δ),length(dimensions),M)
rc_std_matrix                   = zeros(length(PP),length(Δ),length(dimensions),M)
delta_mean_matrix               = zeros(length(PP),length(Δ),length(dimensions),M)
delta_std_matrix                = zeros(length(PP),length(Δ),length(dimensions),M)

if parallelized_over == "Δ"
    for i in eachindex(Δ)
        str = "Δ_"*string(Δ[i])*"_"*dayy
        misclassification_error_matrix[:,i,:,:] = load("Data\\Laplace_Kernel\\Scan_gap\\"*str*".jld")["error"]
        alpha_mean_matrix[:,i,:,:] = load("Data\\Laplace_Kernel\\Scan_gap\\"*str*".jld")["alpha_mean_matrix"] ; alpha_std_matrix[:,i,:,:] = load("Data\\Laplace_Kernel\\Scan_gap\\"*str*".jld")["alpha_std_matrix"]
        rc_mean_matrix[:,i,:,:] = load("Data\\Laplace_Kernel\\Scan_gap\\"*str*".jld")["rc_mean_matrix"] ; rc_std_matrix[:,i,:,:] = load("Data\\Laplace_Kernel\\Scan_gap\\"*str*".jld")["rc_std_matrix"]
        delta_mean_matrix[:,i,:,:] = load("Data\\Laplace_Kernel\\Scan_gap\\"*str*".jld")["delta_mean_matrix"] ; delta_std_matrix[:,i,:,:] = load("Data\\Laplace_Kernel\\Scan_gap\\"*str*".jld")["delta_std_matrix"]
    end
elseif parallelized_over == "d"
            for i in eachindex(dimensions)
        str = "D_"*string(dimensions[i])*"_"*dayy
        misclassification_error_matrix[:,:,i,:] = load("Data\\Laplace_Kernel\\Scan_gap\\"*str*".jld")["error"]
        alpha_mean_matrix[:,:,i,:] = load("Data\\Laplace_Kernel\\Scan_gap\\"*str*".jld")["alpha_mean_matrix"] ; alpha_std_matrix[:,:,i,:] = load("Data\\Laplace_Kernel\\Scan_gap\\"*str*".jld")["alpha_std_matrix"]
        rc_mean_matrix[:,:,i,:] = load("Data\\Laplace_Kernel\\Scan_gap\\"*str*".jld")["rc_mean_matrix"] ; rc_std_matrix[:,:,i,:] = load("Data\\Laplace_Kernel\\Scan_gap\\"*str*".jld")["rc_std_matrix"]
        delta_mean_matrix[:,:,i,:] = load("Data\\Laplace_Kernel\\Scan_gap\\"*str*".jld")["delta_mean_matrix"] ; delta_std_matrix[:,:,i,:] = load("Data\\Laplace_Kernel\\Scan_gap\\"*str*".jld")["delta_std_matrix"]
    end
end
error_avg       = mean(misclassification_error_matrix,dims=4)  ; error_std       = std(misclassification_error_matrix,dims=4)
alpha_mean_avg  = mean(alpha_mean_matrix,dims=4) ; alpha_mean_std  = std(alpha_mean_matrix,dims=4)
rc_mean_avg     = mean(rc_mean_matrix,dims=4) ; rc_mean_std     = std(rc_mean_matrix,dims=4)
delta_mean_avg  = mean(delta_mean_matrix,dims=4) ; delta_mean_std  = std(delta_mean_matrix,dims=4)

s = cut_zeros(error_avg)


β = -(dimensions .- 1 .+ ξ)./(3dimensions .- 3 .+ ξ)
pow_̄α = 2ξ./(3dimensions .- 3 .+ ξ)
pow_rc = -2 ./(3dimensions .- 3 .+ ξ)


## Without margin
# coeff = [0.5,0.4,0.3,0.5,0.5]/5 ; factor = [0.5,1,1,1]
# coeff = [0.25,0.4,0.3,0.5,0.5]/2 ; factor = [0.5,1,1,1]
# p = plot(box=true,grid=nothing,yticks=nothing,legend=:bottomleft,xlabel="P",ylabel="Test Error avg. over $M realisations",title="No Gap")
# for j in 1:length(dimensions)
#     plot!(PP, 10^j * smooth(error_avg[:,1,j,1]),ribbon=10^j * factor[j]* error_std[:,1,j,1],axis=:log,color=j,label="d = $(dimensions[j]) , Slope $(3/4)")
#     # plot!(PP, 10^j * smooth(error_avg[:,1,j,1]),ribbon=10^j * factor[j]* error_std[:,1,j,1],axis=:log,color=j,label="d = $(dimensions[j]) , Slope $(-round(β[j],digits=2))")
#     plot!(PP, 10^j * coeff[j]*PP .^ (-0.75),line=:dash,axis=:log,color=j,label="")
# end
# display(p)
# savefig("Figures\\Laplace_Kernel\\No Gap\\nogap.pdf")


# p = plot(box=true,yticks=nothing,legend=:topright,xlabel="P",ylabel="̄α avg. over $M realisations",title="No Gap")
# for j in 1:length(dimensions)
#     plot!(PP,10^j * alpha_mean_avg[:,1,j,1],ribbon=10^j *0.5*alpha_mean_std[:,1,j,1],axis=:log,color=j,label="d = $(dimensions[j]) , Slope $(round(β[j],digits=2))")
#     plot!(PP,10^j * coeff[j]*PP .^ (pow_̄α[j]),line=:dash,axis=:log,color=j,label="")
#     # plot!(PP,10^j * coeff[j]*PP .^ (β[j]),line=:dot,axis=:log,color=j,label="")
# end
# display(p)
# savefig("Figures\\Laplace_Kernel\\No Gap\\alphabar_no_gap")
#

## With margin = Δ[2]
# factor = 0.25
# coeff = [0.5,0.4,0.3,0.5]
# p = plot(box=true,legend=:topleft,xlabel="P",ylabel="Test Error avg. on $M realisations",title="Gap Δ0 = $(Δ[2])")
# for j in 1:length(dimensions)
#     plot!(PP,(error_avg[:,2,j,1]),ribbon=factor*error_std[:,2,j,1],axis=:log,color=j,label="d = $(dimensions[j])")
#     plot!(PP,100*PP .^-2,line=:dash,axis=:log,color=j,label="")
# end
# display(p)
# savefig("Figures\\0test_error_gap.pdf")

## Investigate the departure from powerlaw
# factor = 0.1
# for j in 1:length(dimensions)
#     p = plot(box=true,legend=:bottomleft,xlabel="P",ylabel="Test Error avg. on $M realisations",title="Departure from powerlaw regime [d=$(dimensions[j])]")
#     # plot!(PP,0.3*PP .^-0.75,line=:dash,axis=:log,color=:black,label="No Gap")
#     for i in 1:length(Δ)
#         plot!(PP[1:s[i,j]],smooth(error_avg[1:s[i,j],i,j,1]),ribbon=factor*error_std[1:s[i,j],i,j,1],axis=:log,color=i,label="Δ0 = $(Δ[i])")
#     end
#     savefig("Figures\\Laplace_Kernel\\Gap\\gap_d"*string(dimensions[j])*".pdf")
#     savefig("Figures\\Laplace_Kernel\\Gap\\gap_d"*string(dimensions[j]))
# end

# factor = 0.1
# for j in 1:length(dimensions)
#     p = plot(box=true,legend=true,xlabel="P",ylabel="Test Error avg. on $M realisations",title="Departure from powerlaw regime [d=$(dimensions[j])]")
#     for i in 2:length(Δ)
        # yerr = factor*smooth(error_avg[1:s[i,j],i,j,1] ./ error_avg[1:s[i,j],1,j,1]).*(error_std[1:s[i,j],i,j,1]./error_avg[1:s[i,j],i,j,1] .+ error_std[1:s[i,j],1,j,1]./error_avg[1:s[i,j],1,j,1])
#         plot!(PP[1:s[i,j]],smooth(error_avg[1:s[i,j],i,j,1] ./ error_avg[1:s[i,j],1,j,1]),yaxis=:log,ribbon=yerr,color=i,label="Δ0 = $(Δ[i])")
#         plot!(PP[1:s[i,j]],exp.(-1/3*(Δ[i])^(2)*PP[1:s[i,j]] .- 0.4),line=:dash,color=i,label="")
#     end
#     plot!(NaN*PP[1:s[1,1]],NaN*exp.(-1/10*(Δ[1])^(2)*PP[1:s[1,1]] .- 0.3),line=:dash,color=:black,label="exp(-⅓⋅Δ²⋅P)")
#     # ylims!((-5,-1))
#     display(p)
#     savefig("Figures\\Laplace_Kernel\\Gap\\departure_d"*string(dimensions[j])*".pdf")
#     savefig("Figures\\Laplace_Kernel\\Gap\\departure_d"*string(dimensions[j]))
# end

## Investigation alphabar
# for j in 1:length(dimensions)
#     p = plot(box=true,legend=:topleft,xlabel="P",ylabel="̄α avg. over $M realisations",title="Departure from powerlaw regime [d=$(dimensions[j])]")
#     for i in 1:length(Δ)
#         plot!(PP,smooth(alpha_mean_avg[:,i,j,1]),ribbon=1/2*alpha_mean_std[:,i,j,1],axis=:log,color=i,label="Δ0 = $(Δ[i])")
#     end
#     plot!(PP,65*PP .^ (pow_̄α[j]),line=:dash,axis=:log,color=:black,label="Slope $(round(pow_̄α[j],digits=2))")
#     display(p)
#     savefig("Figures\\Laplace_Kernel\\alphabar_d"*string(dimensions[j])*"")
# end
#
# for j in 1:length(dimensions)
#     p = plot(box=true,legend=:bottomleft,xlabel="P",ylabel="̄α(Δ0)/̄α(Δ0=0) avg. over $M realisations",title="Departure from powerlaw regime [d=$(dimensions[j])]")
#     for i in 1:length(Δ)
#         yerr = 0.5*smooth(alpha_mean_avg[1:s[i,j],i,j,1] ./ alpha_mean_avg[1:s[i,j],1,j,1]).*(alpha_mean_std[1:s[i,j],i,j,1]./alpha_mean_avg[1:s[i,j],i,j,1] .+ alpha_mean_std[1:s[i,j],1,j,1]./alpha_mean_avg[1:s[i,j],1,j,1])
#         plot!(PP,smooth(alpha_mean_avg[:,i,j,1]./alpha_mean_avg[:,1,j,1]),xaxis=:log,ribbon=1/2*yerr,color=i,label="Δ0 = $(Δ[i])")
#     end
#     display(p)
#     savefig("Figures\\Laplace_Kernel\\departure_alphabar_d"*string(dimensions[j])*".pdf")
# end

## Investigation rc
#
# for j in 1:length(dimensions)
#     p = plot(box=true,legend=:best,xlabel="P",ylabel="̄r_c avg. over $M realisations",title="Departure from powerlaw regime [d=$(dimensions[j])]")
#     for i in 1:length(Δ)
#         plot!(PP,smooth(rc_mean_matrix[:,i,j,1]),ribbon=0*alpha_mean_std[:,i,j,1],axis=:log,color=i,label="Δ0 = $(Δ[i]) , Slope $(round(pow_̄α[j],digits=2))")
#         plot!(PP,3*PP .^ (pow_rc[j]),line=:dash,axis=:log,color=i,label="")
#     end
#     savefig("Figures\\Laplace_Kernel\\rc_d"*string(dimensions[j])*"_cube")
# end

## Investigation error as a function of the gap
p = plot(legend=:topleft,xlabel="Δ0",ylabel="-log ϵ",title="Scaling of ϵ with the gap [P = 1000]")
ind_max = minimum([findfirst(iszero,error_avg[i,:,j,1]) for j in 1:length(dimensions)]) - 1
for j in 1:length(dimensions)
    i=1
    plot!(Δ[2:ind_max],(smooth(error_avg[i,2:ind_max,j,1])),ribbon=0*error_std[i,2:ind_max,j,1],yaxis=:log,color=j,label="d = $(dimensions[j])")
    # savefig("Figures\\Laplace_Kernel\\rc_d"*string(dimensions[j])*"_cube")
end
plot!(Δ[1:ind_max], 3 .+ 2.1exp.(13*Δ[1:ind_max]),yaxis=:log,color=:black,label="y = 3 + 2 exp(13 Δ)")
xlims!(0,0.12)
savefig("Figures\\relation_error_gap")

## Investigation relation Erorr SVband
p = plot(legend=:topleft,xlabel="Δ0",ylabel="log(-log ϵ)",title="Relation between ϵ and Δ ")
for j in 1:length(dimensions)
    i=1
    plot!(delta_mean_avg[i,2:ind_max,j,1],log.(-log.(smooth(error_avg[i,2:ind_max,j,1]))),ribbon=0*error_std[i,:,j,1],color=j,label="d = $(dimensions[j])")
    # savefig("Figures\\Laplace_Kernel\\rc_d"*string(dimensions[j])*"_cube")
end
# plot!(Δ[1:25], 3 .+ 2.1exp.(13*Δ[1:25]),yaxis=:log,color=:black)
# xlims!(0,0.12)
savefig("Figures\\relation_error_svband")
