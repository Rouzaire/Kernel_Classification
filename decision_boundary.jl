cd("C:\\Users\\Ylann Rouzaire\\.julia\\environments\\ML_env")
using Pkg; Pkg.activate("."); Pkg.instantiate();
cd("D:\\Documents\\Ecole\\EPFL\\Internship_2019_ML\\Kernel Classification SVM")
using Plots, SpecialFunctions, JLD, Dates,Distributed, LinearAlgebra, Distributions,ColorSchemes,PyCall,StatsBase
pyplot() ; default(:palette,ColorSchemes.tab10.colors[1:10]); default(:box,true) ; default(:legend,:topright) ; default(:grid,false) ; plot()
SV = pyimport("sklearn.svm")
include("function_definitions.jl")
## This file is meant to investigate the change (if any) in the behaviour
    ## of the decision function (called "f") in presence of a gap \Delta0 at the interface
    ## between labels.
c1 = (-25,20) ; c2 = (0,90) # angles of camera for plotting in 3D
d = 1 # in this investigation, so that visualisation is made simpler/possible, we'll work with data ~Uniform([-1,1]²) It corresponds to d=1 and no normalisation to unit length in the generation of synthetic data
Δ0 = 0.2
Ptrain = 100 ## the larger P, the closer is the decision function f to the interface
## Generating data ~Uniform([-1,1]²) with gap Δ0 at the interface
tmp = 2rand(2,Int(round(1.5Ptrain/(1-Δ0/2)))) .- 1 # generate more than necessary
tmp = tmp[:,[abs(tmp[1,i]) > Δ0/2 for i in 1:Int(round(1.5Ptrain/(1-Δ0/2)))]][:,1:Ptrain] # keep only point out of the gap
Xtrain,Ytrain = tmp,generate_Y(tmp,Δ0)

clf = SV.SVC(C=1E10,kernel="precomputed",cache_size=1000)
GramTrain = Kernel_Matrix(Xtrain,Xtrain)
clf.fit(GramTrain, Ytrain)

# surface(Xtrain[1,:],Xtrain[2,:],clf.predict(GramTrain),camera=c1,xlabel="x",ylabel="y",zlabel="f(x,y)")
# savefig("Figures\\decision_boundary_gap.png")
# savefig("Figures\\decision_boundary_gap.png")
# surface(Xtrain[1,:],Xtrain[2,:],clf.predict(GramTrain),camera=c2,xlabel="x",ylabel="y",zlabel="f(x,y)")
# savefig("Figures\\decision_boundary_gapUP.png")
# savefig("Figures\\decision_boundary_gapUP.pdf")

# Note : the finer the meshgrid, the larger the number of points and the plotting can take enormous time
h = 0.01; nh = Int(2/h) # meshgrid for testing points in order to visualise f
a = 0.1
nx = Int((2a + Δ0)/h)+1 ; ny = nh
grid = Array{Array{Float64,1},2}(undef,ny,nx) # [x,y] coord. of every point on the grid
for i=1:ny , j=1:nx
    grid[i,j] = [h*(j-1) - a - Δ0/2,1-h*i] ## one only constructs points such that -a < x < a and -1 < y < 1
end
Xtest = Array{Float64,2}(undef,2,length(grid)) # necessary format to be accepted by Kernel_Matrix(.,.)
for i in eachindex(grid)
    Xtest[:,i] = grid[i]
end
GramTest = Kernel_Matrix(Xtrain,Xtest)
f = clf.decision_function(GramTest)
pred = clf.predict(GramTest)


surface(Xtest[1,:],Xtest[2,:],f,camera=c1,xlabel="x",ylabel="y",zlabel="f(x,y)",title="Δ0 = $Δ0")
savefig("Figures\\Decision_boundary\\Gap_P"*string(Ptrain)*"_c1")
savefig("Figures\\Decision_boundary\\Gap_P"*string(Ptrain)*"_c1.pdf")

surface(Xtest[1,:],Xtest[2,:],f,camera=c2,xlabel="x",ylabel="y",zlabel="f(x,y)")
savefig("Figures\\Decision_boundary\\Gap_P"*string(Ptrain)*"_c2")
savefig("Figures\\Decision_boundary\\Gap_P"*string(Ptrain)*"_c2.pdf")

surface(Xtest[1,:],Xtest[2,:],pred,camera=c2,xlabel="x",ylabel="y",zlabel="sign(f(x,y))",legend=false)
savefig("Figures\\Decision_boundary\\Gap_P"*string(Ptrain)*"_pred")
savefig("Figures\\Decision_boundary\\Gap_P"*string(Ptrain)*"_pred.pdf")

#
# fshaped = reshape(f,ny,nx) # same shape as the grid
# f0 = zeros(ny) # spans the y axis and contains the x values where the decision function vanishes
# for j in 1:ny
#     f0[j] = extrapolate_root(fshaped[j,:],h,-Δ0/2-a)
# end
# p = plot(xlabel="x",ylabel="y",title="Examples of Decision Boundaries")
# plot!(range(-.1,stop=.1,length=100),fill(1,100),color=:grey,fill=(0,0.25,:grey),label="Gap at interface")
# plot!(range(-.1,stop=.1,length=100),fill(-1,100),color=:grey,fill=(0,0.25,:grey),label="")
# xlims!(-a/2-Δ0/2,a/2+Δ0/2)
# plot!(f0,1:-h:h-1,linestyle=:solid,color=3,label="P = $Ptrain , Gap Δ0 = $Δ0") # ,markersize=1,marker=:o
# plot!(legend=:topright,label="")
# savefig("Figures\\Decision_boundary\\f0")
# savefig("Figures\\Decision_boundary\\f0.pdf")
""
# savefig("Figures\\Decision_boundary\\Gap_P"*string(Ptrain)*"_f0.pdf")
## Accumulate stats in order to have meaningful histograms
# M = 100
# gap = [0.0 0.1 0.2 0.3 0.5]
# PP = Int.(round.(10.0 .^range(log10(20),stop=log10(400),length=10))) ## the larger P, the closer is the decision function f to the interface
# # Creation of the testing grid. Note : the finer the meshgrid, the larger the number of points and complexity is O(N²) = O(1/h)²
#     h = 0.005; nh = Int(2/h) # meshgrid for testing points in order to visualise f
#     nx = Int(round(1/h))+1 ; ny = nh+1 ## no matter the gap, one scans from -0.5 to 0.5, hence the "1" in 1/h
#     grid = Array{Array{Float64,1},2}(undef,ny,nx) # [x,y] coord. of every point on the grid
#     for i=1:ny , j=1:nx
#         grid[i,j] = [h*(j-1) - 0.5,1-h*(i-1)] ## one only constructs points such that -a < x < a and -1 < y < 1
#     end
#     Xtest = Array{Float64,2}(undef,2,length(grid)) # necessary format to be accepted by Kernel_Matrix(.,.)
#     for i in eachindex(grid) Xtest[:,i] = grid[i] end
#
# f0 = zeros(length(gap),length(PP),M,ny) # spans the y axis and contains the x values where the decision function vanishes
# @time for i in eachindex(gap)
#     for j in eachindex(PP)
#         Δ0 = gap[i]
#         Ptrain = PP[j]
#         a = 0.5 .- Δ0
#
#         for m in 1:M
#             println("Gap = $Δ0, P = $Ptrain, M = $m/$M ")
#             # Generating data ~Uniform([-1,1]²) with gap Δ0 at the interface
#             tmp = 2rand(2,Int(round(2Ptrain/(1-Δ0/2)))) .- 1 # generate more than necessary
#             tmp = tmp[:,[abs(tmp[1,i]) > Δ0/2 for i in 1:Int(round(2Ptrain/(1-Δ0/2)))]][:,1:Ptrain] # keep only point out of the gap
#             Xtrain,Ytrain = tmp,generate_Y(tmp,Δ0)
#             clf = SV.SVC(C=1E10,kernel="precomputed",cache_size=2000)
#             clf.fit(Kernel_Matrix(Xtrain), Ytrain)
#             f = clf.decision_function(Kernel_Matrix(Xtrain,Xtest))
#             fshaped = reshape(f,ny,nx) # same shape as the grid
#             for k in 1:ny
#                 f0[i,j,m,k] = extrapolate_root(fshaped[k,:],h,-0.5)
#             end
#         end #M
#     end # PP
# end ## \Delta0
# JLD.save("Data\\Decision_Boundary\\f0.jld","f0",f0,"gap",gap,"M",M,"h",h,"a",0.5 .- gap,"PP",PP)

# Analysis of data
# data = load("Data\\Decision_Boundary\\f0.jld")
# M    = data["M"]
# gap  = data["gap"]
# gap  = [0.0, 0.1, 0.2, 0.3, 0.5]
# PP   = data["PP"]
# h    = data["h"]
# a    = data["a"]
# f0   = data["f0"] # dim1 = Δ0 ; dim2 = Ptrain ;dim3 = M real ; dim4 = f0 along y axis
# f0c = abs.(f0 .- mean(f0,dims=4))
# # One example :
# for a = [1,3,5] , b = [1,10]
#     p = plot(xlabel="x",ylabel="y",title="Examples of Decision Boundaries",legend=:topright)
#     if gap[a] > 0
#         plot!(range(-gap[a]/2,stop=gap[a]/2,length=100),fill(1,100),color=:grey,fill=(0,0.25,:grey),label="Gap at interface")
#         plot!(range(-gap[a]/2,stop=gap[a]/2,length=100),fill(-1,100),color=:grey,fill=(0,0.25,:grey),label="")
#     end
#     xlims!(-0.5,0.5)
#     for i in 1:4
#         plot!(f0[a,b,i,:],1:-h:-1,linestyle=:solid,color=i,label="") # ,markersize=1,marker=:o P = $(PP[b]) , Gap Δ0 = $(gap[a])
#     end
#     savefig("Figures\\Decision_boundary\\examples_P = $(PP[b]) , Gap Δ0 = $(gap[a]).pdf")
#     savefig("Figures\\Decision_boundary\\examples_P = $(PP[b]) , Gap Δ0 = $(gap[a])")
# end
# hP = Array{Any,2}(undef,length(gap),length(PP))
# println()
# for i in eachindex(gap), j in eachindex(PP)
#     try
#         hP[i,j] = normalize(fit(Histogram,abs.(vec(f0[i,j,:,:])),closed=:right),mode=:density)
#         append!(hP[i,j].weights,Int(ceil(hP[i,j].weights[end]/2)))
#     catch
#         println(i,",",j)
#     end
# end
#
# for j in eachindex(PP)
#     p = plot(xlabel="|x*| such that f(x*,y) = 0",ylabel="Probability",title="P = $(PP[j])",legend=:topright)
#     for i in 1:length(gap)
#         try
#             plot!(hP[i,j].edges,hP[i,j].weights,color=i,yaxis=:log,label="Δ0 = $(gap[i])")
#         catch
#         end
#     end
#     savefig("Figures\\Decision_boundary\\hist_gap_P$(PP[j])")
#     savefig("Figures\\Decision_boundary\\hist_gap_P$(PP[j]).pdf")
# end
#
# for i in eachindex(gap)
#     p = plot(xlabel="|x*| such that f(x*,y) = 0",ylabel="Probability",title="Δ0 = $(gap[i])",legend=:topright)
#     for j in 2:length(PP)
#         try
#             plot!(hP[i,j].edges,hP[i,j].weights,color=j,yaxis=:log,label="P = $(Int(round(PP[j],digits=-1)))")
#         catch
#         end
#     end
#     savefig("Figures\\Decision_boundary\\hist_P_gap$(gap[i])")
#     savefig("Figures\\Decision_boundary\\hist_P_gap$(gap[i]).pdf")
# end
