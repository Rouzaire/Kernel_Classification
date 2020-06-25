cd("C:\\Users\\Ylann Rouzaire\\.julia\\environments\\ML_env")
using Pkg; Pkg.activate("."); Pkg.instantiate();
cd("D:\\Documents\\Ecole\\EPFL\\Internship_2019_ML\\Kernel Classification SVM")
using Distributed, Plots, SpecialFunctions, JLD, Dates, LinearAlgebra, Distributions,ColorSchemes,PyCall,StatsBase,SharedArrays
pyplot() ; default(:palette,ColorSchemes.tab10.colors[1:10]); default(:box,true) ; default(:legend,:topright) ; default(:grid,false) ; default(:markerstrokecolor,:auto) ; plot()
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

GramTest  = Kernel_Matrix(Xtrain,Xtest)
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

## Analysis of data
data = load("Data\\Decision_Boundary\\f0.jld")
M    = data["M"]
gap  = vec(data["gap"])
PP   = data["PP"]
h    = data["h"]
a    = data["a"]
f0   = data["f0"] # dim1 = Δ0 ; dim2 = Ptrain ;dim3 = M real ; dim4 = f0 along y axis
f0c = abs.(f0 .- mean(f0,dims=4))

# One example :
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

## Spatial Correlation of decision boundary f(y)
corr_Matrix = zeros(size(f0))
lags = 0:length(f0[1,1,1,:])-1
for i in 1:size(f0)[1] , j in 1:size(f0)[2] , m in 1:size(f0)[3]
    corr_Matrix[i,j,m,:] = StatsBase.autocor(f0[i,j,m,:],lags,demean=true) # "demean=true" centers the data
end
corr_Matrix
corr = mean(corr_Matrix,dims=3) ## pb de nan pour les deux premières valeurs de P

for i in 1:size(f0)[1]
    plot(xlabel="τ",ylabel="C(τ)",title="Δ0 = $(gap[i])",ylims=(-0.25,1))
    for j in 3:size(f0)[2]
        plot!(lags/length(lags)*2,corr[i,j,1,:],label="P = $(PP[j])")
    end
    plot!(lags/length(lags)*2,zeros(length(lags)),color=:grey,linestyle=:dash,label="")
    savefig("Figures\\autocorrelation_gap$(gap[i]).pdf")
end

for j in 3:size(f0)[2]
    plot(xlabel="τ",ylabel="C(τ)",title="P = $(PP[j])",ylims=(-0.25,1))
    for i in 1:size(f0)[1]
        plot!(lags/length(lags)*2,corr[i,j,1,:],label="Δ0 = $(gap[i])")
    end
    plot!(lags/length(lags)*2,zeros(length(lags)),color=:grey,linestyle=:dash,label="")
    savefig("Figures\\autocorrelation_P$(PP[j]).pdf")
end

## D'après ces figures, il se pourrait que le kernel isotrope sous-jacent soit : (reference = https://www.nr.no/directdownload/2437/Abrahamsen_-_A_Review_of_Gaussian_random_fields_and_correlation.pdf)
    # (1-ax)exp(-ax), sa FT est positive donc PDness OK
    # damped sth (damped sin ?)
    # un simple laplace (il y a peut etre des pb avec le fait que les données ne soient pas périodiques)
# Essayons de tirer des realisations de chacun de ces kernels pour voir
k(h) = sinc(h*10pi)
function ke(h) return exp(-h^2 * pi)*cos(pi*h) end
# # k(h) = (1-2h)*exp(-h/100)
function Gram(X)
    K = 1E-5 .+ ones(length(X),length(X))
    for i=1:length(X) , j = i+1:length(X)
        K[i,j] = ke(2abs(X[i]-X[j]))
    end
    return Symmetric(K)
end
X = -1:0.005:1
L = length(X)
Z = rand(MvNormal(zeros(L),Gram(X)),3)
p = plot(xlabel="x",ylabel="y",legend=:topright)
plot!(range(-0.2,stop=0.2,length=100),fill(1,100),color=:grey,fill=(0,0.25,:grey),label="Gap at interface")
plot!(range(-0.2,stop=0.2,length=100),fill(-1,100),color=:grey,fill=(0,0.25,:grey),label="")
plot!(Z[:,1]/5,X,color=:black,label="")
# plot!(Z[:,1]/5,X,color=1,label="")
# plot!(Z[:,2]/5,X,color=2,label="")
# plot!(Z[:,3]/5,X,color=3,label="")
xlims!(-0.5,0.5)
savefig("Figures\\Report\\realGP")

## Let's try to see if these results change when the datat is ~ Uniform on the unit sphere
gap = 0.0
Xtrain,Ytrain = generate_TrainSet(50,2,gap) # trainset uniform on the 2D sphere
clf = SV.SVC(C=1E10,kernel="precomputed",cache_size=1000)
GramTrain = Kernel_Matrix(Xtrain)
clf.fit(GramTrain, Ytrain)
predTrain = clf.predict(GramTrain)
# scatter(Xtrain[1,:],Xtrain[2,:],Xtrain[3,:],color=predTrain .+ 3,xlabel="x",ylabel="y",zlabel="f(x,y)",camera=c1)
# savefig("Figures\\tests")

a = 0.5 ## keep only points such that -a < x < a for efficiency
Xtest = grid_2D_sphere(a,Int(1E6)) # testset almost uniform on the 2D sphere (Fibonacci sphere) # NB : runtime linear in number of test points
GramTest = Kernel_Matrix(Xtrain,Xtest)
predTest = clf.predict(GramTest)
# scatter(Xtest[1,:],Xtest[2,:],Xtest[3,:],xlabel="x",ylabel="y",zlabel="z",label="",ms=1.5)
scatter(Xtest[1,:],Xtest[2,:],Xtest[3,:],color=predTest .+ 3,xlabel="x",ylabel="y",zlabel="f(x,y)",label="",ms=1.5,camera=(0,0))
scatter!([NaN,NaN],[NaN,NaN],m=:o,color=2,label="Predicted -1")
scatter!([NaN,NaN],[NaN,NaN],m=:o,color=4,label="Predicted +1")
savefig("Figures\\Report\\prediction_sphere_zoom")
# savefig("Figures\\Report\\prediction_sphere_zoom.pdf")
# @time @gif for i in 0:2:360
#     scatter!(camera=(0,i))
# end

@time f0 = extrapolate_root_sphere(clf,Xtrain,Xtest)
plot(1:length(f0),f0)
savefig("Figures\\testf")

lags = 0:length(filter(!isnan,f0))-1

c = autocor(filter(!isnan,f0),lags,demean=true)

plot(c)
savefig("Figures\\testc")

# Accumulate stats
M = 1
gap = [0.0,0.1,0.2,0.3,0.5]
gap = [0.0]
PP = Int.(round.(10.0 .^range(log10(50),stop=log10(500),length=8))) ## the larger P, the closer is the decision function f to the interface
# Creation of the testing "grid" such that -a < x < a
    a = 0.5 ; N = Int(1E5) ; Xtest = grid_2D_sphere(a,N) # Note : Expectation[length(Xtest)] = aN

f0 = zeros(length(gap),length(PP),M,396) ## 396 is the length of the returned array of function extrapolate_root_sphere
@time for i in eachindex(gap)
    for j in eachindex(PP)
        Δ0 = gap[i]
        Ptrain = PP[j]

        for m in 1:M
            println("Gap = $Δ0, P = $Ptrain, M = $m/$M")
            # Generating data ~Uniform([-1,1]²) with gap Δ0 at the interface
            Xtrain,Ytrain = generate_TrainSet(Ptrain,2,Δ0) # trainset uniform on the 2D sphere

            clf = SV.SVC(C=1E10,kernel="precomputed",cache_size=2000)
            clf.fit(Kernel_Matrix(Xtrain), Ytrain)
            f = clf.decision_function(Kernel_Matrix(Xtrain,Xtest))
            f0[i,j,m,:] = extrapolate_root_sphere(clf,Xtrain,Xtest)
        end #M
    end # PP
end ## \Delta0
JLD.save("Data\\Decision_Boundary\\f0_sphere.jld","f0",f0,"gap",gap,"M",M,"a",a,"N",N,"PP",PP)

## Analysis of data
data = load("Data\\Decision_Boundary\\f0_sphere.jld")
M    = data["M"]
gap  = data["gap"]
PP   = data["PP"]
a    = data["a"]
f0   = data["f0"] # dim1 = Δ0 ; dim2 = Ptrain ;dim3 = M real ; dim4 = f0
f0c = abs.(f0 .- mean(f0,dims=4))

# One example :
# for aa = [1,2] , b = [1]
#     p = plot(xlabel="x",ylabel="y",title="Examples of Decision Boundaries",legend=:topright)
#     if gap[aa] > 0
#         plot!(range(-gap[aa]/2,stop=gap[aa]/2,length=100),fill(1,100),color=:grey,fill=(0,0.25,:grey),label="Gap at interface")
#         plot!(range(-gap[aa]/2,stop=gap[aa]/2,length=100),fill(-1,100),color=:grey,fill=(0,0.25,:grey),label="")
#     end
#     xlims!(-0.5,0.5)
#     for i in 1:4
#         plot!(f0[aa,b,i,:],-a:2a/size(f0)[end]:a,linestyle=:solid,color=i,label="") # ,markersize=1,marker=:o P = $(PP[b]) , Gap Δ0 = $(gap[aa])
#     end
#     # savefig("Figures\\Decision_boundary\\examples_P = $(PP[b]) , Gap Δ0 = $(gap[a]).pdf")
#     # savefig("Figures\\Decision_boundary\\examples_P = $(PP[b]) , Gap Δ0 = $(gap[a])")
#     savefig("Figures\\Decision_boundary\\test = $(PP[b]) , Gap Δ0 = $(gap[aa])")
# end

# plot(xlabel="P",ylabel="Maximum amplitude of f0")
# for i in 1:length(gap)
#     plot!(PP,mean(maximum(abs.(f0),dims=4),dims=3)[i,:],ribbon = 0.25std(maximum(abs.(f0),dims=4),dims=3)[i,:],label="Gap = $(gap[i])",axis=:log)
# end
# plot!(PP, 2PP .^(-1/2),color=:black,ls=:dash,label="1/√P")
# savefig("Figures\\amplitude.pdf")
""
# hP = Array{Any,2}(undef,length(gap),length(PP))
# println()
# for i in eachindex(gap), j in eachindex(PP)
#     try
#         hP[i,j] = normalize(fit(Histogram,abs.(vec(f0[i,j,:,:])),closed=:right),mode=:density)
#         # append!(hP[i,j].weights,Int(ceil(hP[i,j].weights[end]/2)))
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
#     savefig("Figures\\Decision_boundary\\testhist_gap_P$(PP[j])")
#     # savefig("Figures\\Decision_boundary\\hist_gap_P$(PP[j]).pdf")
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

## Spatial Correlation of decision boundary f(y)
corr_Matrix = zeros(size(f0))
lags = 0:length(f0[1,1,1,:])-1
for i in 1:size(f0)[1] , j in 1:size(f0)[2] , m in 1:size(f0)[3]
    corr_Matrix[i,j,m,:] = StatsBase.autocor(f0[i,j,m,:],lags,demean=true) # "demean=true" centers the data
end
corr_Matrix
corr = mean(corr_Matrix,dims=3) ## pb de nan pour les deux premières valeurs de P

for i in 1:size(f0)[1]
    plot(xlabel="τ",ylabel="C(τ)",title="Δ0 = $(gap[i])")
    # plot(xlabel="τ",ylabel="C(τ)",title="Δ0 = $(gap[i])",ylims=(-0.25,1))
    for j in 3:size(f0)[2]
        plot!(lags/length(lags)*2,corr[i,j,1,:],label="P = $(PP[j])")
    end
    plot!(lags/length(lags)*2,zeros(length(lags)),color=:grey,linestyle=:dash,label="")
    savefig("Figures\\testautocorrelation_gap$(gap[i])")
end
plot()
for k in 1:10
    c = autocor(filter(!isnan,f0[1,1,k,:]))
# x = (f0[1,1,2,:])
    plot!(c)
end
savefig("Figures\\testt")
# for j in 3:size(f0)[2]
#     plot(xlabel="τ",ylabel="C(τ)",title="P = $(PP[j])",ylims=(-0.25,1))
#     for i in 1:size(f0)[1]
#         plot!(lags/length(lags)*2,corr[i,j,1,:],label="Δ0 = $(gap[i])")
#     end
#     plot!(lags/length(lags)*2,zeros(length(lags)),color=:grey,linestyle=:dash,label="")
#     savefig("Figures\\autocorrelation_P$(PP[j])")
# end
