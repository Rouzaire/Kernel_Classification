# cd("C:\\Users\\Ylann Rouzaire\\.julia\\environments\\ML_env")
# using Pkg; Pkg.activate("."); Pkg.instantiate();
# cd("D:\\Documents\\Ecole\\EPFL\\Internship_2019_ML\\Kernel Classification SVM")
# using Plots, SpecialFunctions, JLD, Dates,Distributed, LinearAlgebra, Distributions,ColorSchemes,PyCall
# pyplot() ; default(:palette,ColorSchemes.tab10.colors[1:10]); default(:box,true) ; default(:legend,:best) ; plot()
# SV = pyimport("sklearn.svm")
# include("function_definitions.jl")

## This file is meant to investigate the change (if any) in the behaviour
    ## of the decision function (called "f") in presence of a gap \Delta0 at the interface
    ## between labels.
c1 = (-25,20) ; c2 = (0,90) # angles of camera for plotting in 3D
d = 1 # in this investigation, so that visualisation is made simpler/possible, we'll work with data ~Uniform([-1,1]²) It corresponds to d=1 and no normalisation to unit length in the generation of synthetic data
Δ0 = 0.5
Ptrain = 10000 ## the larger P, the closer is the decision function f to the interface
# tmp = 2(rand(2,Ptrain) .- 0.5) ; Xtrain,Ytrain = tmp,generate_Y(tmp,Δ0) # unif on 2D plane [-1,1]²

# clf = SV.SVC(C=1E10,kernel="precomputed",cache_size=1000)
# GramTrain = Kernel_Matrix(Xtrain,Xtrain)
# clf.fit(GramTrain, Ytrain)

# surface(Xtrain[1,:],Xtrain[2,:],clf.predict(GramTrain),camera=c1,zlabel="f(x,y)")
# surface(Xtrain[1,:],Xtrain[2,:],clf.predict(GramTrain),camera=c2,zlabel="f(x,y)")
# xlabel!("x")
# ylabel!("y")
# # savefig("Figures\\decision_boundary_gapUP.png")
# # savefig("Figures\\decision_boundary_gapUP.pdf")

# Note : the finer the meshgrid, the larger the number of points and the plotting can take enormous time
# h = 0.005 ; nh = Int(2/h) # meshgrid for testing points in order to visualise f
# a = 0.2
# Xtmp = Array{Array{Float64,1},2}(undef,Int(2a/h),nh) # [x,y] coord. of every point on the grid
# for i=1:Int(2a/h) , j=1:nh
#     Xtmp[i,j] = [h*i - a + Δ0/2,h*j - 1] ## one only constructs points such that -a < x < a and -1 < y < 1
# end
# Xtest = Array{Float64,2}(undef,2,length(Xtmp)) # necessary format to be accepted by Kernel_Matrix(.,.)
# for i in eachindex(Xtmp)
#     Xtest[:,i] = Xtmp[i]
# end
# GramTest = Kernel_Matrix(Xtrain,Xtest)
# f = clf.decision_function(GramTest)
# pred = clf.predict(GramTest)

# surface(Xtest[1,:],Xtest[2,:],f,camera=c1,xlabel="x",ylabel="y",zlabel="f(x,y)")
# savefig("Figures\\Decision_boundary\\Gap_P"*string(Ptrain)*"_c1")
# savefig("Figures\\Decision_boundary\\Gap_P"*string(Ptrain)*"_c1.pdf")
# surface(Xtest[1,:],Xtest[2,:],f,camera=c2,xlabel="x",ylabel="y",zlabel="f(x,y)")
# savefig("Figures\\Decision_boundary\\Gap_P"*string(Ptrain)*"_c2")
# savefig("Figures\\Decision_boundary\\Gap_P"*string(Ptrain)*"_c2.pdf")


# fshaped = reshape(f,Int(2a/h),nh)
# f0 = zeros(nh)
# for j in 1:nh
#     f0[j] = extrapolate_root(fshaped[:,j],h,Δ0/2-a)
# end

# p = plot(xlabel="x",ylabel="y",title="Decision Boundary")
# plot!(f0,h-1:h:1,label="f(x,y) = 0") # ,markersize=1,marker=:o
# plot!(fill(mean(f0),nh),h-1:h:1,color=:black,label="Mean of f")
# # xlims!(-0.025,0.025)
# savefig("Figures\\Decision_boundary\\Gap_P"*string(Ptrain)*"_f0")
# savefig("Figures\\Decision_boundary\\Gap_P"*string(Ptrain)*"_f0.pdf")

## Accumulate stats in order to have meaningful histograms
M = 10
Δ0 = 0.5
Ptrain = 100 ## the larger P, the closer is the decision function f to the interface

h = 0.005 ; nh = Int(2/h) # meshgrid for testing points in order to visualise f
a = 0.2

Xtmp = Array{Array{Float64,1},2}(undef,Int(2a/h),Int(nh/2)) # [x,y] coord. of every point on the grid
for i=1:Int(2a/h) , j=1:Int(nh/2)
    Xtmp[i,j] = [h*i - a + Δ0/2,h*j - 1/2] ## one only constructs points such that -a < x < a and -1/2 < y < 1/2
end
Xtest = Array{Float64,2}(undef,2,length(Xtmp)) # necessary format to be accepted by Kernel_Matrix(.,.)
for i in eachindex(Xtmp)  Xtest[:,i] = Xtmp[i] end

f0 = NaN*zeros(M,nh)
for m in 1:M
    println("$m/$M")
    tmp = 2(rand(2,Ptrain) .- 0.5) ; Xtrain,Ytrain = tmp,generate_Y(tmp,Δ0) # unif on 2D plane [-1,1]²
    clf = SV.SVC(C=1E10,kernel="precomputed",cache_size=1000)
    clf.fit(Kernel_Matrix(Xtrain,Xtrain), Ytrain)
    f = clf.decision_function(Kernel_Matrix(Xtrain,Xtest))

    fshaped = reshape(f,Int(2a/h),nh)
    for j in 1:nh
        f0[m,j] = extrapolate_root(fshaped[:,j],h,Δ0/2-a)
    end
end

f0c = abs.(f0 .- mean(f0))
histogram(vec(f0c),label="")
savefig("Figures\\Decision_boundary\\hist_fluct")
