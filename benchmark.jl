cd("C:\\Users\\Ylann Rouzaire\\.julia\\environments\\ML_env")
using Pkg; Pkg.activate("."); Pkg.instantiate();
# using Plots, SpecialFunctions, JLD, Dates,Distributed, LinearAlgebra, Distributions,MLJ
# pyplot() ; plot()
using SpecialFunctions,Distributed,LinearAlgebra,Distributions,PyCall
# @MLJ.load SVC pkg=LIBSVM
include("function_definitions.jl")

Δ0 = 0.01
Ptrain = Int(1E1)
Ptest = Int(1E6)
N = Ptrain + Ptest
dimension = 3
δ = Ptrain^(-1/dimension)

X = generate_X(Ptrain,Ptest,dimension,Δ0)
Y = generate_Y(X,Δ0)
Xtest,Ytest = extract_TestSet(X,Y,Ptest)
Xtrain,Ytrain = extract_TrainSet(X,Y,Ptrain)

# scatter([X[i][1] for i in 1:N],[X[i][2] for i in 1:N],[X[i][3] for i in 1:N],label=nothing,camera=(30,70))
# xlabel!("x")
# ylabel!("y")
# title!("Distribution with Margin Δ0 = 0.5")
# savefig("distrib_X.pdf")

## First SVM tests
svc_model = SVC(cost=1E10)
svc = machine(svc_model, MLJ.table(Xtrain), Ytrain)
fit!(svc)
Ypred = predict(svc, Xtest)
println("𝝐 = ",misclassification_rate(Ypred, Ytest))
G = Gram(Xtrain,1/2)
SVC(kernel="RadialBasis")

## pycall
using PyCall
skl = pyimport("sklearn")
X = [[0, 0], [1, 1]]
y = [0, 1]
clf = skl.svm.SVC(C=1E10,kernel="precomputed")
# clf.cache_size = 200
clf.fit(X, y)
clf.predict([[2., 2.]])
clf.support_vectors_
