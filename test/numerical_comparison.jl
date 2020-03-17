using AugmentedGaussianProcesses; const AGP = AugmentedGaussianProcesses
using Distributions;
using Plots;
using LinearAlgebra;
using StatsBase
using StatsFuns
using MLDataUtils
using RCall
using CSV
cd(@__DIR__)
R"source('sepMGPC_batch.R')"

data = CSV.read("C:\\Users\\Dell\\.julia\\dev\\AugmentedGaussianProcesses\\test\\balance-scale.csv")
y = convert(Vector{Int64},data[1])
X = convert(Array{Float64},data[2:5])
# t_ep = @elapsed epmodel = R"epMGPCInternal($X, $(y),$(size(Z[1],1)), X_test = $X, Y_test= $(y),  max_iters=200, indpoints= FALSE, autotuning=FALSE)"

kernel1 = RBFKernel(1.0,variance=2.0)
kernel2 = RBFKernel(1.0,variance=10.0)

k = 10  ## k-folds

time_ep = []
time_alh = []
time_alsm = []

NLL_ep = []
NLL_alh = []
NLL_alsm = []

Acc_ep = []
Acc_alh = []
Acc_alsm = []

for ((X_training,y_training),(X_test,y_test)) in kfolds((X,y),obsdim=1,k=k)

    alhmodel = SVGP(X_training, y_training, kernel1, LogisticHeavisideLikelihood(), AnalyticVI(), 20, verbose=3, optimizer=true)
    t_alh = @elapsed train!(alhmodel, iterations = 200)
    Z = copy(alhmodel.Z)
    p1 = convert(Array{Float64}, proba_y(alhmodel, X_test))
    p1 = [p1[i,y_test[i]] for i in 1:size(p1)[1]]
    nll_alh = mean(-log.(p1))
    acc_alh = mean(predict_y(alhmodel,X_test).==y_test)
    push!(time_alh, t_alh)
    push!(NLL_alh, nll_alh)
    push!(Acc_alh, acc_alh)

    alsmmodel = SVGP(X_training, y_training, kernel2, LogisticSoftMaxLikelihood(), AnalyticVI(), 20, verbose=3, optimizer=true)
    t_alsm = @elapsed train!(alsmmodel, iterations = 200)
    p2 = convert(Array{Float64}, proba_y(alsmmodel, X_test))
    p2 = [p2[i,y_test[i]] for i in 1:size(p2)[1]]
    nll_alsm = mean(-log.(p2))
    acc_alsm = mean(predict_y(alsmmodel,X_test).==y_test)
    push!(time_alsm, t_alsm)
    push!(NLL_alsm, nll_alsm)
    push!(Acc_alsm, acc_alsm)

    t_ep = @elapsed epmodel = R"epMGPCInternal($X_training, $(y_training),$(size(Z[1],1)), X_test = $X_training, Y_test= $(y_training),
     max_iters=200, indpoints= FALSE, autotuning=TRUE)"
    py_ep = Matrix(rcopy(R"predictMGPC($(epmodel),$(X_test))$prob"))
    push!(time_ep, t_ep)
    p3 = [py_ep[i,y_test[i]] for i in 1:size(py_ep)[1]]
    nll_ep = mean(-log.(p3))
    push!(NLL_ep, nll_ep)
    n = size(X_test,1)
    acc_ep = mean([argmax(py_ep[i,:]) for i in 1:n] .== y_test)
    push!(Acc_ep, acc_ep)

end

time_ep
time_alh
time_alsm

NLL_ep
NLL_alh
NLL_alsm

Acc_ep
Acc_alh
Acc_alsm

mean(Acc_alh)
std(Acc_alh)
mean(Acc_alsm)
std(Acc_alsm)
mean(Acc_ep)
std(Acc_ep)
