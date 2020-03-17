using AugmentedGaussianProcesses; const AGP = AugmentedGaussianProcesses
using Distributions;
using Plots;
using LinearAlgebra;
using StatsBase
using StatsFuns
using MLDataUtils
using RCall
using CSV
using MLDatasets
cd(@__DIR__)
R"source('sepMGPC_stochastic.R')"

data = CSV.read("C:\\Users\\Dell\\.julia\\dev\\AugmentedGaussianProcesses\\test\\shuttle_train.csv")
y = convert(Vector{Int64},data[10])
X = convert(Array{Float64},data[1:9])
data1 = CSV.read("C:\\Users\\Dell\\.julia\\dev\\AugmentedGaussianProcesses\\test\\shuttle_test.csv")
y_test = convert(Vector{Int64},data1[10])
X_test = convert(Array{Float64},data1[1:9])
# X = MNIST.convert2features(MNIST.traintensor())
# X = Array{Float64}(X)
# X = transpose(X)
# X = Array{Float64}(X)
# _, y =MNIST.traindata()
# y = y.+1
#
# X_test = MNIST.convert2features(MNIST.testtensor())
# X_test = Array{Float64}(X_test)
# X_test = transpose(X_test)
# X_test = Array{Float64}(X_test)
# _, y_test =MNIST.testdata()
# y_test = y_test.+1

kernel1 = RBFKernel(1.0,variance=2.0)
kernel2 = RBFKernel(1.0,variance=10.0)

alhmodel = SVGP(X, y, kernel1, LogisticHeavisideLikelihood(), AnalyticSVI(100), 100, verbose=3, optimizer=true)
t_alh = @elapsed train!(alhmodel, iterations = 200)
Z = copy(alhmodel.Z)

# y_test
# alhmodel.likelihood.ind_mapping

p1 = convert(Array{Float64}, proba_y(alhmodel, X_test))
p1 = [p1[i,alhmodel.likelihood.y_class[i]] for i in 1:size(p1)[1]]
nll_alh = mean(-log.(p1))

acc_alh = mean(predict_y(alhmodel,X_test) .== y_test)

alsmmodel = SVGP(X, y, kernel2, LogisticSoftMaxLikelihood(), AnalyticSVI(100), 100, verbose=3, optimizer=true)
t_alsm = @elapsed train!(alsmmodel, iterations = 200)
p2 = convert(Array{Float64}, proba_y(alsmmodel, X_test))
p2 = [p2[i,alsmmodel.likelihood.y_class[i]] for i in 1:size(p2)[1]]
nll_alsm = mean(-log.(p2))
acc_alsm = mean(predict_y(alsmmodel,X_test).==y_test)

function gpflowacc(y_test,y_pred)
    score = 0.0
    for i in 1:length(y_test)
        if argmax(y_pred[i,:])==y_test[i]
            score += 1
        end
    end
    return score/length(y_test)
end

function gpflowloglike(y_test,y_pred)
    score = 0.0
    for i in 1:length(y_test)
        score += log(y_pred[i,y_test[i]])
    end
    return score/length(y_test)
end

t_ep = @elapsed epmodel = R"epMGPCInternal($X, $(y),$(size(Z[1],1)), n_minibatch = 100, X_test = $X_test, Y_test= $(y_test),
 max_iters=200, indpoints= FALSE, autotuning=TRUE)"
py_ep = Matrix(rcopy(R"predictMGPC($(epmodel),$(X_test))$prob"))
# push!(time_ep, t_ep)
p3 = [py_ep[i,y_test[i]] for i in 1:size(py_ep)[1]]
nll_ep = mean(-log.(p3))
# push!(NLL_ep, nll_ep)
n = size(X_test,1)
acc_ep = mean([argmax(py_ep[i,:]) for i in 1:n] .== y_test)
# push!(Acc_ep, acc_ep)
gpflowacc(y_test,py_ep)
-gpflowloglike(y_test,py_ep)

# t_ep1 = @elapsed epmodel1 = R"epMGPCInternal($X, $(y),$(size(Z[1],1)), n_minibatch = 100, X_test = $X_test, Y_test= $(y_test),
#  max_iters=2000, indpoints= FALSE, autotuning=TRUE)"
# py_ep1 = Matrix(rcopy(R"predictMGPC($(epmodel1),$(X_test))$prob"))
# # push!(time_ep, t_ep)
# p31 = [py_ep1[i,y_test[i]] for i in 1:size(py_ep1)[1]]
# nll_ep1 = mean(-log.(p31))
# # push!(NLL_ep, nll_ep)
# n = size(X_test,1)
# acc_ep1 = mean([argmax(py_ep1[i,:]) for i in 1:n] .== y_test)
# # push!(Acc_ep, acc_ep)
# gpflowacc(y_test,py_ep1)
# -gpflowloglike(y_test,py_ep1)

-log(0.6)
