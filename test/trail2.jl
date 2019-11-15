using AugmentedGaussianProcesses; const AGP = AugmentedGaussianProcesses
using Distributions;
using Plots;
using LinearAlgebra;
using StatsBase
using StatsFuns
using MLDatasets

x_train, y_train = MNIST.traindata()
x_train = Array{Float64}(x_train)
x_train = reshape(x_train,(60000,28^2))

num_inducing = 500
kernel = RBFKernel(1.0, variance = 2.0)

mnistmodel = SVGP(x_train[1:1000,:], y_train[1:1000], kernel, LogisticHeavisideLikelihood(), AnalyticSVI(100), num_inducing, verbose=3, optimizer=false)
@time train!(mnistmodel, iterations = 100)

y_pred = predict_y(mnistmodel, x_train[1001:1005,:])
y_prod = proba_y(mnistmodel, x_train[1001:1005,:])

println("Augmented accuracy = : $(mean(predict_y(mnistmodel, x_train[1:1000,:]).==y_train[1:1000]))")


using CSV

data = CSV.read("C:\\Users\\Tian550\\.julia\\dev\\AugmentedGaussianProcesses\\test\\wine.csv")

X = convert(Matrix, data[:,2:13])
y = convert(Vector, data[:,1])

winemodel = SVGP(X,y,kernel,LogisticHeavisideLikelihood(),AnalyticSVI(80),80,verbose=3,optimizer=false)
@time train!(winemodel, iterations=50)
println("Augmented accuracy = : $(mean(predict_y(winemodel, X).== y))")
