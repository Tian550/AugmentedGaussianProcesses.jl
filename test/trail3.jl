using AugmentedGaussianProcesses; const AGP = AugmentedGaussianProcesses
using Distributions;
using Plots;
using LinearAlgebra;
using StatsBase
using StatsFuns

using CSV
data = CSV.read("C:\\Users\\Dell\\.julia\\dev\\AugmentedGaussianProcesses\\test\\glass.csv")
y = convert(Vector{Float64},data[11])
X = convert(Array{Float64},data[1:10])

kernel1 = RBFKernel(1.0,variance=2.0)
kernel2 = RBFKernel(1.0,variance=10.0)

m1 = SVGP(X,y,kernel1,LogisticHeavisideLikelihood(),AnalyticSVI(20),20,verbose=3,optimizer=false)
m2 = SVGP(X,y,kernel2,LogisticSoftMaxLikelihood(),AnalyticSVI(20),20,verbose=3,optimizer=false)

elbo_t1 = []
function cb1(model,iter)
    # @info "Iter $iter" ELBO=ELBO(model) loglike=AGP.expecLogLikelihood(model) gausskl=AGP.GaussianKL(model) pgkl=AGP.PolyaGammaKL(model)
    push!(elbo_t1,ELBO(model))
end

elbo_t2 = []
function cb2(model,iter)
    # @info "Iter $iter" ELBO=ELBO(model) loglike=AGP.expecLogLikelihood(model) gausskl=AGP.GaussianKL(model) pgkl=AGP.PolyaGammaKL(model)
    push!(elbo_t2,ELBO(model))
end

@time train!(m1, iterations=100, callback=cb1)
@time train!(m2, iterations=100, callback=cb2)

println("Augmented accuracy = : $(mean(predict_y(m1,X).==y))")
println("Augmented accuracy = : $(mean(predict_y(m2,X).==y))")

plot(Float64.(elbo_t1),xaxis = "iterations", yaxis="ELBO Value",lab="ELBO1")
plot(Float64.(elbo_t2),xaxis = "iterations", yaxis="ELBO Value",lab="ELBO2")
