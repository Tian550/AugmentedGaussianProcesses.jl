using AugmentedGaussianProcesses
using Distributions;
using Plots;
using LinearAlgebra;
using StatsBase

# Generate data from a mixture of gaussians (you can control the noise)
N_data = 500; N_dim = 2; N_grid = 100
minx=-2.5; maxx=3.5
σs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.8]; N_class = N_dim + 1

function generate_mixture_data(σ)
    centers = zeros(N_class,N_dim)
    for i in 1:N_dim
        centers[i,i] = 1
    end
    centers[end,:] = (1+sqrt(N_class))/N_dim*ones(N_dim); centers ./= sqrt(N_dim)
    ## Generate distributions with desired noise
    distr = [MvNormal(centers[i,:],σ) for i in 1:N_class]
    X = zeros(Float64,N_data,N_dim)
    y = zeros(Int64,N_data)
    true_py = zeros(Float64,N_data)
    for i in eachindex(y)
        y[i] = rand(1:N_class)
        X[i,:] = rand(distr[y[i]])
        true_py[i] = pdf(distr[y[i]],X[i,:])/sum(pdf(distr[k],X[i,:]) for k in 1:N_class)
    end
    return X,y
end
## Create equidistant centers

function plotdata(X,Y,σ)
    p = Plots.plot(size(300,500),lab="",title="sigma = $σ")
    ys = unique(Y)
    cols = [:red,:blue,:green]
    for y in ys
        Plots.plot!(X[Y.==y,1],X[Y.==y,2],color=cols[y],alpha=1.0,t=:scatter,markerstrokewidth=0.0,lab="");
    end
    return p
end;

plot([plotdata(generate_mixture_data(σ)...,σ) for σ in σs]...)

models = Vector{AbstractGP}(undef,length(σs))
kernel = RBFKernel(1.0,variance=2.0)

num_inducing = 50
X,y = generate_mixture_data(σs[1])

# m = VGP(X, y, kernel, LogisticHeavisideLikelihood(), AnalyticVI())
# println("Training with data with noise 0.1")
# @time train!(m, iterations = 5)
#


# m = SVGP(X, y, kernel,LogisticSoftMaxLikelihood(),AnalyticVI(),num_inducing,optimizer=false)
# @time train!(m,iterations=20)

# y_pred = predict_y(m,X)
# p1 = plotdata(X,y_pred,0.0)
# p2 = plotdata(X,y,0.8)
# plot(p1,p2)
# model = m
function cb(model,iter)
    @show iter
end

m = SVGP(X, y, kernel,LogisticHeavisideLikelihood(),AnalyticSVI(10),num_inducing,verbose=3,optimizer=false)

@time train!(m, iterations = 20,callback=cb)

y_pred = predict_y(m,X)
p1 = plotdata(X,y_pred,0.0)
p2 = plotdata(X,y,0.8)
plot(p1,p2)
