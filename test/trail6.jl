using AugmentedGaussianProcesses; const AGP = AugmentedGaussianProcesses
using Distributions;
using Plots;
using LinearAlgebra;
using StatsBase
using StatsFuns
using MLDataUtils
using RCall
cd(@__DIR__)
R"source('sepMGPC_batch.R')"

N_data = 500; N_dim = 2; N_grid = 100
minx=-2.5; maxx=3.5
σs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.8];
N_class = N_dim + 1;

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

X,y = generate_mixture_data(σs[4])
kernel1 = RBFKernel(1.0,variance=2.0)
kernel2 = RBFKernel(1.0,variance=10.0)

elbo_t = []
function cb(model,iter)
    # @info "Iter $iter" ELBO=ELBO(model) loglike=AGP.expecLogLikelihood(model) gausskl=AGP.GaussianKL(model) pgkl=AGP.PolyaGammaKL(model)
    push!(elbo_t,ELBO(model))
end

# k = 10  ## k-folds
# kf_observations = kfolds(y, k = k)
# training_indices = kf_observations.train_indices
# testing_indices = kf_observations.val_indices
y

m1 = SVGP(X,y,kernel1,LogisticHeavisideLikelihood(),AnalyticSVI(10),10,verbose=3,optimizer=true)
@elapsed train!(m1, iterations=100)
m2 = SVGP(X,y,kernel2,LogisticSoftMaxLikelihood(),AnalyticSVI(10),10,verbose=3,optimizer=true)
@elapsed train!(m2, iterations=100)

m1.likelihood.ind_mapping
m1.likelihood.y_class
convert(Array{Float64}, proba_y(m1, X))
proba_y(m1,X)

function plotdata1(X,Y,title)
    p = Plots.plot(size(300,500),lab="",title="$title")
    ys = unique(Y)
    cols = [:red,:blue,:green]
    for y in ys
        Plots.plot!(X[Y.==y,1],X[Y.==y,2],color=cols[y],alpha=1.0,t=:scatter,markerstrokewidth=0.0,lab="");
    end
    return p
end;

SVGP_p1 = plotdata1(X,y_pred,"predictive result")
plot(elbo_t)

t_ep = @elapsed epmodel = R"epMGPCInternal($X, $(y),$(size(Z[1],1)), X_test = $X, Y_test= $(y),  max_iters=200, indpoints= FALSE, autotuning=TRUE)"
y
py_ep = rcopy(R"predictMGPC($(epmodel),$(X))$prob")
mean(-log.(maximum(py_ep, dims=2)))

p = proba_y(m1, X)
p = convert(Array{Float64},p)
p = [p[i,y[i]] for i in 1:size(p)[1]]
mean(-log.(p))

p2 = proba_y(m2, X)
p2 = convert(Array{Float64},p2)
p2 = [p2[i,y[i]] for i in 1:size(p2)[1]]
mean(-log.(p2))

p3 = [py_ep[i,y[i]] for i in 1:size(py_ep)[1]]
mean(-log.(p3))
