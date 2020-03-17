using AugmentedGaussianProcesses; const AGP = AugmentedGaussianProcesses
using Distributions;
using Plots;
using LinearAlgebra;
using StatsBase
using StatsFuns
using RCall

include("metrics.jl")
cd(@__DIR__)
R"source('sepMGPC_batch.R')"

# Generate data from a mixture of gaussians (you can control the noise)
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

kernel = RBFKernel(1.0,variance=2.0)

num_inducing = 50
X,y = generate_mixture_data(σs[4])

alhmodel = SVGP(X, y, kernel,LogisticHeavisideLikelihood(),AnalyticVI(), num_inducing, verbose=3, optimizer=false)
@time train!(alhmodel, iterations = 200)

Z = copy(alhmodel.Z)
t_ep = @elapsed epmodel = R"epMGPCInternal($X, $(y),$(size(Z[1],1)),  X_test = $X, Y_test= $(y),  max_iters=200, indpoints= FALSE, autotuning=FALSE)"
global py_ep = Matrix(rcopy(R"predictMGPC($(epmodel),$(X))$prob"))

ECE_ep, MCE_ep, cal_ep, calh_ep, conf_ep= calibration(y,py_ep,plothist=true,plotline=true,plotconf=true,gpflow=true)

calh_ep


elbo_t = []
function cb(model,iter)
    # @info "Iter $iter" ELBO=ELBO(model) loglike=AGP.expecLogLikelihood(model) gausskl=AGP.GaussianKL(model) pgkl=AGP.PolyaGammaKL(model)
    push!(elbo_t,ELBO(model))
end

kernel = RBFKernel(1.0,variance=2.0)
alhmodel = SVGP(X, y, kernel,LogisticHeavisideLikelihood(),AnalyticVI(), num_inducing, verbose=3,optimizer=false)

Z = copy(alhmodel.Z)
@elapsed train!(alhmodel,iterations=200)

global py_alh = proba_y(alhmodel,X)

ECE_alh, MCE_alh, cal_alh, calh_alh, conf_alh =calibration(y,py_alh,plothist=true,plotline=true,plotconf=true,meanonly=true,threshold=2)

conf_alh

conf_alsm = plot(conf_alsm, xlabel="Confidence",
    ylabel="% of samples",
    xtickfont=font(18),
    ytickfont=font(18),
    guidefont=font(18),
    legendfont=font(12))

savefig(conf_alsm,"expresult\\alsm_conf_04.png")

nBins=10
edges = collect(range(0.0,1.0,length=nBins+1))
global mean_bins = 0.5*(edges[2:end]+edges[1:end-1])
