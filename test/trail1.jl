using AugmentedGaussianProcesses; const AGP = AugmentedGaussianProcesses
using Distributions;
using Plots;
using LinearAlgebra;
using StatsBase
using StatsFuns

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

p = plot([plotdata(generate_mixture_data(σ)...,σ) for σ in σs]...)
# savefig(p,"myplot.png")
models = Vector{AbstractGP}(undef,length(σs))
kernel = RBFKernel(1.0,variance=2.0)

num_inducing = 50
X,y = generate_mixture_data(σs[5])

elbo_t = []
function cb(model,iter)
    # @info "Iter $iter" ELBO=ELBO(model) loglike=AGP.expecLogLikelihood(model) gausskl=AGP.GaussianKL(model) pgkl=AGP.PolyaGammaKL(model)
    push!(elbo_t,ELBO(model))
end

function plotdata1(X,Y,title)
    p = Plots.plot(size(300,500),lab="",title="$title")
    ys = unique(Y)
    cols = [:red,:blue,:green]
    for y in ys
        Plots.plot!(X[Y.==y,1],X[Y.==y,2],color=cols[y],alpha=1.0,t=:scatter,markerstrokewidth=0.0,lab="");
    end
    return p
end;

kernel = RBFKernel(1.0,variance=2.0)
m = VGP(X, y, kernel, LogisticHeavisideLikelihood(), AnalyticVI(), verbose=3,optimizer=false)

@time train!(m, iterations = 20, callback = cb)
y_pred = predict_y(m,X)
SVGP_p1 = plotdata1(X,y_pred,"predictive result")
SVGP_p2 = plotdata1(X,y,"original data")

SVGP_p1 = plot(SVGP_p1,xlabel="x1",
        ylabel="x2",
        xtickfont=font(18),
        ytickfont=font(18),
        guidefont=font(18),
        legendfont=font(18))

SVGP_p2 = plot(SVGP_p2,xlabel="x1",
        ylabel="x2",
        xtickfont=font(18),
        ytickfont=font(18),
        guidefont=font(18),
        legendfont=font(18))

savefig(SVGP_p1, "expresult\\SVGP_p1")
savefig(SVGP_p2, "expresult\\SVGP_p2")

VGP_ELBO = plot(elbo_t, title = "ELBO monitor",
            xlabel="iteration",
            ylabel="ELBO value",
            lab = "ELBO trend",
            legend = :bottomright,
            xtickfont=font(18),
            ytickfont=font(18),
            guidefont=font(18),
            legendfont=font(12))

VGP_ELBO

savefig(SVGP_ELBO, "expresult\\SVGP_ELBO")



AGP.expecLogLikelihood(m)

VGP_ELBO = plot(Float64.(elbo_t),xaxis = "iterations", yaxis="ELBO Value",lab="ELBO")
y_pred = predict_y(m,X)
y_prob = proba_y(m,X)
ys = Matrix(y_prob)
sum(ys,dims=2)
y_prob[!,2]
scatter(eachcol(X)...,zcolor=y_prob[!,2])
VGP_p1 = plotdata1(X,y_pred,"predictive result")
VGP_p2 = plotdata1(X,y,"original data")
VGP_combined = plot(VGP_p1,VGP_p2)

savefig(VGP_ELBO,"VGP_ELBO.png")
savefig(VGP_combined,"VGP_combined.png")

include("metrics.jl")

a,b,c,d,e = calibration(y,y_prob,plothist=true, plotline=true, plotconf=true)

# kernel_alsm = RBFKernel(1.0,variance=10.0)
kernel = AGP.RBFKernel(1.0,variance=2.0)

alsmmodel = VGP(X,y,kernel_alsm,LogisticSoftMaxLikelihood(),AnalyticVI(),verbose=2,optimizer=false)

## Aug. Logistic Heaviside
alhsmodel = SVGP(X,y,kernel,LogisticHeavisideLikelihood(),AnalyticVI(),num_inducing,verbose=2,optimizer=false)

@time train!(alhsmodel,iterations=200)

y_alhs = proba_y(alhsmodel,X)
μ_alhs,σ_alhs = predict_f(alhsmodel,X,covf=true)
y_alhs1 = compute_proba(alhsmodel.likelihood,μ_alhs,σ_alhs)

println("Augmented accuracy = : $(mean(predict_y(alhsmodel,X).==y))")

## GibbsSampling
gmodel = VGP(X,y,kernel,LogisticHeavisideLikelihood(),GibbsSampling(),verbose=2,optimizer=false)
train!(gmodel,iterations=200)

global y_g = proba_y(gmodel,X)
global μ_g,σ_g = predict_f(gmodel,X,covf=true)

println("Gibbs accuracy = : $(mean(predict_y(gmodel,X).==y))")

models = [:g,:a]
labels = Dict(:g=>"Gibbs",:a=>"Augmented VI",:e=>"VI",:g2=>"Variational Gibbs")

ps = []
testi = 4
markerarguments = (:auto,1.0,0.5,:black,stroke(0))

global y_a = proba_y(alhsmodel,X)
global μ_a,σ_a = predict_f(alhsmodel,X,covf=true)
global y_g = proba_y(gmodel,X)
global μ_g,σ_g = predict_f(gmodel,X,covf=true)

p_y = Plots.plot(vec(Matrix(eval(Symbol("y_g")))),vec(Matrix(eval(Symbol("y_a")))), t="scatter", title="p", xaxis=("Gibbs",(0,1),font(15)),
        yaxis=("Augmented VI",(0,1),font(15)),marker = markerarguments)
# Plots.plot!(p_y,tickfontsize=10)
p_μ = Plots.plot(vcat(eval(Symbol("μ_g"))...),vcat(eval(Symbol("μ_a"))...),t=:scatter,lab="",title="mean",xaxis=("Gibbs",(-4,6),font(15)),
        yaxis=("Augmented VI",(-4,6),font(15)),marker=markerarguments)

p_σ = Plots.plot(vcat(eval(Symbol("σ_g"))...),vcat(eval(Symbol("σ_a"))...),t=:scatter,lab="",title="variance",xaxis=("Gibbs",(0,3),font(15)),
        yaxis=("Augmented VI",(0,3),font(15)),marker=markerarguments)




using DelimitedFiles

writedlm("X.txt", X, ", ")
writedlm("y.txt", y, ", ")


elbo_t = []
function cb(model,iter)
    # @info "Iter $iter" ELBO=ELBO(model) loglike=AGP.expecLogLikelihood(model) gausskl=AGP.GaussianKL(model) pgkl=AGP.PolyaGammaKL(model)
    push!(elbo_t,ELBO(model))
end
num_inducing=50
m = SVGP(X,y,kernel,LogisticHeavisideLikelihood(),AnalyticSVI(50),num_inducing,verbose=3,optimizer=false)
@time train!(m, iterations=20, callback=cb)

function plotdata1(X,Y,title)
    p = Plots.plot(size(300,500),lab="",title="$title")
    ys = unique(Y)
    cols = [:red,:blue,:green]
    for y in ys
        Plots.plot!(X[Y.==y,1],X[Y.==y,2],color=cols[y],alpha=1.0,t=:scatter,markerstrokewidth=0.0,lab="");
    end
    return p
end;

using CSV
data = CSV.read("C:\\Users\\Tian550\\.julia\\dev\\AugmentedGaussianProcesses\\test\\csv.csv")
y = convert(Vector,data[1])
X = convert(Array,data[2:14])

X,y = generate_mixture_data(σs[6])
kernel = RBFKernel(1.0, variance=2.0)


## Aug.LogisticHeaviside
alhmodel = VGP(X,y,kernel,LogisticHeavisideLikelihood(),AnalyticVI(),verbose=3,optimizer=false)
@time train!(alhmodel, iterations=10)
y_alhs = proba_y(alhmodel,X)
μ_alhs,σ_alhs = predict_f(alhmodel,X,covf=true)
y_alhs1 = AGP.compute_proba(alhmodel.likelihood,μ_alhs,σ_alhs)

## Gibbs Sampler
gmodel = VGP(X,y,kernel,LogisticHeavisideLikelihood(),GibbsSampling(),verbose=3,optimizer=false)
@time train!(gmodel, iterations=400)

global y_g = proba_y(gmodel,X)
global μ_g,σ_g = predict_f(gmodel,X,covf=true)

println("Augmented accuracy = : $(mean(predict_y(alhmodel,X).==y))")
println("Gibbs accuracy = : $(mean(predict_y(gmodel,X).==y))")

## histogram
include("metrics.jl")

a,b,c,d,e = calibration(y,y_alhs,plothist=true, plotline=true, plotconf=true)

using MLDatasets

x_train, y_train = MNIST.traindata()
x_test, y_test = MNIST.testdata()
x_train = Array{Float64}(x_train)
x_train = reshape(x_train,(60000,28^2))
x_train[1,:]
x_test = reshape(Array{Float64}(x_test),:,28^2)
m = SVGP(x_train,y_train,RBFKernel(),LogisticHeavisideLikelihood(),AnalyticSVI(100),100,verbose=3,optimizer=false)
train!(m,iterations=10000)
predict_y(m,x_test)
y_test

println("Augmented accuracy = : $(mean(predict_y(m,x_train).==y_train))")

num_inducing = 160
model = SVGP(X,y,kernel,LogisticHeavisideLikelihood(),AnalyticSVI(30),num_inducing,verbose=2,optimizer=false)
@time train!(model, iterations=100)
println("Augmented accuracy = : $(mean(predict_y(model,X).==y))")




X,y = generate_mixture_data(σs[3])
kernel = RBFKernel(1.0, variance=2.0)

alhmodel = VGP(X,y,kernel,LogisticHeavisideLikelihood(),AnalyticVI(),verbose=3,optimizer=false)
@time train!(alhmodel, iterations=20)
y_alhs = proba_y(alhmodel,X)
μ_alhs,σ_alhs = predict_f(alhmodel,X,covf=true)
a,b,c,d,e = calibration(y,y_alhs,plothist=true, plotline=true, plotconf=true)

gmodel = VGP(X,y,kernel,LogisticHeavisideLikelihood(),GibbsSampling(),verbose=3,optimizer=false)
@time train!(gmodel, iterations=200)

c = Plots.plot(c, xlabel="Confidence", ylabel="Accuracy")
d = Plots.plot(d, xlabel="Confidence", ylabel="Accuracy")
e = Plots.plot(e, xlabel="Confidence", ylabel="% of Samples")

savefig(c,"6_linear.png")
savefig(d,"6_histogram.png")
savefig(e,"6_reliab.png")

markerarguments = (:auto,1.0,0.5,:black,stroke(0))

global y_a = proba_y(alhmodel,X)
global μ_a,σ_a = predict_f(alhmodel,X,covf=true)
global y_g = proba_y(gmodel,X)
global μ_g,σ_g = predict_f(gmodel,X,covf=true)

p_y = Plots.plot(vec(Matrix(eval(Symbol("y_g")))),vec(Matrix(eval(Symbol("y_a")))), t="scatter", title="p", xaxis=("Gibbs",(0,1)),
        yaxis=("Augmented VI",(0,1)),marker = markerarguments)
# Plots.plot!(p_y,tickfontsize=10)
p_μ = Plots.plot(vcat(eval(Symbol("μ_g"))...),vcat(eval(Symbol("μ_a"))...),t=:scatter,lab="",title="mean",xaxis=("Gibbs",(-5,5)),
        yaxis=("Augmented VI",(-5,5)),marker=markerarguments)

p_σ = Plots.plot(vcat(eval(Symbol("σ_g"))...),vcat(eval(Symbol("σ_a"))...),t=:scatter,lab="",title="variance",xaxis=("Gibbs",(0,3)),
        yaxis=("Augmented VI",(0,3)),marker=markerarguments)


p1 = Plots.plot(p_y, title="p",xlabel="Gibbs",ylabel="Augented VI")
p2 = Plots.plot(p_μ, title="mean")
p3 = Plots.plot(p_σ, title="variance")

savefig(p1, "p.png")
savefig(p2, "mu.png")
savefig(p3, "sigma.png")
