using AugmentedGaussianProcesses; const AGP = AugmentedGaussianProcesses
using Distributions;
using Plots;
using LinearAlgebra;
using StatsBase
using StatsFuns
pyplot()

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

X,y = generate_mixture_data(σs[3])

kernel = RBFKernel(1.0,variance=2.0)
amodel = VGP(X, y, kernel,LogisticHeavisideLikelihood(),AnalyticVI(),verbose=3,optimizer=false)
@time train!(amodel, iterations = 200)

gmodel = VGP(X,y,kernel,LogisticHeavisideLikelihood(),GibbsSampling(),verbose=3,optimizer=false)
@time train!(gmodel, iterations = 200)

models = [:g,:a]#,:g2]
labels = Dict(:g=>"Gibbs",:a=>"Augmented VI")

y_a = proba_y(amodel,X)
μ_a,σ_a = predict_f(amodel,X,covf=true)
global y_g = proba_y(gmodel,X)
global μ_g,σ_g = predict_f(gmodel,X,covf=true)

function adaptlims(p,plims)
    x = xlims(p)
    y = ylims(p)
    plims[1] = min(min(x[1],y[1]),plims[1])
    plims[2] = max(max(x[2],y[2]),plims[2])
    xlims!(p,(plims[1],plims[2]))
    ylims!(p,(plims[1],plims[2]))
end
mulims = [Inf,-Inf]
siglims = [Inf,-Inf]

ps = []
testi = 4
markerarguments = (:auto,1.0,0.5,:black,stroke(0))

# log.(vcat(eval(Symbol("σ_a"))...))

using LaTeXStrings

p_y = Plots.plot(vec(Matrix(eval(Symbol("y_g")))),vec(Matrix(eval(Symbol("y_a")))), t="scatter", title="p",lab="" ,xaxis=("Gibbs",(0,1),font(15)),
        yaxis=("Augmented VI",(0,1),font(15)),marker = markerarguments)
# Plots.plot!(p_y,tickfontsize=10)
p_μ = Plots.plot(vcat(eval(Symbol("μ_g"))...),vcat(eval(Symbol("μ_a"))...),t=:scatter,lab="",title="μ",xaxis=("Gibbs",(-4,5),font(15)),
        yaxis=("Augmented VI",(-4,4),font(15)),marker=markerarguments)

p_σ = Plots.plot(log.(vcat(eval(Symbol("σ_g"))...)),log.(vcat(eval(Symbol("σ_a"))...)),t=:scatter,lab="",title="log σ²",xaxis=("Gibbs",(-5,1),font(15)),
        yaxis=("Augmented VI",(-5,1),font(15)),marker=markerarguments)


plot!(p_y, x->x,-10:10,color=:red,lab="",xlims=(0,1),ylims=(0,1))
plot!(p_μ, x->x,-10:10,color=:red,lab="",xlims=(-4,5),ylims=(-4,4))
plot!(p_σ, x->x,-10:10,color=:red,lab="",xlims=(-5,1),ylims=(-5,1))


cd(@__DIR__)
savefig(p_y,"expresult\\p_y.png")
savefig(p_μ,"expresult\\p_mu.png")
savefig(p_σ,"expresult\\p_sigma.png")




num_ind = [5, 10, 20, 40, 80, 160]

tt = [1.676, 1.977, 2.528, 4.531, 8.374, 14.304]

pred_acc = [46.1, 50.6, 53.4, 70.8, 78.0, 98.9]



tt = plot(num_ind, tt,
     xlabel = "# of inducing",
     ylabel = "training time",
     lab = "time",
     xtickfont=font(18),
     ytickfont=font(18),
     guidefont=font(18),
     legendfont=font(12))

pa = plot(num_ind, pred_acc,
     xlabel = "# of inducing",
     ylabel = "predictive accuracy",
     lab = "accuracy",
     xtickfont=font(18),
     ytickfont=font(18),
     guidefont=font(18),
     legendfont=font(12))

savefig(tt, "expresult\\tt.png")
savefig(pa, "expresult\\pa.png")
