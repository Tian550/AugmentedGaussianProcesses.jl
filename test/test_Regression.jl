using Distributions
using AugmentedGaussianProcesses
using LinearAlgebra
using Random: seed!
seed!(42)
doPlot=!false
if !@isdefined doPlots
    doPlots = true
end
if !@isdefined verbose
    verbose = 3
end
if doPlots
    using Plots
    pyplot()
end
N_data = 200
N_test = 20
N_dim = 2
noise = 0.2
minx=-5.0
maxx=5.0
function latent(x)
    # return sin.(0.5*x[:,1].*x[:,2])
    return x[:,1].*sin.(x[:,2])
end
X = rand(N_data,N_dim)*(maxx-minx).+minx
x_test = range(minx,stop=maxx,length=N_test)
X_test = hcat([j for i in x_test, j in x_test][:],[i for i in x_test, j in x_test][:])
y = latent(X)+rand(Normal(0,noise),size(X,1))
y_test = latent(X_test)
(nSamples,nFeatures) = (N_data,1)
ps = []; t_full = 0; t_sparse = 0; t_stoch = 0;

kernel = RBFKernel(1.5)
autotuning=!false
optindpoints=true
fullm=!true
sparsem=true
stochm=!true
println("Testing the regression model")
if fullm
    println("Testing the full model")
    t_full = @elapsed fullmodel = VGP(X,y,kernel,GaussianLikelihood(noise),AnalyticInference(),Autotuning=autotuning,verbose=verbose)
    t_full += @elapsed train!(fullmodel,iterations=50)
    y_full = predict_y(fullmodel,X_test,covf=false); rmse_full = norm(y_full[1]-y_test,2)/sqrt(length(y_test))
    if doPlots
        p1=plot(x_test,x_test,reshape(y_full[1],N_test,N_test),t=:contour,fill=true,cbar=false,clims=[-5,5],lab="",title="Regression")
        push!(ps,p1)
    end
end

if sparsem
    println("Testing the sparse model")
    t_sparse = @elapsed sparsemodel = SVGP(X,y,kernel,GaussianLikelihood(noise),AnalyticInference(),20,Stochastic=false,Autotuning=autotuning,verbose=verbose)
    t_sparse += @elapsed train!(sparsemodel,iterations=100)
    y_sparse = predict_y(sparsemodel,X_test,covf=false); rmse_sparse = norm(y_sparse[1]-y_test,2)/sqrt(length(y_test))
    if doPlots
        p2=plot(x_test,x_test,reshape(y_sparse[1],N_test,N_test),t=:contour,fill=true,cbar=false,clims=[-5,5],lab="",title="Sparse Regression")
        plot!(sparsemodel.Z[1][:,1],sparsemodel.Z[1][:,2],t=:scatter,lab="inducing points")
        push!(ps,p2)
    end
end

if stochm
    println("Testing the sparse stochastic model")
    t_stoch = @elapsed stochmodel = AugmentedGaussianProcesses.SparseGPRegression(X,y,Stochastic=true,batchsize=20,Autotuning=autotuning,verbose=verbose,m=20,noise=noise,kernel=kernel,OptimizeIndPoints=optindpoints)
    t_stoch += @elapsed stochmodel.train(iterations=1000)
    y_stoch = stochmodel.predict(X_test); rmse_stoch = norm(y_stoch-y_test,2)/sqrt(length(y_test))
    if doPlots
        p3=plot(x_test,x_test,reshape(y_stoch,N_test,N_test),t=:contour,fill=true,cbar=true,clims=(minx*1.1,maxx*1.1),lab="",title="Stoch. Sparse Regression")
        plot!(stochmodel.inducingPoints[:,1],stochmodel.inducingPoints[:,2],t=:scatter,lab="inducing points")
        push!(ps,p3)
    end
end
t_full != 0 ? println("Full model : RMSE=$(rmse_full), time=$t_full") : nothing
t_sparse != 0 ? println("Sparse model : RMSE=$(rmse_sparse), time=$t_sparse") : nothing
t_stoch != 0 ? println("Stoch. Sparse model : RMSE=$(rmse_stoch), time=$t_stoch") : nothing

if doPlots
    ptrue=plot(x_test,x_test,reshape(latent(X_test),N_test,N_test),t=:contour,fill=true,cbar=false,clims=[-5,5],lab="")
    plot!(ptrue,X[:,1],X[:,2],t=:scatter,lab="training points",title="Truth")
    display(plot(ptrue,ps...))
end

return true
