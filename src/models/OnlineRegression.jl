
#Online Gaussian Process Regression

mutable struct OnlineGPRegression <: OnlineGPModel
    @commonfields
    @functionfields
    @stochasticfields
    @kernelfields
    @gaussianparametersfields
    @onlinefields

    function OnlineGPRegression(X::AbstractArray,y::AbstractArray;kmeansalg::KMeansAlg=StreamOnline(),Sequential::Bool=false,AdaptiveLearningRate::Bool=false,
                                    Autotuning::Bool=false,optimizer::Optimizer=Adam(α=0.1),OptimizeIndPoints::Bool=false,
                                    nEpochs::Integer = 10000,BatchSize::Integer=-1,κ_s::Float64=1.0,τ_s::Integer=100,
                                    kernel=0,noise::Real=1e-3,m::Integer=0,AutotuningFrequency::Integer=2,
                                    ϵ::Real=1e-5,μ_init::Array{Float64,1}=[0.0],SmoothingWindow::Integer=5,
                                    VerboseLevel::Integer=0)
            this = new();
            this.ModelType = Regression;
            this.Name = "Online Sparse Gaussian Process Regression";
            initCommon!(this,X,y,noise,ϵ,nEpochs,VerboseLevel,Autotuning,AutotuningFrequency,optimizer);
            this.nSamplesUsed = BatchSize
            initOnline!(this,kmeansalg,Sequential,m)
            initFunctions!(this);
            initStochastic!(this,AdaptiveLearningRate,BatchSize,κ_s,τ_s,SmoothingWindow);
            initGaussian!(this,μ_init);
            initKernel!(this,kernel); this.nFeatures = this.m
            return this;
    end
end