"""
**Laplace likelihood**

Laplace likelihood for regression: ``\\frac{1}{2\\sigma}\\exp\\left(-\\frac{|y-f|}{\\sigma}``
see [wiki page](https://en.wikipedia.org/wiki/Laplace_distribution)

---

For the analytical solution, it is augmented via:
```math
#TODO
```

"""
struct LaplaceLikelihood{T<:Real} <: RegressionLikelihood{T}
    β::T
    λ::LatentArray{Vector{T}}
    θ::LatentArray{Vector{T}}
    function LaplaceLikelihood{T}(β::T) where {T<:Real}
        new{T}(β)
    end
    function LaplaceLikelihood{T}(β::T,λ::AbstractVector{<:AbstractVector{T}},θ::AbstractVector{<:AbstractVector{T}}) where {T<:Real}
        new{T}(β,λ,θ)
    end
end

function LaplaceLikelihood(β::T) where {T<:Real}
    LaplaceLikelihood{T}(β)
end

function init_likelihood(β::T,likelihood::LaplaceLikelihood{T},inference::Inference{T},nLatent::Int,nSamplesUsed::Int) where T
    if inference isa AnalyticVI || inference isa GibbsSampling
        LaplaceLikelihood{T}(β,[abs2.(T.(rand(T,nSamplesUsed))) for _ in 1:nLatent],[zeros(T,nSamplesUsed) for _ in 1:nLatent])
    else
        LaplaceLikelihood{T}(β)
    end
end

function pdf(l::LaplaceLikelihood,y::Real,f::Real)
    pdf(Laplace(f,l.β),y)
end

function Base.show(io::IO,model::LaplaceLikelihood{T}) where T
    print(io,"Laplace likelihood")
end

function compute_proba(l::LaplaceLikelihood{T},μ::AbstractVector{T},β²::AbstractVector{T}) where {T<:Real}
    N = length(μ)
    st = TDist(l.ν)
    nSamples = 2000
    μ_pred = zeros(T,N)
    β²_pred = zeros(T,N)
    temp_array = zeros(T,nSamples)
    for i in 1:N
        # e = expectation(Normal(μ[i],sqrt(β²[i])))
        # μ_pred[i] = μ[i]
        #
        # β²_pred[i] = e(x->pdf(LocationScale(x,1.0,st))^2) - e(x->pdf(LocationScale(x,1.0,st)))^2
        if β²[i] <= 1e-3
            pyf =  LocationScale(μ[i],1.0,st)
            for j in 1:nSamples
                temp_array[j] = rand(pyf)
            end
        else
            d = Normal(μ[i],sqrt(β²[i]))
            for j in 1:nSamples
                temp_array[j] = rand(LocationScale(rand(d),1.0,st))
            end
        end
        μ_pred[i] = μ[i];
        β²_pred[i] = cov(temp_array)
    end
    return μ_pred,β²_pred
end

###############################################################################

function local_updates!(model::VGP{<:LaplaceLikelihood,<:AnalyticVI})
    model.likelihood.β .= broadcast((Σ,μ,y)->0.5*(Σ+abs2.(μ-y).+model.likelihood.ν),diag.(model.Σ),model.μ,model.y)
    model.likelihood.θ .= broadcast(β->0.5*(model.likelihood.ν+1.0)./β,model.likelihood.β)
end

function local_updates!(model::SVGP{<:LaplaceLikelihood,<:AnalyticVI})
    model.likelihood.β .= broadcast((K̃,κ,Σ,μ,y)->0.5*(K̃ + opt_diag(κ*Σ,κ) + abs2.(κ*μ-y[model.inference.MBIndices]) .+model.likelihood.ν),model.K̃,model.κ,model.Σ,model.μ,model.y)
    model.likelihood.θ .= broadcast(β->0.5*(model.likelihood.ν+1.0)./β,model.likelihood.β)
end

""" Return the gradient of the expectation for latent GP `index` """
function expec_μ(model::VGP{<:LaplaceLikelihood,<:AnalyticVI},index::Integer)
    return model.likelihood.θ[index].*model.y[index]
end

function ∇μ(model::VGP{<:LaplaceLikelihood,<:AnalyticVI})
    return hadamard.(model.likelihood.θ,model.y)
end

""" Return the gradient of the expectation for latent GP `index` """
function expec_μ(model::SVGP{<:LaplaceLikelihood,<:AnalyticVI},index::Integer)
    return model.likelihood.θ[index].*model.y[index][model.inference.MBIndices]
end

function ∇μ(model::SVGP{<:LaplaceLikelihood,<:AnalyticVI})
    return hadamard.(model.likelihood.θ,getindex.(model.y,[model.inference.MBIndices]))
end

function expec_Σ(model::AbstractGP{<:LaplaceLikelihood,<:AnalyticVI},index::Integer)
    return 0.5*model.likelihood.θ[index]
end

function ∇Σ(model::AbstractGP{<:LaplaceLikelihood,<:AnalyticVI})
    return 0.5*model.likelihood.θ
end

function ELBO(model::AbstractGP{<:LaplaceLikelihood,<:AnalyticVI})
    return expecLogLikelihood(model) - InverseGammaKL(model) - GaussianKL(model)
end

function expecLogLikelihood(model::VGP{LaplaceLikelihood{T},AnalyticVI{T}}) where T
    tot = -0.5*model.nLatent*model.nSample*log(twoπ)
    tot -= 0.5.*sum(broadcast(β->sum(log.(β).-model.nSample*digamma(model.likelihood.α)),model.likelihood.β))
    tot -= 0.5.*sum(broadcast((β,Σ,μ,y)->dot(model.likelihood.α./β,Σ+abs2.(μ)-2.0*μ.*y-abs2.(y)),model.likelihood.β,diag.(model.Σ),model.μ,model.y))
    return tot
end

function expecLogLikelihood(model::SVGP{LaplaceLikelihood{T},AnalyticVI{T}}) where T
    tot = -0.5*model.nLatent*model.inference.nSamplesUsed*log(twoπ)
    tot -= 0.5.*sum(broadcast(β->sum(log.(β).-model.inference.nSamplesUsed*digamma(model.likelihood.α)),model.likelihood.β))
    tot -= 0.5.*sum(broadcast((β,K̃,κ,Σ,μ,y)->dot(model.likelihood.α./β,(K̃+opt_diag(κ*Σ,κ)+abs2.(κ*μ)-2.0*(κ*μ).*y[model.inference.MBIndices]-abs2.(y[model.inference.MBIndices]))),model.likelihood.β,model.K̃,model.κ,model.Σ,model.μ,model.y))
    return model.inference.ρ*tot
end

function gradpdf(::LaplaceLikelihood,y::Int,f::T) where {T<:Real}
    @error "Not implemented yet"
end

function hessiandiagpdf(::LaplaceLikelihood,y::Int,f::T) where {T<:Real}
    @error "Not implemented yet"
end