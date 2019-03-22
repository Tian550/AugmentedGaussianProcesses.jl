"""
Logistic likelihood : ``p(y|f) = σ(yf) = (1+exp(-yf))⁻¹ ``
More info on [wiki page](https://en.wikipedia.org/wiki/Logistic_function)
"""
abstract type AbstractLogisticLikelihood{T<:Real} <: ClassificationLikelihood{T} end

function pdf(l::AbstractLogisticLikelihood,y::Real,f::Real)
    logistic(y*f)
end

function Base.show(io::IO,model::AbstractLogisticLikelihood{T}) where T
    print(io,"Bernoulli likelihood with logistic link")
end


function compute_proba(l::AbstractLogisticLikelihood{T},μ::AbstractVector{T},σ²::AbstractVector{T}) where {T<:Real}
    N = length(μ)
    pred = zeros(T,N)
    for i in 1:N
        if σ²[i] <= 0.0
            pred[i] = logistic(μ[i])
        else
            pred[i] =  expectation(logistic,Normal(μ[i],sqrt(σ²[i])))
        end
    end
    return pred
end

###############################################################################

struct AugmentedLogisticLikelihood{T<:Real} <: AbstractLogisticLikelihood{T}
    c::AbstractVector{AbstractVector{T}}
    θ::AbstractVector{AbstractVector{T}}
    function AugmentedLogisticLikelihood{T}() where {T<:Real}
        new{T}()
    end
    function AugmentedLogisticLikelihood{T}(c::AbstractVector{<:AbstractVector{<:Real}},θ::AbstractVector{<:AbstractVector{<:Real}}) where {T<:Real}
        new{T}(c,θ)
    end
end

function AugmentedLogisticLikelihood()
    AugmentedLogisticLikelihood{Float64}()
end

isaugmented(::AugmentedLogisticLikelihood{T}) where T = true

function init_likelihood(likelihood::AugmentedLogisticLikelihood{T},nLatent::Integer,nSamplesUsed) where T
    AugmentedLogisticLikelihood{T}([abs.(rand(T,nSamplesUsed)) for _ in 1:nLatent],[zeros(T,nSamplesUsed) for _ in 1:nLatent])
end


function local_updates!(model::VGP{<:AugmentedLogisticLikelihood,<:AnalyticInference})
    model.likelihood.c .= broadcast((μ,Σ)->sqrt.(Σ+abs2.(μ)),model.μ,diag.(model.Σ))
    model.likelihood.θ .= broadcast(c->0.5*tanh.(0.5*c)./c,model.likelihood.c)
end

function local_updates!(model::SVGP{<:AugmentedLogisticLikelihood,<:AnalyticInference})
    model.likelihood.c .= broadcast((μ,Σ,K̃,κ)->sqrt.(K̃+opt_diag(κ*Σ,κ)+abs2.(κ*μ)),model.μ,model.Σ,model.K̃,model.κ)
    model.likelihood.θ .= broadcast(c->0.5*tanh.(0.5*c)./c,model.likelihood.c)
end

""" Return the gradient of the expectation for latent GP `index` """
function expec_μ(model::VGP{<:AugmentedLogisticLikelihood},index::Integer)
    return 0.5*model.y[index]
end

function ∇μ(model::VGP{<:AugmentedLogisticLikelihood})
    return 0.5*model.y
end

""" Return the gradient of the expectation for latent GP `index` """
function expec_μ(model::SVGP{<:AugmentedLogisticLikelihood},index::Integer)
    return 0.5.*model.y[index][model.inference.MBIndices]
end

function ∇μ(model::SVGP{<:AugmentedLogisticLikelihood})
    return 0.5.*getindex.(model.y,[model.inference.MBIndices])
end

function expec_Σ(model::AbstractGP{<:AugmentedLogisticLikelihood},index::Integer)
    return 0.5*model.likelihood.θ[index]
end

function ∇Σ(model::AbstractGP{<:AugmentedLogisticLikelihood})
    return 0.5*model.likelihood.θ
end

function ELBO(model::AbstractGP{<:AugmentedLogisticLikelihood})
    return expecLogLikelihood(model) - GaussianKL(model) - PolyaGammaKL(model)
end

function expecLogLikelihood(model::VGP{AugmentedLogisticLikelihood{T}}) where T
    tot = -model.nLatent*(0.5*model.nSample*logtwo)
    tot += sum(broadcast((μ,y,θ,Σ)->0.5.*(sum(μ.*y)-dot(θ,Σ+abs2.(μ))),
                        model.μ,model.y,model.likelihood.θ,diag.(model.Σ)))
    return tot
end

function expecLogLikelihood(model::SVGP{AugmentedLogisticLikelihood{T}}) where T
    tot = -model.nLatent*(0.5*model.inference.nSamplesUsed*logtwo)
    tot += sum(broadcast((κμ,y,θ,κΣκ,K̃)->0.5.*(sum(κμ.*y)-dot(θ,K̃+κΣκ+abs2.(κμ)))),
                        model.κ.*model.μ,model.y,model.likelihood.θ,opt_diag(model.κ.*model.Σ,model.κ'),model.K̃)
    return model.inference.ρ*tot
end

###############################################################################

struct LogisticLikelihood{T<:Real} <: AbstractLogisticLikelihood{T}
    function LogisticLikelihood{T}() where {T<:Real}
        new{T}()
    end
end

function gradpdf(::LogisticLikelihood,y::Int,f::T) where {T<:Real}
    σ=y*f
    σ*(one(T)-σ)
end

function hessiandiagpdf(::LogisticLikelihood,y::Int,f::T) where {T<:Real}
    σ=y*f
    σ*(one(T)-2σ + abs2(σ))
end