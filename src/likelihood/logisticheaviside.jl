struct LogisticHeavisideLikelihood{T<:Real} <: MultiClassLikelihood{T}
    Y::AbstractVector{BitVector} #Mapping from instances to classes
    class_mapping::AbstractVector{Any} # Classes labels mapping
    ind_mapping::Dict{Any,Int} # Mapping from label to index
    y_class::AbstractVector{Int64} # GP Index for each sample
    c::AbstractVector{AbstractVector{T}} # Second moment of fₖ
#     α::AbstractVector{T} # Variational parameter of Gamma distribution
#     β::AbstractVector{T} # Variational parameter of Gamma distribution
    θ::AbstractVector{AbstractVector{T}} # Variational parameter of Polya-Gamma distribution
#     γ::AbstractVector{AbstractVector{T}} # Variational parameter of Poisson distribution
    function LogisticHeavisideLikelihood{T}() where {T<:Real}
        new{T}()
    end
    function LogisticHeavisideLikelihood{T}(Y::AbstractVector{<:BitVector},
    class_mapping::AbstractVector, ind_mapping::Dict{<:Any,<:Int},y_class::AbstractVector{<:Int}) where {T<:Real}
        new{T}(Y,class_mapping,ind_mapping,y_class)
    end
    function LogisticHeavisideLikelihood{T}(Y::AbstractVector{<:BitVector},
    class_mapping::AbstractVector, ind_mapping::Dict{<:Any,<:Int},y_class::AbstractVector{<:Int},
    c::AbstractVector{<:AbstractVector{<:Real}},  θ::AbstractVector{<:AbstractVector}) where {T<:Real}
        new{T}(Y,class_mapping,ind_mapping,y_class,c,θ)
    end
end

function LogisticHeavisideLikelihood()
    LogisticHeavisideLikelihood{Float64}()
end;

function pdf(l::LogisticHeavisideLikelihood,f::AbstractVector)
    logisticheaviside(f)
end


function pdf(l::LogisticHeavisideLikelihood,y::Integer,f::AbstractVector)
    logisticheaviside(f)[y]
end

function logisticheaviside(f::AbstractVector)
    p = ones(length(f))
    for i in 1:length(f)
        for j in 1:length(f)
            p[i] *= i == j ? 1 : logistic(f[i]-f[j])
        end
    end
    return p./sum(p)
end

# function pdf(l::LogisticHeavisideLikelihood,y::Integer,f::AbstractVector)
#     logisticheaviside(f)[y]
# end

function Base.show(io::IO,model::LogisticHeavisideLikelihood{T}) where T
    print(io,"Logistic-Heaviside Likelihood")
end

function init_likelihood(likelihood::LogisticHeavisideLikelihood{T},inference::Inference{T},nLatent::Integer,nSamplesUsed::Integer,nFeatures::Integer) where T
    if inference isa AnalyticVI || inference isa GibbsSampling
        c = [ones(T,nSamplesUsed) for i in 1:nLatent]
#         α = nLatent*ones(T,nSamplesUsed)
#         β = nLatent*ones(T,nSamplesUsed)
        θ = [abs.(rand(T,nSamplesUsed))*2 for i in 1:nLatent]
#         γ = [abs.(rand(T,nSamplesUsed)) for i in 1:nLatent]
        LogisticHeavisideLikelihood{T}(likelihood.Y,likelihood.class_mapping,likelihood.ind_mapping,likelihood.y_class,c,θ)
    else
        return likelihood
    end
end

## Local Updates Section ##

function local_updates!(model::VGP{T,LogisticHeavisideLikelihood{T},AnalyticVI{T},V}) where {T<:Real,V}
    ## updates for model.likelihood.c
    for i in 1:model.nSample ## Traversing all the observations
        for j in 1:model.nLatent ## Traversing all the labels
            k = model.likelihood.y_class[i] ## k stands for the label of ith observation
            if k == j
                model.likelihood.c[k][i] = 0 ## There is no kth polya-gamma variable
                model.likelihood.θ[k][i] = 0 ## There is no kth polya-gamma variable
            else
                t = model.nFeatures ## t is the number of row elements
                model.likelihood.c[j][i] = sqrt((model.μ[k][i])^2 + (model.μ[j][i])^2 + model.Σ[k][(i-1)*t+i] + model.Σ[j][(i-1)*t+i])
                ## updates for model.likelihood.θ
                model.likelihood.θ[j][i] = 0.5/model.likelihood.c[j][i]*tanh(0.5*model.likelihood.c[j][i])
            end
        end
    end
end

function local_updates!(model::SVGP{T,LogisticHeavisideLikelihood{T},AnalyticVI{T},V}) where {T<:Real,V<:AbstractVector{T}}
    ## updates for model.likelihood.c
    for i in 1:model.inference.nSamplesUsed
        for j in 1:model.nLatent
            k = model.likelihood.y_class[model.inference.MBIndices[i]]
            if k == j
                model.likelihood.c[k][i] = 0
                model.likelihood.θ[k][i] = 0
            else
                model.likelihood.c[j][i] =  sqrt(abs2(dot(view(model.κ[k],i,:),model.μ[k])) +
                                            abs2(dot(view(model.κ[j],i,:),model.μ[j])) +
                                            transpose(view(model.κ[k],i,:)) * model.Σ[k] * view(model.κ[k],i,:) +
                                            transpose(view(model.κ[j],i,:)) * model.Σ[j] * view(model.κ[j],i,:) +
                                            model.K̃[j][i] +
                                            model.K̃[k][i])
                ## updates for model.likelihood.θ
                model.likelihood.θ[j][i] = 0.5/model.likelihood.c[j][i]*tanh(0.5*model.likelihood.c[j][i])
            end
        end
    end
end


## Global Gradient Section ##

function ∇E_μ(model::VGP{T,<:LogisticHeavisideLikelihood}) where {T}
    e = deepcopy(model.η₁)
    for i in 1:model.nFeatures
        for j in 1:model.nLatent
            k = model.likelihood.y_class[i]
            if k == j
                container = [l for l in 1:model.nLatent]
                deleteat!(container, k)
                e[k][i] = (model.nLatent - 1)/2 + sum(model.likelihood.θ[a][i] * model.μ[a][i] for a in container)
            else
                e[j][i] = -0.5 + model.likelihood.θ[j][i] * model.μ[k][i]
            end
        end
    end
    return e
end

function ∇E_μ(model::SVGP{T,<:LogisticHeavisideLikelihood}) where {T}
    e = Vector([zeros(model.inference.nSamplesUsed) for _ in 1:model.nLatent])
    for i in model.inference.nSamplesUsed
        for j in 1:model.nLatent
            k = model.likelihood.y_class[model.inference.MBIndices[i]]
            if k == j
                container = [l for l in 1:model.nLatent]
                deleteat!(container, k)
                e[k][i] = (model.nLatent - 1)/2 + sum(model.likelihood.θ[a][i] * dot(view(model.κ[a],i,:),model.μ[a]) for a in container)
            else
                e[j][i] = -0.5 + model.likelihood.θ[j][i] * dot(view(model.κ[k],i,:),model.μ[k])
            end
        end
    end
    return e
end

∇E_μ(model::SVGP{T,<:LogisticHeavisideLikelihood},i::Int) where {T} = ∇E_μ(model)[i]

function ∇E_Σ(model::AbstractGP{T,<:LogisticHeavisideLikelihood}) where {T}
    v = deepcopy(model.likelihood.θ)
    for i in model.inference.nSamplesUsed ## Traversing all the observations
        for j in 1:model.nLatent ## Traversing all the labels
            k = model.likelihood.y_class[model.inference.MBIndices[i]] ## k stands for the label of ith observation
            if k == j
                container = [l for l in 1:model.nLatent]
                deleteat!(container, k)
                v[k][i] = sum([model.likelihood.θ[a][i] for a in container])
            else
                v[j][i] = model.likelihood.θ[j][i]
            end
        end
    end
    return v
end

∇E_Σ(model::AbstractGP{T,<:LogisticHeavisideLikelihood},i::Int) where {T} = ∇E_Σ(model)[i]

## ELBO Section ##

function ELBO(model::AbstractGP{T,<:LogisticHeavisideLikelihood,<:AnalyticVI}) where {T}
    return expecLogLikelihood(model) - GaussianKL(model) - PolyaGammaKL(model)
end

function expecLogLikelihood(model::VGP{T,<:LogisticHeavisideLikelihood,<:AnalyticVI}) where {T}
    tot = 0
    for i in 1:model.nSample
        for j in 1:model.nLatent
            k = model.likelihood.y_class[i]
            if k == j
                continue
            else
                tot +=  0.5*(model.μ[k][i]-model.μ[j][i]-
                        (abs2(model.μ[k][i])+abs2(model.μ[j][i])+model.Σ[k][i,i] +
                        model.Σ[j][i,i])*model.likelihood.θ[j][i])
            end
        end
    end
    tot -= model.nSample*logtwo
    return tot
end

function expecLogLikelihood(model::SVGP{T,<:LogisticHeavisideLikelihood,<:AnalyticVI}) where {T}
    tot = 0
    for i in 1:model.inference.nSamplesUsed
        for j in 1:model.nLatent
            k = model.likelihood.y_class[model.inference.MBIndices[i]]
            if k == j
                continue
            else
                tot += 0.5* (dot(view(model.κ[k],i,:),model.μ[k]) -
                            dot(view(model.κ[j],i,:),model.μ[j]) -
                            (abs2(dot(view(model.κ[k],i,:),model.μ[k])) +
                            abs2(dot(view(model.κ[j],i,:),model.μ[j])) +
                            transpose(view(model.κ[k],i,:)) * model.Σ[k] * view(model.κ[k],i,:) +
                            transpose(view(model.κ[j],i,:)) * model.Σ[j] * view(model.κ[j],i,:)) *
                            model.likelihood.θ[j][i])
            end
        end
    end
    tot -= model.inference.nSamplesUsed*logtwo
    return model.inference.ρ*tot
end

function PolyaGammaKL(model::VGP{T,<:LogisticHeavisideLikelihood,<:AnalyticVI}) where {T}
    tot = 0
    for i in 1:model.nSample
        k = model.likelihood.y_class[i]
        for j in 1:model.nLatent
            if k == j
                continue
            else
                tot += -log(cosh(0.5*model.likelihood.c[j][i])) + 0.5*abs2(model.likelihood.c[j][i])*model.likelihood.θ[j][i]
            end
        end
    end
    return -tot
end

function PolyaGammaKL(model::SVGP{T,<:LogisticHeavisideLikelihood,<:AnalyticVI}) where {T}
    tot = 0
    for i in 1:model.inference.nSamplesUsed
        k = model.likelihood.y_class[model.inference.MBIndices[i]]
        for j in 1:model.nLatent
            if k == j
                continue
            else
                tot += -log(cosh(0.5*model.likelihood.c[j][i])) + 0.5*abs2(model.likelihood.c[j][i])*model.likelihood.θ[j][i]
            end
        end
    end
    return -(model.inference.ρ * tot)
end


function sample_global!(model::VGP{T,<:LogisticHeavisideLikelihood,<:GibbsSampling}) where {T}
    model.Σ .= inv.(Symmetric.(Diagonal.(2.0.*∇E_Σ(model)).+model.invKnn))
    model.μ .= rand.(MvNormal.(model.Σ.*(∇E_μ(model).+model.invKnn.*model.μ₀),model.Σ))
    return nothing
end
