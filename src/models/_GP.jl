abstract type _GP{T<:Real} end;

struct _VGP{T<:Real,V<:AbstractVector{T},M} <: _GP{T}
    μ::V
    Σ::M
    X::AbstractMatrix
    kernel::Kernel{T}
    K::M
    K⁻¹::M
    μ₀::PriorMean{T}
    η₁::V
    η₂::M
end

_VGP(X::AbstractMatrix,μ::V,Σ::M,kernel::Kernel{T},K::M,μ₀::PriorMean{T},η₁::V,η₂::M) where {T<:Real,V<:AbstractVector{T},M}
    _VGP{T,V,M}(μ,Σ,X,kernel,K,inv(K),μ₀,η₁,η₂)
end

struct _SVGP{T<:Real,V<:AbstractVector{T},M} <: _GP{T}
    μ::V
    Σ::M
    Z::AbstractMatrix
    kernel::Kernel{T}
    K::M
    K⁻¹::M
    κ::Matrix{T}
    K̃::V
    μ₀::PriorMean{T}
    η₁::V
    η₂::M
end

_SVGP(Z::AbstractMatrix,μ::V,Σ::M,kernel::Kernel{T},K::M,κ::Matrix{T},μ₀::PriorMean{T}) where {T<:Real,V<:AbstractVector{T},M}
    _SVGP{T,V,M}(μ,Σ,Z,kernel,K,similar(K),κ,ones(T,size(κ,2)),μ₀,similar(μ),similar(Σ))
end

Base.length(gp::_GP) = length(gp.μ)
Base.mean(gp::_GP) = gp.μ
Base.cov(gp::_GP) = gp.Σ
corr(gp::_VGP,X::AbstractMatrix) = kernelmatrix(gp.kernel,X,gp.X)
corr(gp::_SVGP,X::AbstractMatrix) = kernelmatrix(gp.kernel,X,gp.Z)

diag_quad(A::AbstractMatrix,x::AbstractArray) = opt_diag(x*A,x)
quad(A::AbstractMatrix,x::AbstractArray) = x*A*transpose(x)

cond_mean(gp::_GP,X::AbstractMatrix,k::Matrix) = corr(gp,X)*gp.K⁻¹*gp.μ
cond_diagcov(gp::_GP,X::AbstractMatrix,k::Matrix) = kerneldiagmatrix(gp.kernel,X)-diag_quad(gp.K⁻¹*(I-gp.Σ*gp.K⁻¹),corr(gp,X))
cond_cov(gp::_GP,X::AbstractMatrix,k::Matrix) = kernelmatrix(gp.kernel,X)-quad(gp.K⁻¹*(I-gp.Σ*gp.K⁻¹),corr(gp,X)
cond_mean_and_diagcov(gp::_GP,X::AbstractMatrix,k::Matrix=corr(gp,X))=  _cond_mean(gp,X,k),_cond_diagcov(gp,X,k)
cond_mean_and_cov(gp::_GP,X::AbstractMatrix,k::Matrix=corr(gp,X))=  _cond_mean(gp,X,k),_cond_cov(gp,X,k)
function Base.show(io::IO,model::_GP)
    print(io,"Latent Gaussian Process of size $(length(gp)),\n mean: $(mean(gp))\n cov $(cov(gp))")
end
