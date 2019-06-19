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

function _VGP(X::AbstractMatrix,μ::V,Σ::M,kernel::Kernel{T},μ₀::PriorMean{T}) where {T<:Real,V<:AbstractVector{T},M}
    K = Symmetric(kernelmatrix(X,kernel))
    η₂ = -0.5*inv(Σ); η₁ = -2*η₂*μ;
    _VGP{T,V,M}(μ,Σ,X,kernel,K,inv(K+T(jitter)*getvariance(kernel)*I),μ₀,η₁,η₂)
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

function _SVGP(Z::AbstractMatrix,μ::V,Σ::M,kernel::Kernel{T},K::M,κ::Matrix{T},μ₀::PriorMean{T}) where {T<:Real,V<:AbstractVector{T},M}
    _SVGP{T,V,M}(μ,Σ,Z,kernel,K,similar(K),κ,ones(T,size(κ,2)),μ₀,similar(μ),similar(Σ))
end

Base.length(gp::_GP) = length(gp.μ)
Statistics.mean(gp::_GP) = gp.μ
Statistics.cov(gp::_GP) = gp.Σ
corr(gp::_VGP,X::AbstractMatrix) = kernelmatrix(gp.kernel,X,gp.X)
corr(gp::_SVGP,X::AbstractMatrix) = kernelmatrix(gp.kernel,X,gp.Z)
kernel(gp::_GP) = gp.kernel

diag_quad(A::AbstractMatrix,x::AbstractArray) = opt_diag(x*A,x)
quad(A::AbstractMatrix,x::AbstractArray) = x*A*transpose(x)

cond_mean(gp::_GP,X::AbstractMatrix,k::Matrix=corr(gp,X)) = k*gp.K⁻¹*gp.μ
cond_diagcov(gp::_GP,X::AbstractMatrix,k::Matrix) = kerneldiagmatrix(gp.kernel,X)-diag_quad(gp.K⁻¹*(I-gp.Σ*gp.K⁻¹),corr(gp,X))
cond_cov(gp::_GP,X::AbstractMatrix,k::Matrix) = kernelmatrix(gp.kernel,X)-quad(gp.K⁻¹*(I-gp.Σ*gp.K⁻¹),corr(gp,X))
cond_mean_and_diagcov(gp::_GP,X::AbstractMatrix,k::Matrix=corr(gp,X))=  cond_mean(gp,X,k),cond_diagcov(gp,X,k)
cond_mean_and_cov(gp::_GP,X::AbstractMatrix,k::Matrix=corr(gp,X))=  cond_mean(gp,X,k),cond_cov(gp,X,k)
function Base.show(io::IO,gp::_GP)
    print(io,"Latent Gaussian Process with $(length(gp)) points, variational parameters:\n\n\t μ: mean=$(mean(mean(gp))), median=$(median(mean(gp))), μᵢ∈[$(minimum(mean(gp))),$(maximum(mean(gp)))]\n\n\t Σ: det=$(det(cov(gp))), trace=$(tr(cov(gp))), diag(Σ)ᵢ∈[$(minimum(diag(cov(gp)))),$(maximum(diag(cov(gp))))]")
    print(io,"\n\nkernel: $(kernel(gp))")
end

function compute_self_kernelmatrices(gp::_VGP{T}) where T
    kernelmatrix!(gp.K.data,gp.X,gp.kernel)
    gp.K⁻¹ .= inv(gp.K+T(jitter)*getvariance(gp.kernel)*I)
end

function natural_to_variational(gp::_VGP{T}) where T
    gp.Σ .= -0.5*inv(gp.η₂)
    gp.μ .= gp.Σ*gp.η₁
end
