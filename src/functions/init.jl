""" Verify that the data is self-consistent and consistent with the likelihood """
function check_data!(X::AbstractArray{T1,N1},y::AbstractArray{T2,N2},likelihood::Likelihood) where {T1<:Real,T2,N1,N2}
    @assert (size(y,1)==size(X,1)) "There is not the same number of samples in X and y";
    @assert N1 <= 2 "The input matrix X can only be a vector or a matrix"
    if N1 == 1 #Reshape a X vector as a matrix
        X = reshape(X,length(X),1);
    end
    y,nLatent,likelihood = treat_labels!(y,likelihood)
    return X,y,nLatent,likelihood
end

""" Verify that the likelihood and inference are compatible (are implemented) """
function check_implementation(model::Symbol,likelihood::L,inference::I) where {I<:Inference,L<:Likelihood}
    if isa(likelihood,GaussianLikelihood)
        if model == :GP && isa(inference,Analytic)
            return true
        elseif model == :SVGP && isa(inference,AnalyticVI)
            return true
        elseif model == :OnlineVGP
            return true
        else
            return false
        end
    elseif isa(likelihood,StudentTLikelihood)
        if isa(inference,AnalyticVI)
            return true
        else
            return false
        end
    elseif isa(likelihood,LogisticLikelihood)
        if isa(inference,AnalyticVI) || (model == :VGP && isa(inference,GibbsSampling))
            return true
        else
            return false
        end
    elseif isa(likelihood,BayesianSVM)
        if isa(inference,AnalyticVI)
            return true
        else
            return false
        end
    elseif isa(likelihood,SoftMaxLikelihood)
        if isa(inference,MCMCIntegrationVI)
            return true
        else
            return false
        end
    elseif isa(likelihood,LogisticSoftMaxLikelihood)
        if isa(inference,AnalyticVI) || (model == :VGP && isa(inference,GibbsSampling))
            return true
        elseif isa(inference,MCMCIntegrationVI)
            return true
        else
            return false
        end
    else
        return false
    end
end
