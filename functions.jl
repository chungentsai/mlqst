using LinearAlgebra
using Kronecker
using Random
using Statistics


function fidelity(rho::Array{ComplexF64, 2}, sigma::Array{ComplexF64, 2})
    S = sqrt(sigma)
    return real((tr(sqrt(S * rho * S'))) ^ 2)
end


function random_state(q::Int64)
    d = 2 ^ q
    ρ = randn(ComplexF64, d, d)
    ρ = ρ * ρ' 
    ρ = ρ / tr(ρ)
    return ρ
end


function w_state(q::Int64)
    w = zeros(ComplexF64, 2 ^ q)
    for i = 1: q
        w[2 ^ (i - 1) + 1] = 1
    end
    W = w * w'
    W = W / tr(W)
    return W
end


function pauli_observables(q::Int64, Q::Int64)
    # Pauli matrices
    σ = zeros(ComplexF64, 2, 2, 4); 
    σ[:, :, 1] = [0 1; 1 0];
    σ[:, :, 2] = [0 -im; im 0];
    σ[:, :, 3] = [1 0; 0 -1];
    σ[:, :, 4] = [1 0; 0 1];
    
    d = 2 ^ q    # dimension
    observables = zeros(ComplexF64, d, d, Q ^ q)
    i = 1
    @inbounds for idx in Iterators.product(ntuple(i -> 1: Q, q)...)
        X = ones(ComplexF64, 1, 1)
        @inbounds for j in idx
            X = X ⊗ view(σ, :, :, j)
        end
        observables[:, :, i] = collect(X)
        i = i + 1
    end

    return observables 
end


function pauli_povm(q::Int64, Q::Int64)
    X = pauli_observables(q, Q)
    d = 2 ^ q
    K = Q ^ q
    POVM = zeros(ComplexF64, d, d, K)
    @inbounds for k in 1:K
        E   = eigen(view(X, :, :, k))
        E_1 = E.vectors[:, real(E.values) .> 0]
        POVM[:,:,k] = E_1 * E_1'
    end

    return POVM
end


function measure(ρ::Array{ComplexF64, 2}, 
                 POVM::Array{ComplexF64, 3}, 
                 idx_obs::Array{Int64, 1})
    n = length(idx_obs)
    outcomes = zeros(Bool, n)
    @inbounds for i in 1: n
        if rand() <= real(view(POVM, :, :, view(idx_obs, i)) ⋅ ρ)
            outcomes[i] = 1
        end
    end
    return outcomes

end


function generate_data(POVM::Array{ComplexF64, 3}, 
                       idx_obs::Array{Int64, 1}, 
                       outcomes::Array{Bool, 1})
    d = size(POVM)[1]
    n = length(idx_obs)
    data = zeros(ComplexF64, d, d, n)
    @inbounds for i in 1: n
        if outcomes[i] == 1
            data[:, :, i] = view(POVM, :, :, view(idx_obs, i))
        else
            data[:, :, i] = I - view(POVM, :, :, view(idx_obs, i))
        end
    end
    return data

end


function compute_λ(data::Array{ComplexF64, 3}, ρ::Array{ComplexF64, 2})
    n = size(data)[3]
    λ = zeros(Float64, n)
    @inbounds for i in 1: n
        λ[i] = real(view(data, :, :, i) ⋅ ρ)
    end
    return λ

end


function f(λ::Array{Float64, 1})
    return mean(-log.(λ))
end


function ∇f(data::Array{ComplexF64, 3}, λ::Array{Float64, 1})
    d, ~, n = size(data)
    G = zeros(ComplexF64, d, d)
    @inbounds for i in 1: n
        G = G .- view(data, :, :, i) ./ view(λ, i)
    end
    return G / n 
end


function log_barrier_projection(
    u::Array{Float64, 1},
    ε::Float64
    )
    # compute argmin_{x∈Δ} D_h(x,u) where h(x)=∑_{i=1}^d -log(x_i)
    # minimize ϕ(θ) = θ - ∑_i log(θ + u_i^{-1})

    θ::Float64 = 1 - minimum(1 ./ u)
    a::Array{Float64, 1} = 1 ./ ((1 ./ u) .+ θ)
    ∇::Float64 = 1 - sum(a)
    ∇2::Float64 = a ⋅ a
    λt::Float64 = abs(∇) / sqrt(∇2)

    while λt > ε
        a = 1 ./ ((1 ./ u) .+ θ)
        ∇ = 1 - norm(a, 1)
        ∇2 =  a ⋅ a
        θ = θ - ∇ / ∇2
        λt = abs(∇) / sqrt(∇2)
    end

    return (1 ./ ((1 ./ u) .+ θ))
end


function α(ρ, v)
    return -real(tr(ρ * v * ρ) / tr(ρ * ρ))
end


function dual_norm2(ρ, σ)
    A = ρ * σ
    return real(tr(A * A))
end