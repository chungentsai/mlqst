using Arpack
using TimerOutputs
using ExponentialUtilities
using Printf
using LinearAlgebra


function RρR(
    n_epoch::Int64, 
    n_rate::Int64, 
    io::IOStream, 
    ρ_true::Array{ComplexF64, 2}, 
    N::Int64, 
    f::Function, 
    ∇f::Function, 
    compute_λ::Function
    )
    # A. I. Lvovsky, Iterative maximum-likelihood reconstruction in quan- tum homodyne tomography, 2004 (https://arxiv.org/abs/quant-ph/0311097)

    output = init_output(n_epoch)
    println("RρR starts.")
    to = TimerOutput()


    d::Int64 = size(ρ_true)[1]
    ρ = Matrix{ComplexF64}(I, d, d) / d    # initialize at the maximally

    @timeit to "iteration" λ = compute_λ(ρ)

    @inbounds for t = 1: n_epoch
        # update iterate
        @timeit to "iteration" begin
            grad = ∇f(λ)
            ρ = grad * ρ * grad
            ρ /= tr(ρ)
            λ  = compute_λ(ρ)
        end

        update_output!(output, t, t, fidelity(ρ_true, ρ), f(λ),
                       TimerOutputs.time(to["iteration"]) * 1e-9)
        print_output(io, output, t)
    end

    return output
end


function QEM(
    n_epoch::Int64, 
    n_rate::Int64, 
    io::IOStream, 
    ρ_true::Array{ComplexF64, 2}, 
    N::Int64, 
    f::Function, 
    ∇f::Function, 
    compute_λ::Function
    )
    # C.-M. Lin, H.-C. Cheng, and Y.-H. Li, Maximum-likelihood quantum state tomography by Cover's method with non-asymptotic analysis, 2022 (https://arxiv.org/abs/2110.00747)

    output = init_output(n_epoch)
    println("QEM starts.")
    to = TimerOutput()

    d::Int64 = size(ρ_true)[1]
    ρ = Matrix{ComplexF64}(I, d, d) / d    # initialize at the maximally 
    
    @timeit to "iteration" λ = compute_λ(ρ)

    @inbounds for t = 1: n_epoch

        # update iterate
        @timeit to "iteration" begin
            grad = ∇f(λ)
            ρ = exp(log(ρ) + log(-grad))
            ρ /= tr(ρ)
            λ = compute_λ(ρ)
        end

        update_output!(output, t, t, fidelity(ρ_true, ρ), f(λ),
                       TimerOutputs.time(to["iteration"]) * 1e-9)
        print_output(io, output, t)
    end

    return output
end


function M_FW(
    n_epoch::Int64, 
    n_rate::Int64, 
    io::IOStream, 
    ρ_true::Array{ComplexF64, 2}, 
    N::Int64, 
    f::Function, 
    ∇f::Function, 
    compute_λ::Function
    )
    # A. Carderera, M. Besançon, and S. Pokutta, Simple steps are all you need: Frank-Wolfe and generalized self-concordant functions, 2021 (https://proceedings.neurips.cc/paper/2021/hash/2b323d6eb28422cef49b266557dd31ad-Abstract.html)

    output = init_output(n_epoch)
    println("M_FW starts.")
    to = TimerOutput()

    d::Int64 = size(ρ_true)[1]
    ρ = Matrix{ComplexF64}(I, d, d) / d    # initialize at the maximally 
    ρ_prev = Matrix{ComplexF64}(I, d, d) / d
    λ = zeros(Float64, 1, N)
    λ_prev = zeros(Float64, 1, N)
    fval = 0

    @timeit to "iteration" λ = compute_λ(ρ)
    
    @inbounds for t = 1: n_epoch

        # update iterate
        @timeit to "iteration" begin
            ρ_prev =  ρ
            λ_prev  = λ

            grad = ∇f(λ)
            σ, v   = eigs(-grad, nev = 1, which = :LM)
            V      = v * v'
            η      = 2.0 / (t + 1.0)
            ρ = ρ + η * (V - ρ)

            λ = compute_λ(ρ) 
            fval = f(λ)
            
            if t > 1 && fval > output["fval"][t-1]
                λ = λ_prev
                ρ = ρ_prev
                fval = output["fval"][t-1]
            end
        end

        update_output!(output, t, t, fidelity(ρ_true, ρ), fval,
                       TimerOutputs.time(to["iteration"]) * 1e-9)
        print_output(io, output, t)
    end

    return output
end


function DA(
    n_epoch::Int64, 
    n_rate::Int64, 
    io::IOStream, 
    ρ_true::Array{ComplexF64, 2}, 
    N::Int64, 
    f::Function, 
    ∇f::Function, 
    compute_λ::Function
    )

    output = init_output(n_epoch)
    println("DA starts.")
    to = TimerOutput()

    d::Int64 = size(ρ_true)[1]
    ρ_bar = Matrix{ComplexF64}(I, d, d) / d
    ρ::Matrix{ComplexF64} = Matrix{ComplexF64}(I, d, d) / d
    σ_inv::Matrix{ComplexF64} = Matrix{ComplexF64}(I, d, d) * d
    sum_dual_norm2 = 0
    
    @timeit to "iteration" λ = compute_λ(ρ_bar)

    @inbounds for t = 1: n_epoch

        # update iterate
        @timeit to "iteration" begin
            grad = ∇f(λ)
            sum_dual_norm2 += dual_norm2(ρ_bar, grad + α(ρ_bar, grad) * Matrix{ComplexF64}(I, d, d))
            η = sqrt(d) / sqrt(4 * d + 1 + sum_dual_norm2)

            # update step
            Λ, U = eigen(Hermitian(σ_inv + η * grad))
            σ_inv = U * Diagonal(Λ) * adjoint(U)

            # projection step
            Λ = log_barrier_projection(Λ, 1e-5)
            ρ = U * Diagonal(Λ) * adjoint(U)

            # averaging step
            ρ_bar = (t * ρ_bar + ρ) / (t + 1.0)
            λ = compute_λ(ρ_bar)
        end

        update_output!(output, t, t, fidelity(ρ_true, ρ_bar), f(λ),
                       TimerOutputs.time(to["iteration"]) * 1e-9)
        print_output(io, output, t)
    end

    return output
end