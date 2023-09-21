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
    compute_λ::Function,
    verbose
    )
    # A. I. Lvovsky, Iterative maximum-likelihood reconstruction in quan- tum homodyne tomography, 2004 (https://arxiv.org/abs/quant-ph/0311097)
    name = "RρR"
    println(name * " starts.")
    @printf(io, "%s\n%d\n%d\n", name, n_epoch, n_rate)
    output = init_output(n_epoch)
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
        print_output(io, output, t, verbose)
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
    compute_λ::Function,
    verbose
    )
    # C.-M. Lin, H.-C. Cheng, and Y.-H. Li, Maximum-likelihood quantum state tomography by Cover's method with non-asymptotic analysis, 2022 (https://arxiv.org/abs/2110.00747)
    name = "QEM"
    println(name * " starts.")
    @printf(io, "%s\n%d\n%d\n", name, n_epoch, n_rate)
    output = init_output(n_epoch)
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
        print_output(io, output, t, verbose)
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
    compute_λ::Function,
    verbose
    )
    # A. Carderera, M. Besançon, and S. Pokutta, Simple steps are all you need: Frank-Wolfe and generalized self-concordant functions, 2021 (https://proceedings.neurips.cc/paper/2021/hash/2b323d6eb28422cef49b266557dd31ad-Abstract.html)
    name = "MonoFW"
    println(name * " starts.")
    @printf(io, "%s\n%d\n%d\n", name, n_epoch, n_rate)
    output = init_output(n_epoch)
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
        print_output(io, output, t, verbose)
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
    compute_λ::Function,
    verbose
    )
    name = "DA"
    println(name * " starts.")
    @printf(io, "%s\n%d\n%d\n", name, n_epoch, n_rate)
    output = init_output(n_epoch)
    to = TimerOutput()

    d::Int64 = size(ρ_true)[1]
    ρ_bar::Matrix{ComplexF64} = Matrix{ComplexF64}(I, d, d) / d
    ρ::Matrix{ComplexF64} = Matrix{ComplexF64}(I, d, d) / d
    ∑grad::Matrix{ComplexF64} = zeros(ComplexF64, d, d)
    ∑dual_norm2 = 0
    
    @timeit to "iteration" λ = compute_λ(ρ_bar)

    @inbounds for t = 1: n_epoch
        # update iterate
        @timeit to "iteration" begin
            # compute learning rates
            grad = ∇f(λ)
            ∑dual_norm2 += dual_norm2(ρ_bar, grad + α(ρ_bar, grad) * Matrix{ComplexF64}(I, d, d))
            η = sqrt(d) / sqrt(4 * d + 1 + ∑dual_norm2)
            
            # update step
            ∑grad += grad
            Λ_inv, U = eigen(Hermitian(η * ∑grad))
            
            # projection step
            Λ = log_barrier_projection(1 ./ Λ_inv, 1e-5)
            ρ = U * Diagonal(Λ) * adjoint(U)

            # averaging step
            ρ_bar = (t * ρ_bar + ρ) / (t + 1.0)
            λ = compute_λ(ρ_bar)  
        end

        update_output!(output, t, t, fidelity(ρ_true, ρ_bar), f(λ),
                       TimerOutputs.time(to["iteration"]) * 1e-9)
        print_output(io, output, t, verbose)
    end

    return output
end



function FW(
    n_epoch::Int64, 
    n_rate::Int64, 
    io::IOStream, 
    ρ_true::Array{ComplexF64, 2}, 
    N::Int64, 
    f::Function, 
    ∇f::Function, 
    compute_λ::Function,
    verbose
    )
    # Renbo Zhao and Robert M. Freund, Analysis of the Frank–Wolfe method for convex composite optimization involving a logarithmically-homogeneous barrier, 2023 (https://link.springer.com/article/10.1007/s10107-022-01820-9)
    name = "FW"
    println(name * " starts.")
    @printf(io, "%s\n%d\n%d\n", name, n_epoch, n_rate)
    output = init_output(n_epoch)
    to = TimerOutput()

    d::Int64 = size(ρ_true)[1]
    ρ = Matrix{ComplexF64}(I, d, d) / d    # initialize at the maximally 
    ρ_prev = Matrix{ComplexF64}(I, d, d) / d
    λ = zeros(Float64, 1, N)
    fval = 0

    @timeit to "iteration" λ = compute_λ(ρ)
    
    @inbounds for t = 1: n_epoch

        # update iterate
        @timeit to "iteration" begin
            grad = ∇f(λ)
            σ, v   = eigs(-grad, nev = 1, which = :LM)
            V      = v * v'
            direction = V - ρ

            G = real(dot(grad, -direction))
            hess = grad * grad'
            D = real(dot(direction, hess, direction))^0.5
            η      = min(G / (D * (G + D)), 1)

            ρ = ρ + η * direction

            λ = compute_λ(ρ) 
            fval = f(λ)
        end

        update_output!(output, t, t, fidelity(ρ_true, ρ), fval,
                       TimerOutputs.time(to["iteration"]) * 1e-9)
        print_output(io, output, t, verbose)
    end

    return output
end