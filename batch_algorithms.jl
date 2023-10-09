using Arpack
using TimerOutputs
using ExponentialUtilities
using Printf
using LinearAlgebra


function iMLE(n_epoch::Int64, n_rate::Int64)
    name = "iMLE"
    println(name * " starts.")
    @printf(io, "%s\n%d\n%d\n", name, n_epoch, n_rate)
    output = init_output(n_epoch)
    to = TimerOutput()


    d::Int64 = size(ρ_true)[1]
    ρ::Matrix{ComplexF64} = Matrix{ComplexF64}(I, d, d) / d
    λ::Vector{Float64} = zeros(Float64, N)

    @timeit to "iteration" λ = compute_λ(ρ)

    @inbounds for t = 1: n_epoch
        @timeit to "iteration" begin
            grad = ∇f(λ)
            ρ = grad * ρ * grad
            ρ /= tr(ρ)
            λ  = compute_λ(ρ)
        end

        update_output!(output, t, t, fidelity(ρ_true, ρ), f(λ),
                       TimerOutputs.time(to["iteration"]) * 1e-9)
        print_output(io, output, t, VERBOSE)
    end

    return output
end


function QEM(n_epoch::Int64, n_rate::Int64)
    name = "QEM"
    println(name * " starts.")
    @printf(io, "%s\n%d\n%d\n", name, n_epoch, n_rate)
    output = init_output(n_epoch)
    to = TimerOutput()

    d::Int64 = size(ρ_true)[1]
    ρ::Matrix{ComplexF64} = Matrix{ComplexF64}(I, d, d) / d
    λ::Vector{Float64} = zeros(Float64, N)
    
    @timeit to "iteration" λ = compute_λ(ρ)

    @inbounds for t = 1: n_epoch
        @timeit to "iteration" begin
            grad = ∇f(λ)
            ρ = exp(log(ρ) + log(-grad))
            ρ /= tr(ρ)
            λ = compute_λ(ρ)
        end

        update_output!(output, t, t, fidelity(ρ_true, ρ), f(λ),
                       TimerOutputs.time(to["iteration"]) * 1e-9)
        print_output(io, output, t, VERBOSE)
    end

    return output
end


function M_FW(n_epoch::Int64, n_rate::Int64)
    name = "MonoFW"
    println(name * " starts.")
    @printf(io, "%s\n%d\n%d\n", name, n_epoch, n_rate)
    output = init_output(n_epoch)
    to = TimerOutput()

    d::Int64 = size(ρ_true)[1]
    ρ::Matrix{ComplexF64} = Matrix{ComplexF64}(I, d, d) / d
    ρ_prev::Matrix{ComplexF64} = Matrix{ComplexF64}(I, d, d) / d
    λ::Vector{Float64} = zeros(Float64, N)
    λ_prev::Vector{Float64} = zeros(Float64, 1, N)
    fval::Float64 = 0

    @timeit to "iteration" λ = compute_λ(ρ)
    
    @inbounds for t = 1: n_epoch

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
        print_output(io, output, t, VERBOSE)
    end

    return output
end



function BPG(n_epoch::Int64, n_rate::Int64)
    name = "BPG"
    println(name * " starts.")
    @printf(io, "%s\n%d\n%d\n", name, n_epoch, n_rate)
    output = init_output(n_epoch)
    to = TimerOutput()

    d::Int64 = size(ρ_true)[1]
    ρ::Matrix{ComplexF64} = Matrix{ComplexF64}(I, d, d) / d
    ρ_inv::Matrix{ComplexF64} = Matrix{ComplexF64}(I, d, d) * d
    λ::Vector{Float64} = zeros(Float64, N)
    η::Float64 = 1

    @timeit to "iteration" λ = compute_λ(ρ)

    @inbounds for t = 1: n_epoch
        @timeit to "iteration" begin            
            grad::Matrix{ComplexF64} = ∇f(λ)
            Λ_inv::Array{Float64, 1}, U::Matrix{ComplexF64} = eigen(Hermitian(ρ_inv + η * grad))
            Λ = log_barrier_projection(1 ./ Λ_inv, 1e-5)
            ρ = U * Diagonal(Λ) * adjoint(U)
            ρ_inv = U * Diagonal(1 ./ Λ) * adjoint(U)

            λ = compute_λ(ρ)  
        end

        update_output!(output, t, t, fidelity(ρ_true, ρ), f(λ),
                       TimerOutputs.time(to["iteration"]) * 1e-9)
        print_output(io, output, t, VERBOSE)
    end

    return output
end


function FW(n_epoch::Int64, n_rate::Int64)
    name = "Frank-Wolfe"
    println(name * " starts.")
    @printf(io, "%s\n%d\n%d\n", name, n_epoch, n_rate)
    output = init_output(n_epoch)
    to = TimerOutput()

    d::Int64 = size(ρ_true)[1]
    ρ::Matrix{ComplexF64} = Matrix{ComplexF64}(I, d, d) / d
    λ::Vector{Float64} = zeros(Float64, N)
    fval::Float64 = 0

    @timeit to "iteration" λ = compute_λ(ρ)
    
    @inbounds for t = 1: n_epoch

        @timeit to "iteration" begin
            
            grad = zeros(ComplexF64, d, d)
            hess = zeros(ComplexF64, d, d)
            @inbounds for i in 1: N
                temp = view(data, :, :, i) ./ view(λ, i)
                grad -= temp
                hess += temp * temp 
            end

            σ, v   = eigs(-grad, nev = 1, which = :LM)
            V      = v * v'
            direction = V - ρ

            G = real(dot(grad, -direction))
            D = real(dot(direction, hess, direction))^0.5
            η      = min(G / (D * (G + D)), 1)

            ρ = ρ + η * direction

            λ = compute_λ(ρ) 
            fval = f(λ)
        end

        update_output!(output, t, t, fidelity(ρ_true, ρ), fval,
                       TimerOutputs.time(to["iteration"]) * 1e-9)
        print_output(io, output, t, VERBOSE)
    end

    return output
end


function EMD(n_epoch::Int64, n_rate::Int64)
    name = "EMD"
    println(name * " starts.")
    @printf(io, "%s\n%d\n%d\n", name, n_epoch, n_rate)
    output = init_output(n_epoch)
    to = TimerOutput()

    d::Int64 = size(ρ_true)[1]
    ρ::Matrix{ComplexF64} = Matrix{ComplexF64}(I, d, d) / d
    λ::Vector{Float64} = zeros(Float64, N)

    @timeit to "iteration" begin
        λ = compute_λ(ρ)
        fval = f(λ)
    end
    α0::Float64 = 10
    r::Float64 = 0.5
    τ::Float64 = 0.5

    @inbounds for t = 1: n_epoch
        @timeit to "iteration" begin
            grad::Matrix{ComplexF64} = ∇f(λ)
            
            # Armijo line search
            α = α0
            ρα = exp(log(ρ) - α*grad)
            ρα /= tr(ρα)
 
            round = 0
            while τ*real(grad ⋅ (ρα - ρ)) + fval < f(compute_λ(ρα)) && round < 10
                α *= r
                ρα = exp(log(ρ) - α*grad)
                ρα /= tr(ρα)
                round += 1
            end
            if round < 10
                ρ = ρα
            else
                ρ = ρ
            end
            λ = compute_λ(ρ)
            fval = f(λ)
        end

        update_output!(output, t, t, fidelity(ρ_true, ρ), fval,
                       TimerOutputs.time(to["iteration"]) * 1e-9)
        print_output(io, output, t, VERBOSE)
    end

    return output
end



function diluted_iMLE(n_epoch::Int64, n_rate::Int64)
    name = "Diluted iMLE"
    println(name * " starts.")
    @printf(io, "%s\n%d\n%d\n", name, n_epoch, n_rate)
    output = init_output(n_epoch)
    to = TimerOutput()


    d::Int64 = size(ρ_true)[1]
    ρ::Matrix{ComplexF64} = Matrix{ComplexF64}(I, d, d) / d
    λ::Vector{Float64} = zeros(Float64, N)
    α0::Float64 = 1

    @timeit to "iteration" begin
        λ = compute_λ(ρ)
        fval = f(λ)
    end

    @inbounds for t = 1: n_epoch
        @timeit to "iteration" begin
            grad::Matrix{ComplexF64} = ∇f(λ)
            τ::Float64 = 1e-4
            r::Float64 = 0.5
            
            # Armijo line search
            α = α0
            temp = (Matrix{ComplexF64}(I, d, d) - α*grad) / (1+α)
            ρα = temp * ρ * temp
            ρα /= tr(ρα)

            round = 0
            while τ*real(grad ⋅ (ρα - ρ)) + fval < f(compute_λ(ρα)) && round < 10
                α *= r
                temp = (Matrix{ComplexF64}(I, d, d) - α*grad) / (1+α)
                ρα = temp * ρ * temp
                ρα /= tr(ρα)
                round += 1
            end
            if round < 10
                ρ = ρα
            else
                ρ = ρ
            end
            λ = compute_λ(ρ)
            fval = f(λ)
        end

        update_output!(output, t, t, fidelity(ρ_true, ρ), fval,
                       TimerOutputs.time(to["iteration"]) * 1e-9)
        print_output(io, output, t, VERBOSE)
    end

    return output
end
