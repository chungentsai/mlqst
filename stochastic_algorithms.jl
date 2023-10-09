using Arpack
using TimerOutputs
using ExponentialUtilities
using Printf
using LinearAlgebra


function SQSB(n_epoch::Int64, n_rate::Int64)
    name = "SQSB"
    println(name * " starts.")
    @printf(io, "%s\n%d\n%d\n", name, n_epoch, n_rate)
    len_output = n_epoch * n_rate
    output = init_output(len_output)
    to = TimerOutput()

    d::Int64 = size(ρ_true)[1]
    ρ_bar::Matrix{ComplexF64} = Matrix{ComplexF64}(I, d, d) / d
    ρ::Matrix{ComplexF64} = Matrix{ComplexF64}(I, d, d) / d

    n_iter::Int64 = n_epoch * N
    period::Int64 = N ÷ n_rate
    @timeit to "iteration" begin
        idx = rand(1:N, n_iter)
        η::Float64 = sqrt( log( d ) / n_iter / d )
        η = η / ( 1.0 + η )
        σ = ( 1.0 - η ) * Matrix{ComplexF64}(I, d, d)
    end
 
    @inbounds for iter = 1:n_iter
        @timeit to "iteration" begin
            grad::Matrix{ComplexF64} = - view(data, :, :, idx[iter]) / real(view(data, :, :, idx[iter]) ⋅ ρ)
            ρ = exp(log(ρ) + log(σ - η * grad))
            ρ /= tr(ρ)
            
            ρ_bar = (iter * ρ_bar + ρ) / (iter + 1.0)
        end

        if mod(iter, period) == 0
            λ = compute_λ(ρ_bar)
            update_output!(output, iter÷period, iter/N, fidelity(ρ_true, ρ_bar), f(λ),
                            TimerOutputs.time(to["iteration"]) * 1e-9)
            print_output(io, output, iter÷period, VERBOSE)
        end
    end

    return output
end


function SQLBOMD(n_epoch::Int64, n_rate::Int64)
    name = "SQLBOMD"
    println(name * " starts.")
    @printf(io, "%s\n%d\n%d\n", name, n_epoch, n_rate)
    len_output = n_epoch * n_rate
    output = init_output(len_output)
    to = TimerOutput()

    d::Int64 = size(ρ_true)[1]
    ρ_bar::Matrix{ComplexF64} = Matrix{ComplexF64}(I, d, d) / d
    ρ::Matrix{ComplexF64} = Matrix{ComplexF64}(I, d, d) / d
    ρ_inv::Matrix{ComplexF64} = Matrix{ComplexF64}(I, d, d) * d

    n_iter::Int64 = n_epoch * N
    period::Int64 = N ÷ n_rate
    @timeit to "iteration" begin
        idx = rand(1:N, n_iter)
        η = sqrt( d * log( n_iter ) )
        η = η / ( sqrt( n_iter ) + η )
    end
 
    @inbounds for iter = 1:n_iter
        @timeit to "iteration" begin
            # update
            grad::Matrix{ComplexF64} = - view(data, :, :, idx[iter]) / real(view(data, :, :, idx[iter]) ⋅ ρ)
            Λ_inv::Array{Float64, 1}, U::Matrix{ComplexF64} = eigen(Hermitian(ρ_inv + η * grad))
            Λ = log_barrier_projection(1 ./ Λ_inv, 1e-5)
            ρ = U * Diagonal(Λ) * adjoint(U)
            ρ_inv = U * Diagonal(1 ./ Λ) * adjoint(U)

            ρ_bar = (iter * ρ_bar + ρ) / (iter + 1.0)
        end

        if mod(iter, period) == 0
            λ = compute_λ(ρ_bar)
            update_output!(output, iter÷period, iter/N, fidelity(ρ_true, ρ_bar), f(λ),
                            TimerOutputs.time(to["iteration"]) * 1e-9)
            print_output(io, output, iter÷period, VERBOSE)
        end
    end

    return output 
end


function LB_SDA(n_epoch::Int64, n_rate::Int64)
    name = "1-sample LB-SDA"
    println(name * " starts.")
    @printf(io, "%s\n%d\n%d\n", name, n_epoch, n_rate)
    len_output = n_epoch * n_rate
    output = init_output(len_output)
    to = TimerOutput()

    d::Int64 = size(ρ_true)[1]
    ρ_bar::Matrix{ComplexF64} = Matrix{ComplexF64}(I, d, d) / d
    ρ::Matrix{ComplexF64} = Matrix{ComplexF64}(I, d, d) / d
    ∑grad::Matrix{ComplexF64} = zeros(ComplexF64, d, d)
    ∑dual_norm2::Float64 = 0
    
    n_iter::Int64 = n_epoch * N
    period::Int64 = N ÷ n_rate
    
    @timeit to "iteration" begin
        idx = rand(1:N, n_iter)  
    end
 
    @inbounds for iter = 1:n_iter
        @timeit to "iteration" begin

            grad::Matrix{ComplexF64} = - view(data, :, :, idx[iter]) / real(view(data, :, :, idx[iter]) ⋅ ρ_bar)

            # compute learning rates
            ∑dual_norm2 += dual_norm2(ρ, grad + α(ρ, grad) * Matrix{ComplexF64}(I, d, d))
            η = sqrt(d) / sqrt(4 * d + 1 + ∑dual_norm2)
            
            # update step
            ∑grad += grad
            Λ_inv, U = eigen(Hermitian(η * ∑grad))
            
            # projection step
            Λ = log_barrier_projection(1 ./ Λ_inv, 1e-5)
            ρ = U * Diagonal(Λ) * adjoint(U)

            ρ_bar = (iter * ρ_bar + ρ) / (iter + 1.0)
        end

        if mod(iter, period) == 0
            λ = compute_λ(ρ_bar)
            update_output!(output, iter÷period, iter/N, fidelity(ρ_true, ρ_bar), f(λ),
                            TimerOutputs.time(to["iteration"]) * 1e-9)
            print_output(io, output, iter÷period, VERBOSE)
        end
    end

    return output 
end


function d_sample_LB_SDA(n_epoch::Int64, n_rate::Int64)
    name = "d-sample LB-SDA"
    println(name * " starts.")
    @printf(io, "%s\n%d\n%d\n", name, n_epoch, n_rate)
    len_output = n_epoch * n_rate
    output = init_output(len_output)
    to = TimerOutput()

    d::Int64 = size(ρ_true)[1]
    ρ_bar::Matrix{ComplexF64} = Matrix{ComplexF64}(I, d, d) / d
    ρ::Matrix{ComplexF64} = Matrix{ComplexF64}(I, d, d) / d
    ∑grad::Matrix{ComplexF64} = zeros(ComplexF64, d, d)
    ∑dual_norm2::Float64 = 0
    batch_size::Int64 = d
    
    n_iter::Int64 = n_epoch * (N ÷ batch_size)
    period::Int64 = (N ÷ batch_size) ÷ n_rate
    
    @timeit to "iteration" begin
        idx = rand(1:N, (batch_size, n_iter))  
    end
 
    @inbounds for iter = 1:n_iter
        @timeit to "iteration" begin
            grad::Matrix{ComplexF64} = zeros(ComplexF64, d, d)
            @inbounds for j = 1:batch_size
                grad += - view(data, :, :, idx[j, iter]) / real(view(data, :, :, idx[j, iter]) ⋅ ρ_bar)
            end
            grad /= batch_size

            # compute learning rates
            ∑dual_norm2 += dual_norm2(ρ, grad + α(ρ, grad) * Matrix{ComplexF64}(I, d, d))
            η = sqrt(d) / sqrt(4 * d + 1 + ∑dual_norm2)
            
            # update step
            ∑grad += grad
            Λ_inv, U = eigen(Hermitian(η * ∑grad))
            
            # projection step
            Λ = log_barrier_projection(1 ./ Λ_inv, 1e-5)
            ρ = U * Diagonal(Λ) * adjoint(U)

            ρ_bar = (iter * ρ_bar + ρ) / (iter + 1.0)
        end

        if mod(iter, period) == 0
            λ = compute_λ(ρ_bar)
            update_output!(output, iter÷period, iter/(N÷batch_size), fidelity(ρ_true, ρ_bar), f(λ),
                            TimerOutputs.time(to["iteration"]) * 1e-9)
            print_output(io, output, iter÷period, VERBOSE)
        end
    end

    return output 
end