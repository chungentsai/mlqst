include("./myplot.jl")

using MKL
using TimerOutputs;
using Random
using Dates

include("./functions.jl")
include("./utils.jl")
include("./batch_algorithms.jl")
include("./stochastic_algorithms.jl")
include("./settings.jl")

BLAS.set_num_threads(8);

mkpath("./records")
const filename = "./records/" * Dates.format(now(), "yyyy-mm-dd-HH-MM-SS")
const io       = open(filename, "a")

const to = TimerOutput();
reset_timer!(to)

@timeit to "Setup" begin
    const ρ_true = w_state(q) # true density matrix
    const POVM = pauli_povm(q, Q)
    const idx_obs = rand(1: M, N)
    const outcomes = measure(ρ_true, POVM, idx_obs)
    const data = generate_data(POVM, idx_obs, outcomes)
end


∇f(λ) = ∇f(data, λ)
compute_λ(ρ) = compute_λ(data, ρ)
run_alg(alg, n_epoch, n_rate) = alg(n_epoch, n_rate, io, ρ_true, N, f, ∇f, compute_λ, VERBOSE)


try
    global results = Dict()

    for alg in stochastic_algs
        results[alg] = run_alg(alg, N_EPOCH_S, N_RATE_S)
    end

    for alg in batch_algs
        results[alg] = run_alg(alg, N_EPOCH_B, N_RATE_B)
    end

finally
    close(io)

end


#myPlot(filename)