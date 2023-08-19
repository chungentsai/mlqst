using Printf


function init_output(len::Int64)
    output = Dict()
    output["n_epoch"]      = zeros(Float64, len)
    output["fidelity"]     = zeros(Float64, len)
    output["fval"]         = zeros(Float64, len)
    output["elapsed_time"] = zeros(Float64, len)
    return output
end


function update_output!(output, index, t, fid, fval, time)
    output["n_epoch"][index] = t
    output["fidelity"][index] = fid
    output["fval"][index] = fval
    output["elapsed_time"][index] = time
end


function print_output(io, output, index)
    @printf(io, "%.1f\t%E\t%E\t%E\n", output["n_epoch"][index], output["fidelity"][index], output["elapsed_time"][index], output["fval"][index])
    @printf("%.1f\t%E\t%E\t%E\n", output["n_epoch"][index], output["fidelity"][index], output["elapsed_time"][index], output["fval"][index])
    flush(io)
end