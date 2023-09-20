using PyPlot
using DelimitedFiles


function myPlot(filename)
    #rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
    #rcParams["figure.dpi"] = 300
    #rcParams["font.size"] = 15
    #rcParams["legend.fontsize"] = 15
    N_EPOCH = 200

    A = readdlm(filename, '\t', Any, '\n')
    n_line = size(A)[1]
    algs = String[]
    results = Dict()
    approx_opt = Inf

    i = 1
    while i <= n_line
        alg = A[i, 1]
        n_epoch = A[i+1, 1]
        n_rate = A[i+2, 1]

        push!(algs, alg)
        results[alg] = Dict()
        s = i+2+1
        t = i+2+n_epoch*n_rate
        results[alg]["n_epoch"] = A[s:t, 1]
        results[alg]["fidelity"] = A[s:t, 2]
        results[alg]["elapsed_time"] = A[s:t, 3]
        results[alg]["fval"] = A[s:t, 4]
        approx_opt = minimum([minimum(results[alg]["fval"]), approx_opt])
        i = t+1
    end

    path = replace(filename, "records"=> "figures") * "/"
    mkpath(path)
    close("all")

    figure(1)
    for alg in algs
        plot(results[alg]["n_epoch"], results[alg]["fidelity"], linewidth=2)
        hold
    end
    legend(algs)
    xlabel("Number of Epochs")
    ylabel("Fidelity")
    xlim([0, N_EPOCH])
    ylim([0, 1])
    grid("on")
    savefig(path * "/epoch-fidelity.png")


    figure(2)
    for alg in algs
        semilogy(results[alg]["n_epoch"], results[alg]["fval"] .- approx_opt, linewidth=2)
        hold
    end
    legend(algs)
    xlabel("Number of epochs")
    ylabel("Approximate optimization error")
    xlim([0, N_EPOCH])
    #ylim([1e-6, 1e-2])
    grid("on")
    savefig(path * "/epoch-error.png")


    figure(3)
    for alg in algs
        semilogx(results[alg]["elapsed_time"], results[alg]["fidelity"], linewidth=2)
        hold
    end
    legend(algs)
    xlabel("Elapsed time (seconds)")
    ylabel("Fidelity")
    #xlim([1, 3*1e5])
    ylim([0, 1])
    grid("on")
    savefig(path * "/time-fidelity.png")


    figure(4)
    for alg in algs
        loglog(results[alg]["elapsed_time"], results[alg]["fval"] .- approx_opt, linewidth=2)
        hold
    end
    legend(algs)
    xlabel("Elapsed time (seconds)")
    ylabel("Approximate optimization error")
    #xlim([1, 3*1e5])
    #ylim([1e-6, 1e-2])
    grid("on")
    savefig(path * "/time-error.png")

end