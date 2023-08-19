using PyPlot
using DelimitedFiles


function myPlot(filename, n_epoch_s, n_epoch_b, n_rate_s, stochastic_algs, batch_algs)
    #rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
    #rcParams["figure.dpi"] = 300
    #rcParams["font.size"] = 14
    #rcParams["legend.fontsize"] = 14

    A = readdlm(filename, '\t', Float64, '\n')

    approxOpt = minimum(A[:, 4])
    algs = vcat(stochastic_algs, stochastic_algs)
    label = string.(algs)
    results = Dict()

    s = 1
    t = 1
    for alg in algs
        if alg in stochastic_algs
            t = s + n_rate_s * n_epoch_s - 1
        elseif alg in batch_algs
            t = s + n_epoch_b - 1
        end
        results[alg] = Dict()
        results[alg]["n_epoch"] = A[s:t, 1]
        results[alg]["fidelity"] = A[s:t, 2]
        results[alg]["elapsed_time"] = A[s:t, 3]
        results[alg]["fval"] = A[s:t, 4]
        s = t + 1
    end

    foldername = "./figures/" * filename * "/"
    mkpath(foldername)
    close("all")


    figure(1)
    for alg in algs
        plot(results[alg]["n_epoch"], results[alg]["fidelity"], linewidth=2)
        hold
    end
    legend(label)
    xlabel("Number of Epochs")
    ylabel("Fidelity")
    xlim([0, n_epoch_s])
    ylim([0, 1])
    grid("on")
    savefig(foldername * "/epoch-fidelity.png")


    figure(2)
    for alg in algs
        semilogy(results[alg]["n_epoch"], results[alg]["fval"] .- approxOpt, linewidth=2)
        hold
    end
    legend(label)
    xlabel("Number of epochs")
    ylabel("Approximate optimization error")
    xlim([0, n_epoch_s])
    ylim([1e-5, 1e-1])
    grid("on")
    savefig(foldername * "/epoch-error.png")


    figure(3)
    for alg in algs
        semilogx(results[alg]["elapsed_time"], results[alg]["fidelity"], linewidth=2)
        hold
    end
    legend(label)
    xlabel("Elapsed time (seconds)")
    ylabel("Fidelity")
    xlim([1, 3*1e5])
    ylim([0, 1])
    grid("on")
    savefig(foldername * "/time-fidelity.png")


    figure(4)
    for alg in algs
        loglog(results[alg]["elapsed_time"], results[alg]["fval"] .- approxOpt, linewidth=2)
        hold
    end
    legend(label)
    xlabel("Elapsed time (seconds)")
    ylabel("Approximate optimization error")
    ylim([1e-5, 1e-1])
    grid("on")
    savefig(foldername * "/time-error.png")

end