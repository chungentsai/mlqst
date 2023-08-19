# problem parameters
# to change the generation of the true quantum state, please check main.jl
const q = 4               # num qubits
const Q = 4               # two outcomes
const N_EPOCH_S = 200     # num. epochs
const N_RATE_S = 2
const N_EPOCH_B = 600
const N_RATE_B = 1        # record values every (numEpochs ÷ rate) iteration
                          # for stochastic algorithms only
                          # also used in myPlot()

const N = Q ^ q * 100     # sample size
const M = Q ^ q           # num observables
const d = 2 ^ q           # dimension

# algorithms settings
# implemented algorithms:
# stochastic algorithms: SDA, SMD, SQSB
# batch algorithms: RρR, QEM, M_FW, DA

stochastic_algs = [SDA, SMD, SQSB]
batch_algs = [DA, QEM, RρR, M_FW]