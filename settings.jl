# problem parameters
# to change the generation of the true quantum state, please check main.jl
const q = 6               # num qubits
const Q = 4               # two outcomes
const N_EPOCH_S = 200     # num. epochs
const N_RATE_S = 1
const N_EPOCH_B = 1600
const N_RATE_B = 1        # record values every (numEpochs ÷ rate) iteration
                          # for stochastic algorithms only
                          # also used in myPlot()
const N = Q ^ q * 100     # sample size
const M = Q ^ q           # num observables
const d = 2 ^ q           # dimension
const VERBOSE = false

# algorithms settings
# implemented algorithms:
# stochastic algorithms: d_sample_LB_SDA, LB_SDA, SQLBOMD, SQSB
# batch algorithms: DA, BPG, QEM, RρR, M_FW

batch_algs = [BPG, QEM, FW]
stochastic_algs = [SQSB, SQLBOMD, LB_SDA, d_sample_LB_SDA]