# problem parameters
# to change the generation of the true quantum state, please check main.jl
const q = 2               # num qubits
const Q = 4               # two outcomes
const N_EPOCH_S = 200     # num. epochs
const N_RATE_S = 1
const N_EPOCH_B = 600
const N_RATE_B = 1        # record values every (numEpochs ÷ rate) iteration
const N = Q ^ q * 100     # sample size
const M = Q ^ q           # num observables
const d = 2 ^ q           # dimension
const VERBOSE = false

# algorithms settings
batch_algs = [BPG, QEM, FW, EMD, iMLE, diluted_iMLE]
stochastic_algs = [SQSB, SQLBOMD, LB_SDA, d_sample_LB_SDA]