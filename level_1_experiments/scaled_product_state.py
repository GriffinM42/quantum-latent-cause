import sys
sys.path.insert(1, 'C:\\Users\\gevmo\\OneDrive\\Music\\Documents\\2026 - Summer\\Causal Inference\\quantum-latent-cause')

import lib.quantum_causal_inference as qci
import lib.inference_hyperparams as ih
import numpy as np
import math

dx = 8
dy = 8
dz = 8

esti_state = np.empty([dx*dy, dx*dy])
esti_state.fill(1/(dx*dy))

print(qci.mi_xy(esti_state, dx, dy))
problem = qci.QProblem(esti_state, dx, dy, dz)

penalties = ih.penalties
tolerance = ih.tolerance * (max(math.log2(dx), math.log2(dy)))
entrop_thresh = ih.entrop_thresh
extern_thresh = ih.extern_thresh
dep_gate = ih.dep_gate * (max(math.log2(dx), math.log2(dy)))
smoothing = ih.smoothing
damping = ih.damping
log_reg = ih.log_reg
n = ih.n

null_fam = []
sig_lvl = ih.sig_lvl

print(esti_state)
print("Smooth MI: ", qci.mi_xy(((1-smoothing)*esti_state) + ((smoothing/(dx*dy))*np.eye(dx*dy)), dx, dy))


result = qci.QInferGraph(problem, penalties, tolerance, entrop_thresh, extern_thresh, dep_gate, 
                         smoothing, damping, log_reg, n, null_fam, sig_lvl, True)

print(result.result_message)