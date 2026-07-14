import sys
sys.path.insert(1, 'C:\\Users\\gevmo\\OneDrive\\Music\\Documents\\2026 - Summer\\Causal Inference\\quantum-latent-cause')

import lib.quantum_causal_inference as qci
import lib.inference_hyperparams as ih
import numpy as np
import math

dx = 2
dy = 2
dz = 2

esti_state = np.zeros((8, 8))

a = math.sqrt(1/4)
b = math.sqrt(3/4)

for i in range(8):
    for j in range(8):
        prod = 1
        if i == 0 or i == 3 or i == 4 or i == 7:
            prod *= a
        elif i == 1 or i == 2:
            prod *= b
        else:
            prod *= -b
        if j == 0 or j == 3 or j == 4 or j == 7:
            prod *= np.conjugate(a)
        elif j == 1 or j == 2:
            prod *= np.conjugate(b)
        else:
            prod *= np.conjugate(-b)
        esti_state[i][j] = prod

esti_state = 0.25 * esti_state
esti_state = qci.tr_x(esti_state, dx, dy, dz)

problem = qci.QProblem(esti_state, dx, dy, dz)

penalties = ih.penalties
tolerance = ih.tolerance
entrop_thresh = ih.entrop_thresh
extern_thresh = ih.extern_thresh
dep_gate = ih.dep_gate
smoothing = ih.smoothing
damping = ih.damping
log_reg = ih.log_reg
n = ih.n

null_fam = []
sig_lvl = ih.sig_lvl



result = qci.QInferGraph(problem, penalties, tolerance, entrop_thresh, extern_thresh, dep_gate, 
                         smoothing, damping, log_reg, n, null_fam, sig_lvl)

print(result.result_message)