import sys
sys.path.insert(1, 'C:\\Users\\gevmo\\OneDrive\\Music\\Documents\\2026 - Summer\\Causal Inference\\quantum-latent-cause')

import lib.quantum_causal_inference as qci
import lib.inference_hyperparams as ih
import numpy as np
import math

dx = 2
dy = 2
dz = 2

esti_state = np.array([[0.5, 0, 0, 0.5], 
                       [0, 0, 0, 0], 
                       [0, 0, 0, 0], 
                       [0.5, 0, 0, 0.5]])

ux = np.array([[1, 1], [1, -1]])/math.sqrt(2)
uy = np.array([[0, complex(0, -1)], [complex(0, 1), 0]])

operator = np.kron(ux, uy)

problem = qci.QProblem(esti_state, dx, dy, dz)
problem_alt = qci.QProblem(np.matmul(operator, np.matmul(esti_state, np.conjugate(np.transpose(operator)))), dx, dy, dz)

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
                         smoothing, damping, log_reg, n, null_fam, sig_lvl, True)

print(result.result_message)

result_alt = qci.QInferGraph(problem_alt, penalties, tolerance, entrop_thresh, extern_thresh, dep_gate, 
                         smoothing, damping, log_reg, n, null_fam, sig_lvl, True)

print(result_alt.result_message)