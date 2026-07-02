import sys
sys.path.insert(1, 'C:\\Users\\gevmo\\OneDrive\\Music\\Documents\\2026 - Summer\\Causal Inference\\quantum-latent-cause')

import quantum_causal_inference as qci
import numpy as np

dx = 2
dy = 2
dz = 2

esti_state = np.array([[0.5, 0, 0, 0], 
                       [0, 0, 0, 0], 
                       [0, 0, 0, 0], 
                       [0, 0, 0, 0.5]])

problem = qci.QProblem(esti_state, dx, dy, dz)

penalties = np.arange(0.0, 1.0, 0.01)
tolerance = 0.2
entrop_thresh = 0.8
extern_thresh = None
dep_gate = 0.2
smoothing = 0.01
damping = 0.2
log_reg = 0.2
n = 1000

null_fam = []
sig_lvl = 0.2

result = qci.QInferGraph(problem, penalties, tolerance, entrop_thresh, extern_thresh, dep_gate, 
                         smoothing, damping, log_reg, n, null_fam, sig_lvl)

print(result.result_message)