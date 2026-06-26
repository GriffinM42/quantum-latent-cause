import quantum_causal_inference as qci
import numpy as np

esti_state1 = np.array([[0.5, 0, 0, 0.5], 
                       [0, 0, 0, 0], 
                       [0, 0, 0, 0], 
                       [0.5, 0, 0, 0.5]])
esti_state2 = np.array([[0.25, 0, 0, 0], 
                       [0, 0.25, 0, 0], 
                       [0, 0, 0.25, 0], 
                       [0, 0, 0, 0.25]])
esti_state3 = np.array([[0.5, 0, 0, 0], 
                       [0, 0, 0, 0], 
                       [0, 0, 0, 0], 
                       [0, 0, 0, 0.5]])

dx = 2
dy = 2
dz = 2

problem = qci.QProblem(esti_state3, dx, dy, dz)

penalties = [0, 0.1, 0.5, 0.8, 1, 1.5, 2, 3]
tolerance = 0.5
entrop_thresh = 0.8
extern_thresh = 1
dep_gate = 0.2
smoothing = 0.01
damping = 0.2
log_reg = 0.2
n = 1000
null_fam = [qci.QProblem(esti_state3, dx, dy, dz), qci.QProblem(esti_state1, dx, dy, dz)]
sig_lvl = 0.2

result = qci.QInferGraph(problem, penalties, tolerance, entrop_thresh, extern_thresh, dep_gate, 
                         smoothing, damping, log_reg, n, null_fam, sig_lvl)

print(result)


