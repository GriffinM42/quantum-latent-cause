import sys
sys.path.insert(1, 'C:\\Users\\gevmo\\OneDrive\\Music\\Documents\\2026 - Summer\\Causal Inference\\quantum-latent-cause')

import lib.quantum_causal_inference as qci
import numpy as np
import math


def get_dephase_control(q_bits: int, dephase_factor: float):
    dim = pow(2, q_bits)
    
    mat = np.zeros((dim*dim, dim*dim), dtype='complex128')
    for i in range(dim):
        for j in range(dim):
            basis = np.zeros((dim, dim))
            sum = np.zeros((dim, dim), dtype='complex128')
            basis[i][j] = 1

            if i != j:
                sum[i][j] = 1-dephase_factor
            else:
                sum[i][j] = 1
            
            mat += np.kron(basis, sum)/dim
    
    return mat

dephase_factors = np.arange(0, 1.01, 0.01)

dx = 2
dy = 2
dz = 2

# penalties = [0, 0.1, 0.5, 0.8, 1, 1.5, 2, 3]
penalties = np.arange(0, 100) / 100
tolerance = 0.2
entrop_thresh = 0.8
extern_thresh = 1
dep_gate = 0.2
smoothing = 0.01
damping = 0.2
log_reg = 0.2
n = 100

null_fam = []#[qci.QProblem(esti_state3, dx, dy, dz), qci.QProblem(esti_state1, dx, dy, dz)]
sig_lvl = 0.2

for factor in dephase_factors:
    problem = qci.QProblem(get_dephase_control(1, factor), dx, dy, dz)

    result = qci.QInferGraph(problem, penalties, tolerance, entrop_thresh, extern_thresh, dep_gate, 
                            smoothing, damping, log_reg, n, null_fam, sig_lvl, True)

    print(f"{factor}: {result.result_message}")
