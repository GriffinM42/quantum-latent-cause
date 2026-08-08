import sys
sys.path.insert(1, 'C:\\Users\\gevmo\\OneDrive\\Music\\Documents\\2026 - Summer\\Causal Inference\\quantum-latent-cause')

import lib.quantum_causal_inference as qci
import numpy as np
import math

def get_kron_list(q_bits: int, noise_factor: float):
    k0 = math.sqrt(1-((3*noise_factor)/4))*np.eye(2)
    k1 = math.sqrt(noise_factor/4)*np.array([[0, 1], [1, 0]])
    k2 = math.sqrt(noise_factor/4)*np.array([[0, complex(0, -1)], [complex(0, 1), 0]])
    k3 = math.sqrt(noise_factor/4)*np.array([[1, 0], [0, -1]])

    if q_bits == 1:
        return [k0, k1, k2, k3]

    ops = get_kron_list(q_bits-1, noise_factor)
    operators = []

    for op in ops:
        operators.append(np.kron(op, k0))
        operators.append(np.kron(op, k1))
        operators.append(np.kron(op, k2))
        operators.append(np.kron(op, k3))

    return operators
    

def get_noisy_control(q_bits: int, noise_factor: float):
    dim = pow(2, q_bits)

    operators = get_kron_list(q_bits, noise_factor)
    mat = np.zeros((dim*dim, dim*dim), dtype='complex128')
    for i in range(dim):
        for j in range(dim):
            basis = np.zeros((dim, dim))
            sum = np.zeros((dim, dim), dtype='complex128')
            basis[i][j] = 1
            for op in operators:
                sum += np.matmul(op, np.matmul(basis, np.conjugate(np.transpose(op))))
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
    problem = qci.QProblem(get_noisy_control(1, factor), dx, dy, dz)

    result = qci.QInferGraph(problem, penalties, tolerance, entrop_thresh, extern_thresh, dep_gate, 
                            smoothing, damping, log_reg, n, null_fam, sig_lvl, True)

    print(f"{factor}: {result.result_message}")
