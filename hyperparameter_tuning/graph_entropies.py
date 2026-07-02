import sys
sys.path.insert(1, 'C:\\Users\\gevmo\\OneDrive\\Music\\Documents\\2026 - Summer\\Causal Inference\\quantum-latent-cause')

import quantum_causal_inference as qci
import numpy as np
import matplotlib.pyplot as plt

# Pure State (00 Bell State)
esti_state1 = np.array([[0.5, 0, 0, 0.5], 
                       [0, 0, 0, 0], 
                       [0, 0, 0, 0], 
                       [0.5, 0, 0, 0.5]])
# Evenly Distributed
esti_state2 = np.array([[0.25, 0, 0, 0], 
                       [0, 0.25, 0, 0], 
                       [0, 0, 0.25, 0], 
                       [0, 0, 0, 0.25]])
# Dephased GHZ
esti_state3 = np.array([[0.5, 0, 0, 0], 
                       [0, 0, 0, 0], 
                       [0, 0, 0, 0], 
                       [0, 0, 0, 0.5]])

dx = 2
dy = 2
dz = 2

problem = qci.QProblem(esti_state3, dx, dy, dz)

# penalties = [0, 0.1, 0.5, 0.8, 1, 1.5, 2, 3]
penalties = np.arange(0, 1000) / 1000
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

result = qci.QInferGraph(problem, penalties, tolerance, entrop_thresh, extern_thresh, dep_gate, 
                         smoothing, damping, log_reg, n, null_fam, sig_lvl)

print(result.result_message)

entropies_list = []
for candidate in result.candidate_entropies:
    entropies_list.append(candidate[3])

entropies_list.sort(reverse=True)

plt.plot(np.arange(len(entropies_list)), entropies_list, 'ro')
plt.show()