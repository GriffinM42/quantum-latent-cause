import sys
sys.path.insert(1, 'C:\\Users\\gevmo\\OneDrive\\Music\\Documents\\2026 - Summer\\Causal Inference\\quantum-latent-cause')

import numpy as np
import lib.quantum_causal_inference as qci
import lib.inference_hyperparams as ih
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import table

q = 0.4
p1 = 0.1
p2 = 0.2

def get_xyz(q, p1, p2):
    A = np.zeros((64, 64))
    for x in range(0, 4):
        for y in range(0, 4):
            for z in range(0, 4):
                prod = 1
                prod *= q if (z & 2) == 2 else (1-q)
                prod *= q if (z & 1) == 2 else (1-q)
                prod *= p1 if ((z&2) != (x&2)) else (1-p1)
                prod *= p1 if ((z&1) != (x&1)) else (1-p1)
                prod *= p2 if ((z&2) != (y&2)) else (1-p2)
                prod *= p2 if ((z&1) != (y&1)) else (1-p2)
                A[x*16 + y*4 + z][x*16 + y*4 + z] = prod
    return A

# x = get_xyz(0.4, 0.3, 0.2)
# print(qci.mi_xy(qci.tr_z(x, 4, 4, 4), 4, 4))

dx = 4
dy = 4
dz = 4

problem = qci.QProblem(qci.tr_z(get_xyz(q, p1, p2), dx, dy, dz), dx, dy, dz)


penalties = ih.penalties
tolerance = ih.tolerance
entrop_thresh = ih.entrop_thresh
extern_thresh = ih.extern_thresh
dep_gate = ih.dep_gate
smoothing = ih.smoothing
damping = ih.damping_noisy_lat
log_reg = ih.log_reg_noisy_lat
n = ih.n

null_fam = []#[qci.QProblem(esti_state3, dx, dy, dz), qci.QProblem(esti_state1, dx, dy, dz)]
sig_lvl = ih.sig_lvl



result = qci.QInferGraph(problem, penalties, tolerance, entrop_thresh, extern_thresh, dep_gate, 
                         smoothing, damping, log_reg, n, null_fam, sig_lvl)


print(result.result_message)

entropies_list = []
for candidate in result.candidate_entropies:
    entropies_list.append(candidate[3])

entropies_list.sort(reverse=True)

plt.plot(np.arange(len(entropies_list)), entropies_list, 'ro')
plt.savefig("noisy_latent_entropies.png")
plt.show()
