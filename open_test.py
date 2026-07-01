import numpy as np
import quantum_causal_inference as qci
import matplotlib.pyplot as plt

A = np.zeros((64, 64))

q = 0.4
p1 = 0.1
p2 = 0.2

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

print(A)

prob = qci.tr_z(A, 4, 4, 4)
print(prob)

dx = 4
dy = 4
dz = 4

problem = qci.QProblem(prob, dx, dy, dz)

# penalties = [0, 0.1, 0.5, 0.8, 1, 1.5, 2, 3]
penalties = np.arange(0, 100) / 100
tolerance = 0.05
entrop_thresh = 0.8
extern_thresh = None
dep_gate = 0
smoothing = 0.01
damping = 0.2
log_reg = 0.2
n = 1000

null_fam = []#[qci.QProblem(esti_state3, dx, dy, dz), qci.QProblem(esti_state1, dx, dy, dz)]
sig_lvl = 0.2

result = qci.QInferGraph(problem, penalties, tolerance, entrop_thresh, extern_thresh, dep_gate, 
                         smoothing, damping, log_reg, n, null_fam, sig_lvl)

print(result.result_message)

if result.candidate_entropies is not None and len(result.candidate_entropies) > 0:
    entropies_list = []
    for candidate in result.candidate_entropies:
        entropies_list.append(candidate[3])

    entropies_list.sort(reverse=True)

    plt.plot(np.arange(len(entropies_list)), entropies_list, 'ro')
    plt.show()