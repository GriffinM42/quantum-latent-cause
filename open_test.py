import numpy as np
import numpy.linalg as lin
import lib.quantum_causal_inference as qci
import lib.inference_hyperparams as ih
import matplotlib.pyplot as plt

A = np.zeros((64, 64))

q = 0.4
p1 = 0.3
p2 = 0.4

for x in range(0, 4):
    for y in range(0, 4):
        for z in range(0, 4):
            prod = 1
            prod *= q if (z & 2) == 2 else (1-q)
            prod *= q if (z & 1) == 1 else (1-q)
            prod *= p1 if ((z&2) != (x&2)) else (1-p1)
            prod *= p1 if ((z&1) != (x&1)) else (1-p1)
            prod *= p2 if ((z&2) != (y&2)) else (1-p2)
            prod *= p2 if ((z&1) != (y&1)) else (1-p2)
            A[x*16 + y*4 + z][x*16 + y*4 + z] = prod

print(A)
print(np.trace(A))
prob = qci.tr_z(A, 4, 4, 4)
print(np.trace(prob))

dx = 4
dy = 4
dz = 4

p_x = np.trace(prob.reshape(dx, dy, dx, dy), axis1=1, axis2=3)
p_y = np.trace(prob.reshape(dx, dy, dx, dy), axis1=0, axis2=2)

print(qci.mi_xy(prob, dx, dy))

problem = qci.QProblem(prob, dx, dy, dz)

# penalties = [0, 0.1, 0.5, 0.8, 1, 1.5, 2, 3]
penalties = ih.penalties
tolerance = ih.tolerance
entrop_thresh = ih.entrop_thresh
extern_thresh = ih.extern_thresh
dep_gate = ih.dep_gate
smoothing = ih.smoothing
damping = ih.damping
log_reg = ih.log_reg
n = 100

null_fam = []#[qci.QProblem(esti_state3, dx, dy, dz), qci.QProblem(esti_state1, dx, dy, dz)]
sig_lvl = ih.sig_lvl

result = qci.QInferGraph(problem, penalties, tolerance, entrop_thresh, extern_thresh, dep_gate, 
                         smoothing, damping, log_reg, n, null_fam, sig_lvl)

print(result.result_message)

print(qci.tr_xy(result.candidate_entropies[0].state, dx, dy, dz))
print(lin.eig(qci.tr_xy(result.candidate_entropies[0].state, dx, dy, dz))[0])
print(np.trace(result.candidate_entropies[0].state))

if result.candidate_entropies is not None and len(result.candidate_entropies) > 0:
    entropies_list = []
    for candidate in result.candidate_entropies:
        entropies_list.append(candidate.entrop_z)

    entropies_list.sort(reverse=True)

    plt.plot(np.arange(len(entropies_list)), entropies_list, 'ro')
    plt.show()
