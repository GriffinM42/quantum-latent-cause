import sys
sys.path.insert(1, 'C:\\Users\\gevmo\\OneDrive\\Music\\Documents\\2026 - Summer\\Causal Inference\\quantum-latent-cause')

import lib.quantum_causal_inference as qci
import lib.inference_hyperparams as ih
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

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
penalties = ih.penalties
tolerance = ih.tolerance
entrop_thresh = ih.entrop_thresh
extern_thresh = ih.extern_thresh
dep_gate = ih.dep_gate
smoothing = ih.smoothing
damping = ih.damping_ghz
log_reg = ih.log_reg_ghz
n = ih.n

null_fam = [qci.QProblem(esti_state3, dx, dy, dz), qci.QProblem(esti_state1, dx, dy, dz)]
sig_lvl = ih.sig_lvl

result = qci.QInferGraph(problem, penalties, tolerance, entrop_thresh, extern_thresh, dep_gate, 
                         smoothing, damping, log_reg, n, null_fam, sig_lvl, True)

print(result.result_message)

cmi_list = []
sz_list = []
for witness in result.witnesses:
    cmi_list.append(witness.cmi)
    sz_list.append(witness.entrop_z)

red_dot, = plt.plot(cmi_list, sz_list, 'ro', label="witness")

green_line, = plt.plot([ih.tolerance, ih.tolerance], [0, np.max(plt.ylim())], 'g-', linewidth=1.5, label="conditional mutual information threshold")

# green_line = mlines.Line2D([], [], color='g', linewidth=1.5, linestyle='-', label="conditional mutual information threshold")    

plt.legend(handles=[red_dot, green_line], fontsize=12)

plt.xlabel("MI(X;Y|Z)", fontsize=14)
plt.ylabel("S(Z)", fontsize=14)
plt.title("Conditional Mutual Information vs. Entropy Curve", fontsize=16)
plt.savefig("dephased_ghz_cmi_vs_sz.png")
plt.show()
