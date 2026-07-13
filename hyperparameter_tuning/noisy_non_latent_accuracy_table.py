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
p_vals = np.append(np.insert(np.arange(0.1, 1.0, 0.1).round(decimals=1), 0, 0.01), 0.99)

def get_xy(q, p):
    A = np.zeros((16, 16))
    for x in range(0, 4):
        for y in range(0, 4):
            prod = 1
            prod *= q if x&2 == 2 else (1-q)
            prod *= q if x&1 == 1 else (1-q)
            prod *= p if ((y&2) != (x&2)) else (1-p)
            prod *= p if ((y&1) != (x&1)) else (1-p)
            A[x*4 + y][x*4 + y] = prod
    return A

# x = get_xyz(0.4, 0.3, 0.2)
# print(qci.mi_xy(qci.tr_z(x, 4, 4, 4), 4, 4))

dx = 4
dy = 4
dz = 4


penalties = ih.penalties
tolerance = ih.tolerance
entrop_thresh = ih.entrop_thresh
extern_thresh = ih.extern_thresh
dep_gate = ih.dep_gate
smoothing = ih.smoothing
damping = ih.damping_noisy_lat
log_reg = ih.log_reg_noisy_lat
n = 100#ih.n

null_fam = []#[qci.QProblem(esti_state3, dx, dy, dz), qci.QProblem(esti_state1, dx, dy, dz)]
sig_lvl = ih.sig_lvl


results = []
df = pd.DataFrame()

for i in range(len(p_vals)):
    results.append([])
    problem = qci.QProblem(get_xy(q, p_vals[i]), dx, dy, dz)

    alpha = qci.getMinAlpha(problem, penalties, tolerance, entrop_thresh, extern_thresh, dep_gate, 
                                smoothing, damping, log_reg, n, null_fam, sig_lvl)
    if alpha is not None:
        results[i].append((np.ceil(alpha*100)/100))
        print(f"{p_vals[i]}: {(np.ceil(alpha*100)/100)}")
    else:
        results[i].append(None)
        print(f"{p_vals[i]}: {None}")


#Make table
for i in range(len(p_vals)):
    df.insert(df.shape[1], p_vals[i], results[i])

df = df.set_axis(["min_alpha"], axis = "index")

print(df)

tbl = plt.subplot(111, frame_on=False)
tbl.xaxis.set_visible(False)
tbl.yaxis.set_visible(False)
table(tbl, df, loc="center")
plt.savefig("noisy_latent_accuracy.png")
plt.show()
