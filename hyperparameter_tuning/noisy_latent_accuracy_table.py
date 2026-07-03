import sys
sys.path.insert(1, 'C:\\Users\\gevmo\\OneDrive\\Music\\Documents\\2026 - Summer\\Causal Inference\\quantum-latent-cause')

import numpy as np
import lib.quantum_causal_inference as qci
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import table

q = 0.4
p1_vals = np.arange(0.1, 1.0, 0.1).round(decimals=1)
p2_vals = np.arange(0.1, 1.0, 0.1).round(decimals=1)

accurate_result = "latent Markovizing witness"

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


penalties = np.arange(0, 10) / 10
tolerance = 0.05
entrop_thresh = 0.9
extern_thresh = None
dep_gate = 0
smoothing = 0.01
damping = 0.9396656310614254
log_reg = 0.009754753112145118
n = 100

null_fam = []#[qci.QProblem(esti_state3, dx, dy, dz), qci.QProblem(esti_state1, dx, dy, dz)]
sig_lvl = 0.2


results = []
df = pd.DataFrame()

for i in range(len(p1_vals)):
    results.append([])
    for j in range(len(p2_vals)):
        problem = qci.QProblem(qci.tr_z(get_xyz(q, p1_vals[i], p2_vals[j]), dx, dy, dz), dx, dy, dz)
        res = qci.QInferGraph(problem, penalties, tolerance, entrop_thresh, extern_thresh, dep_gate, 
                         smoothing, damping, log_reg, n, null_fam, sig_lvl).result_message
        if accurate_result == res[:len(accurate_result)]:
            results[i].append('T')
        else:
            results[i].append('F')


#Make table
for i in range(len(p1_vals)):
    df.insert(df.shape[1], p1_vals[i], results[i])

df = df.set_axis(p2_vals, axis = "index")

print(df)

tbl = plt.subplot(111, frame_on=False)
tbl.xaxis.set_visible(False)
tbl.yaxis.set_visible(False)
table(tbl, df, loc="center")
plt.savefig("noisy_latent_accuracy.png")
