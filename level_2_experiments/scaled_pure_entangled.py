import sys
sys.path.insert(1, 'C:\\Users\\gevmo\\OneDrive\\Music\\Documents\\2026 - Summer\\Causal Inference\\quantum-latent-cause')

import lib.quantum_causal_inference as qci
import lib.inference_hyperparams as ih
import numpy as np
import math

from qiskit import QuantumCircuit
from qiskit.circuit import QuantumRegister, ClassicalRegister
from qiskit_experiments.library import StateTomography

from qiskit_aer import AerSimulator

qb_x = 3
qb_y = 3
qb_z = 3

dx = int(math.exp2(qb_x))
dy = int(math.exp2(qb_y))
dz = int(math.exp2(qb_z))

ideal_state = np.zeros((dx*dy, dx*dy)) 
ideal_state[0][0] = 0.5
ideal_state[0][dx*dy-1] = 0.5
ideal_state[dx*dy-1][0] = 0.5
ideal_state[dx*dy-1][dx*dy-1] = 0.5

def get_bell_00():
    q = QuantumRegister(qb_x + qb_y)
    b = ClassicalRegister(qb_x + qb_y)
    circ = QuantumCircuit(q, b)
    circ.h(q[0])
    for i in range(qb_x + qb_y-1):
        circ.cx(q[0], q[i+1])
    return circ

def tomo_circ(get_circ):
    aer = AerSimulator()
    exp = StateTomography(get_circ())
    data = exp.run(backend=aer).block_for_results()
    df = data.analysis_results(dataframe=True)
    p_xy = df['value'].values[0].data
    return p_xy

esti_state = tomo_circ(get_bell_00)

print(esti_state)
print("Tomography trace distance: ", qci.trace_dist(esti_state, ideal_state))

problem = qci.QProblem(esti_state, dx, dy, dz)

penalties = ih.penalties
tolerance = ih.tolerance + (max(qb_x, qb_y)+1) * ih.smoothing + 0.05
entrop_thresh = ih.entrop_thresh
extern_thresh = ih.extern_thresh
dep_gate = ih.dep_gate + (max(qb_x, qb_y)+1) * ih.smoothing + 0.05
smoothing = ih.smoothing
damping = ih.damping
log_reg = ih.log_reg
n = ih.n

null_fam = []
sig_lvl = ih.sig_lvl

result = qci.QInferGraph(problem, penalties, tolerance, entrop_thresh, extern_thresh, dep_gate, 
                         smoothing, damping, log_reg, n, null_fam, sig_lvl, True)

print(result.result_message)
