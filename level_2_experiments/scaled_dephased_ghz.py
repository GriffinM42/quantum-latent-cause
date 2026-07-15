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
qb_z = 1

dx = int(math.exp2(qb_x))
dy = int(math.exp2(qb_y))
dz = int(math.exp2(qb_z))

ideal_state = np.zeros((dx*dy, dx*dy)) 
ideal_state[0][0] = 0.5
ideal_state[dx*dy-1][dx*dy-1] = 0.5

def get_ghz_3():
    q = QuantumRegister(qb_x + qb_y + 2)
    b = ClassicalRegister(qb_x + qb_y + 2)
    circ = QuantumCircuit(q, b)
    circ.h(q[0])

    for i in range(qb_x + qb_y):
        circ.cx(q[0], q[i+1])
    return circ

def tomo_circ(get_circ):
    aer = AerSimulator()
    exp = StateTomography(get_circ(), measurement_indices=np.arange(0, qb_x + qb_y, 1).data)
    data = exp.run(backend=aer).block_for_results()
    df = data.analysis_results(dataframe=True)
    p_xy = df['value'].values[0].data
    return p_xy

esti_state = tomo_circ(get_ghz_3)

print(esti_state)
print("Tomography trace distance: ", qci.trace_dist(esti_state, ideal_state))

problem = qci.QProblem(esti_state, dx, dy, dz)

penalties = ih.penalties
tolerance = ih.tolerance
entrop_thresh = ih.entrop_thresh
extern_thresh = ih.extern_thresh
dep_gate = ih.dep_gate
smoothing = ih.smoothing
damping = ih.damping_ghz
log_reg = ih.log_reg_ghz
n = ih.n

null_fam = []
sig_lvl = ih.sig_lvl

result = qci.QInferGraph(problem, penalties, tolerance, entrop_thresh, extern_thresh, dep_gate, 
                         smoothing, damping, log_reg, n, null_fam, sig_lvl, True)

print(result.result_message)
