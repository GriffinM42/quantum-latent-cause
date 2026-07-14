import sys
sys.path.insert(1, 'C:\\Users\\gevmo\\OneDrive\\Music\\Documents\\2026 - Summer\\Causal Inference\\quantum-latent-cause')

import lib.quantum_causal_inference as qci

from qiskit_ibm_runtime import QiskitRuntimeService
import lib.inference_hyperparams as ih
import numpy as np
import math

from qiskit import QuantumCircuit
from qiskit.circuit import QuantumRegister, ClassicalRegister
from qiskit_experiments.library import StateTomography

ideal_state = np.zeros((8, 8))

a = math.sqrt(1/4)
b = math.sqrt(3/4)

dx = 2
dy = 2
dz = 2

for i in range(8):
    for j in range(8):
        prod = 1
        if i == 0 or i == 3 or i == 4 or i == 7:
            prod *= a
        elif i == 1 or i == 2:
            prod *= b
        else:
            prod *= -b
        if j == 0 or j == 3 or j == 4 or j == 7:
            prod *= np.conjugate(a)
        elif j == 1 or j == 2:
            prod *= np.conjugate(b)
        else:
            prod *= np.conjugate(-b)
        ideal_state[i][j] = prod

ideal_state = 0.25 * ideal_state
ideal_state = qci.tr_z(ideal_state, dx, dy, dz)

def get_teleport():
    q = QuantumRegister(3)
    b = ClassicalRegister(3)
    circ = QuantumCircuit(q, b)
    circ.ry(2 * math.pi/3, q[2])
    circ.h(q[1])
    circ.cx(q[1], q[0])

    circ.cx(q[2], q[1])
    circ.h(q[2])
    return circ

def tomo_circ(get_circ):
    service = QiskitRuntimeService()
    backend = service.least_busy(simulator=False, operational=True)
    exp = StateTomography(get_circ(), measurement_indices=[0, 1])
    data = exp.run(backend=backend).block_for_results()
    df = data.analysis_results(dataframe=True)
    p_xy = df['value'].values[0].data
    return p_xy

dx = 2
dy = 2
dz = 2

esti_state = tomo_circ(get_teleport)

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
                         smoothing, damping, log_reg, n, null_fam, sig_lvl)

print(result.result_message)
