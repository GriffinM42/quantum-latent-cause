import sys
sys.path.insert(1, 'C:\\Users\\gevmo\\OneDrive\\Music\\Documents\\2026 - Summer\\Causal Inference\\quantum-latent-cause')

import quantum_causal_inference as qci

from qiskit_ibm_runtime import QiskitRuntimeService

from qiskit import QuantumCircuit
from qiskit.circuit import QuantumRegister, ClassicalRegister
from qiskit_experiments.library import StateTomography

from qiskit_aer import AerSimulator
from qiskit.primitives import BackendSamplerV2

import numpy as np

def get_ghz_3():
    q = QuantumRegister(3)
    b = ClassicalRegister(3)
    circ = QuantumCircuit(q, b)
    circ.h(q[0])
    circ.cx(q[0], q[1])
    circ.cx(q[0], q[2])
    return circ

def tomo_circ(get_circ):
    service = QiskitRuntimeService()
    backend = service.least_busy(simulator=False, operational=True)
    exp = StateTomography(get_circ())
    data = exp.run(backend=backend, seed_simulation=10000).block_for_results()
    df = data.analysis_results(dataframe=True)
    p_xy = df['value'].values[0].data
    return qci.tr_z(p_xy, 2, 2, 2)

dx = 2
dy = 2
dz = 2

esti_state = tomo_circ(get_ghz_3)

print(esti_state)

problem = qci.QProblem(esti_state, dx, dy, dz)

penalties = np.arange(0.0, 1.0, 0.01)
tolerance = 0.05
entrop_thresh = 1
extern_thresh = None
dep_gate = 0.2
smoothing = 0.01
damping = 0.6962200606112787
log_reg = 0.007043188253465324
n = 100

null_fam = []
sig_lvl = 0.2

result = qci.QInferGraph(problem, penalties, tolerance, entrop_thresh, extern_thresh, dep_gate, 
                         smoothing, damping, log_reg, n, null_fam, sig_lvl)

print(result.result_message)
