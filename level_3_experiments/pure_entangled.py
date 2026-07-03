import sys
sys.path.insert(1, 'C:\\Users\\gevmo\\OneDrive\\Music\\Documents\\2026 - Summer\\Causal Inference\\quantum-latent-cause')

import quantum_causal_inference as qci

from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.transpiler import generate_preset_pass_manager

from qiskit import QuantumCircuit
from qiskit.circuit import QuantumRegister, ClassicalRegister
from qiskit_experiments.library import StateTomography

import numpy as np
import numpy.linalg as lin

service = QiskitRuntimeService()

def get_bell_00():
    q = QuantumRegister(2)
    b = ClassicalRegister(2)
    circ = QuantumCircuit(q, b)
    circ.h(q[0])
    circ.cx(q[0], q[1])
    return circ

def tomo_circ(get_circ):
    backend = service.least_busy(simulator=False, operational=True)
    exp = StateTomography(get_circ())
    data = exp.run(backend=backend, seed_simulation=1000).block_for_results()
    df = data.analysis_results(dataframe=True)
    p_xy = df['value'].values[0].data
    return p_xy

dx = 2
dy = 2
dz = 2

esti_state = tomo_circ(get_bell_00)

print(esti_state)

problem = qci.QProblem(esti_state, dx, dy, dz)

penalties = np.arange(0.0, 1.0, 0.01)
tolerance = 0.2
entrop_thresh = 0.8
extern_thresh = None
dep_gate = 0.2
smoothing = 0.01
damping = 0.2
log_reg = 0.2
n = 100

null_fam = []
sig_lvl = 0.2

result = qci.QInferGraph(problem, penalties, tolerance, entrop_thresh, extern_thresh, dep_gate, 
                         smoothing, damping, log_reg, n, null_fam, sig_lvl)

print(result.result_message)