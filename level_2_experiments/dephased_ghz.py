import sys
sys.path.insert(1, 'C:\\Users\\gevmo\\OneDrive\\Music\\Documents\\2026 - Summer\\Causal Inference\\quantum-latent-cause')

import lib.quantum_causal_inference as qci
import lib.inference_hyperparams as ih

from qiskit import QuantumCircuit
from qiskit.circuit import QuantumRegister, ClassicalRegister
from qiskit_experiments.library import StateTomography

from qiskit_aer import AerSimulator

def get_ghz_3():
    q = QuantumRegister(4)
    b = ClassicalRegister(4)
    circ = QuantumCircuit(q, b)
    circ.h(q[0])
    circ.cx(q[0], q[1])
    circ.cx(q[0], q[2])
    circ.cx(q[0], q[3])
    return circ

def tomo_circ(get_circ):
    aer = AerSimulator()
    exp = StateTomography(get_circ(), measurement_indices=[0, 1])
    data = exp.run(backend=aer).block_for_results()
    df = data.analysis_results(dataframe=True)
    p_xy = df['value'].values[0].data
    return p_xy

dx = 2
dy = 2
dz = 2

esti_state = tomo_circ(get_ghz_3)

print(esti_state)

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
