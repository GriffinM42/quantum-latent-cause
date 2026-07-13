import sys
sys.path.insert(1, 'C:\\Users\\gevmo\\OneDrive\\Music\\Documents\\2026 - Summer\\Causal Inference\\quantum-latent-cause')

import lib.quantum_causal_inference as qci
import lib.inference_hyperparams as ih
import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import QuantumRegister, ClassicalRegister
from qiskit_experiments.library import StateTomography

from qiskit_aer import AerSimulator

ideal_state = np.array([[0.25, 0.25, 0.25, 0.25], 
                       [0.25, 0.25, 0.25, 0.25], 
                       [0.25, 0.25, 0.25, 0.25], 
                       [0.25, 0.25, 0.25, 0.25]])

def get_product_state():
    q = QuantumRegister(2)
    b = ClassicalRegister(2)
    circ = QuantumCircuit(q, b)
    circ.h(q[0])
    circ.h(q[1])
    return circ

def tomo_circ(get_circ):
    aer = AerSimulator()
    exp = StateTomography(get_circ())
    data = exp.run(backend=aer).block_for_results()
    df = data.analysis_results(dataframe=True)
    print(df)
    p_xy = df['value'].values[0].data
    print(df['value'].values[1])
    return p_xy

def D(a, b):
    c = a - b
    c_t = np.conjugate(np.transpose(c))
    return 0.5 * np.trace(qci.sqrt(np.matmul(c_t, c)))

dx = 2
dy = 2
dz = 2

esti_state = tomo_circ(get_product_state)

print(esti_state)
print(D(esti_state, ideal_state))
print(qci.trace_dist(esti_state, ideal_state))
print(qci.trace_dist(ideal_state, esti_state))
