from qiskit import QuantumCircuit
from qiskit.circuit import QuantumRegister, ClassicalRegister
from qiskit_experiments.library import StateTomography

from qiskit_aer import AerSimulator
from qiskit.primitives import BackendSamplerV2

import numpy as np
import numpy.linalg as lin

aer = AerSimulator()

def get_bell_00():
    q = QuantumRegister(2)
    b = ClassicalRegister(2)
    circ = QuantumCircuit(q, b)
    circ.h(q[0])
    circ.cx(q[0], q[1])
    return circ

def tomo_circ(get_circ):
    exp = StateTomography(get_circ())
    data = exp.run(backend=aer, seed_simulation=10000).block_for_results()
    df = data.analysis_results(dataframe=True)
    # x = df[df['name'] == 'state']['value']
    p_xy = df['value'].values[0]
    return p_xy

p_xy = tomo_circ(get_bell_00)


print(p_xy)
