import numpy as np
from qiskit import QuantumRegister, ClassicalRegister


q = QuantumRegister(2,'qubit')


c = ClassicalRegister(2, 'bi')

print(q)
print(q[0],q)