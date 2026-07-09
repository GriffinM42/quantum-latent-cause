import math
import numpy as np
import lib.quantum_causal_inference as qci

A = np.zeros((64, 64))

q = 0.4
p1 = 0.01
p2 = 0.5

for x in range(0, 4):
    for y in range(0, 4):
        for z in range(0, 4):
            prod = 1
            prod *= q if (z & 2) == 2 else (1-q)
            prod *= q if (z & 1) == 1 else (1-q)
            prod *= p1 if ((z&2) != (x&2)) else (1-p1)
            prod *= p1 if ((z&1) != (x&1)) else (1-p1)
            prod *= p2 if ((z&2) != (y&2)) else (1-p2)
            prod *= p2 if ((z&1) != (y&1)) else (1-p2)
            A[x*16 + y*4 + z][x*16 + y*4 + z] = prod

print(A)

dx = 4
dy = 4
dz = 4

prob = qci.tr_z(A, 4, 4, 4)
print(prob)

p_x = np.trace(prob.reshape(dx, dy, dx, dy), axis1=1, axis2=3)
p_y = np.trace(prob.reshape(dx, dy, dx, dy), axis1=0, axis2=2)



print(qci.vn_entropy(p_x))
print(qci.vn_entropy(p_y))
print(qci.mi_xy(prob, dx, dy))