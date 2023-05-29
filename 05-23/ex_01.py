import numpy as np
from matplotlib import pyplot as plt
import scipy
from matplotlib import pyplot as plt

# consts
J = [-1.1, -2.1, -3.8]  # 12, 13, 23
h = np.array([0.6, 0, 0])
# 01 02 12
# 10 12


sz = np.array([[1, 0.0], [0.0, -1]])
one = np.eye(2)
sx = np.eye(2)[::-1]

sz_1 = np.kron(sz, np.kron(one, one))
sz_2 = np.kron(one, np.kron(sz, one))
sz_3 = np.kron(one, np.kron(one, sz))
sz_i = [sz_1, sz_2, sz_3]


sx_1 = np.kron(sx, np.kron(one, one))
sx_2 = np.kron(one, np.kron(sx, one))
sx_3 = np.kron(one, np.kron(one, sx))
sx_i = [sx_1, sx_2, sx_3]


# H0 = - sum_i(Sx_i)
H0 = -1 * (sx_1 + sx_2 + sx_3)

print(H0,"\n","\n")
#J = [1, 2, 2]  # 12, 13, 23
#h = np.array([1, 0, 0])
#to be changed

H1_2 = np.zeros((8, 8))
H1_1 = np.zeros((8, 8))
H1_2 -= h[0] * sz_1 + h[1] * sz_2 + h[2] * sz_3


H1_1 += np.dot(sz_1, sz_2) * J[0]
H1_1 += np.dot(sz_1, sz_3) * J[1]
H1_1 += np.dot(sz_2, sz_3) * J[2]


H1 = H1_1 + H1_2



dl = 0.1
lamb = np.arange(0,1,dl)
#lamb = 0.1

results = []
results2 = []
tmp = 0
for l in lamb:
    H_l = l*H1 - ((1-l)*H0)
    H_l = (H_l + (np.conjugate(H_l)).T) / 2

    eigen_values, eigen_vectors = np.linalg.eigh(H_l)

    delta_E = eigen_values[1]-eigen_values[0]

    tmp += dl*delta_E**(-2)
    results.append(delta_E)
    print(eigen_values)
plt.plot(lamb,results)
plt.xlabel("adiabatic parameter")
plt.ylabel("Energy gap")
plt.title("Energy gap along AQC")
plt.ylim(0,2)
plt.show()

print("time",tmp)

