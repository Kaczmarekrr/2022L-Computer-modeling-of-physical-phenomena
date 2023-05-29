import numpy as np
from matplotlib import pyplot as plt



# consts
J_list = [-1.1, -2.1, -3.8]  # 12, 13, 23
h_list = np.array([0.6, 0, 0])
# 01 02 12
# 10 12 




sz = np.array([[1, 0.0], [0.0, -1]])
one = np.eye(2)
sx = one = np.eye(2)[::-1]

sz_1 = np.kron(sz, np.kron(one, one))
sz_2 = np.kron(one, np.kron(sz, one))
sz_3 = np.kron(one, np.kron(one, sz))
sz_i = [sz_1,sz_2,sz_3]


sx_1 = np.kron(sx, np.kron(one, one))
sx_2 = np.kron(one, np.kron(sx, one))
sx_3 = np.kron(one, np.kron(one, sx))
sx_i = [sx_1,sx_2,sx_3]


n = 0
Node = {
    "Q1": [sz_1, one, h_list[0]],
    "Q2": [sz_2, one, h_list[1]],
    "Q3": [sz_3, one, h_list[2]],
}


 # 12, 13, 23

mmm = 20
lam = np.linspace(0,1,mmm)
print(lam)
H0 = np.zeros((mmm,8,8))
H1 = np.zeros((mmm,8,8))    
H = np.zeros((mmm,8,8)) 
A = np.zeros((mmm,1))
ground = np.zeros((mmm,1))
for k,l in enumerate(lam):
    for i in range(3):
        j= i+1
        if j>2:
            j = 0
        H1[k] = H1[k] - (J_list[i] * np.matmul(sz_i[i],sz_i[j])) - (sz_i[i]* h_list[i])
        H0[k] -= sx_i[i]



    H[k] = (1 - l) * H0[k] + l*H1[k]
    H[k] = (H[k] + H[k].T) / 2

    eigen_values, _ = np.linalg.eigh(H[k])
    A[k] = eigen_values[0]
    eigen_values, _ = np.linalg.eigh(H0[k])
    ground[k] = eigen_values[0]
    

# TODO: this
#T_ac = np.sum((1/A**2)*(1/mmm))

print(A)
plt.plot(lam,A)
plt.plot(lam,ground)
plt.show()

