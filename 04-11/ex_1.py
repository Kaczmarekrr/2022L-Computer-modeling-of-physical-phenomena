import numpy as np
from tqdm import tqdm

zero = np.matrix([
0, 1, 1, 1, 0,
1, 0, 0, 0, 1,
1, 0, 0, 0, 1,
1, 0, 0, 0, 1,
1, 0, 0, 0, 1,
0, 1, 1, 1, 0
])

one = np.matrix([
0, 1, 1, 0, 0,
0, 0, 1, 0, 0,
0, 0, 1, 0, 0,
0, 0, 1, 0, 0,
0, 0, 1, 0, 0,
0, 0, 1, 0, 0
])


two = np.matrix([
1, 1, 1, 0, 0,
0, 0, 0, 1, 0,
0, 0, 0, 1, 0,
0, 1, 1, 0, 0,
1, 0, 0, 0, 0,
1, 1, 1, 1, 1,
])

noisy0 =np.matrix([
0, 1, 1, 1, 0,
1, 0, 0, 0, 0,
1, 0, 0, 0, 1,
0, 0, 0, 0, 0,
0, 0, 0, 0, 0,
0, 0, 1, 0, 0,
])

noisy2 = np.matrix([
1, 1, 1, 0, 0,
0, 0, 0, 0, 0,
0, 0, 0, 0, 0,
0, 0, 1, 0, 0,
1, 0, 0, 0, 0,
1, 1, 0, 0, 1,
])

noisy2b = np.matrix([
1, 1, 1, 0, 0,
0, 0, 0, 1, 0,
0, 0, 0, 1, 0,
0, 0, 0, 0, 0,
0, 0, 0, 0, 0,
0, 0, 0, 0, 0,
])
data = np.array([zero,one,two])
data2 = np.array([noisy0,noisy2,noisy2b])



def new_weights(data):
    w = np.zeros(data[0].shape)
    for example in range(len(data)):
        #for i in range(example.shape[0]):
            #for j in range(example.shape[1]):
                #w[i,j] += example[i] * example[j] 
        w += example @ example -1 

    
    return w/len(data)


def update(energy, data, weights):
    energy = np.zeros(len(data))

    for i in range(len(energy)):
        energy[i] = -(1/2)*data[i]  * np.sum(weights[i]*data)


for step in range(1000):
    new_weights