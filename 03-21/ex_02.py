import numpy as np
from matplotlib import pyplot as plt
import imageio
import os
from tqdm import tqdm
from scipy.signal import argrelextrema


import numpy as np
from matplotlib import pyplot as plt
import imageio
import os
from tqdm import tqdm
from scipy.signal import argrelextrema



##########################


def make_gif(path,name):
    filenames = os.listdir(path)
    images = []
    for filename in tqdm(filenames):
        images.append(imageio.v2.imread(path + filename))

    kargs = {"duration": 0.2}
    imageio.mimsave(f"{name}.gif", images, **kargs)


def step(u, v,F,k):
    D_u = 2 * 10 ** (-5)
    D_v = 1 * 10 ** (-5)
    dt = 1
    dx = 0.02

    u_2nd = (np.roll(u, -1,axis=0) + np.roll(u, +1,axis=0) + np.roll(u, -1,axis=1) + np.roll(u, +1,axis=1) - 4 * u) / dx**2
    v_2nd = (np.roll(v, -1,axis=0) + np.roll(v, +1,axis=0) + np.roll(v, -1,axis=1) + np.roll(v, +1,axis=1) - 4 * v) / dx**2


    u_derivative = D_u * u_2nd - u * v**2 + F * (1 - u)
    v_derivative = D_v * v_2nd + u * v**2 - (F + k) * v

    u = u + u_derivative * dt
    v = v + v_derivative * dt
    return u, v


def main():
    # constants
    F = [0.025,0.03,0.01,0.04,0.06,0.037]
    k = [0.055,0.062,0.047,0.07,0.0615,0.06]
    N = 100 

    for i, f_i in enumerate(F):
        k_i = k[i]
        U = np.ones((N,N))
        V = np.zeros((N,N))
        for i in range(int(N / 4), int(3 * N / 4)):
            for j in range(int(N / 4), int(3 * N / 4)):
                U[i,j] = np.random.random() * 0.2 + 0.4
                V[i,j] = np.random.random() * 0.2 + 0.2

        for time_step in tqdm(range(10_000)):
            #where_max = argrelextrema(v, np.greater)
            #maxims_v[time_step,where_max] = 1
            U, V = step(U, V,f_i,k_i)
            if time_step % 200 == 0:

                plt.figure()
                plt.title(time_step)
                plt.contourf(np.arange(100),np.arange(100),V)

                if time_step < 10:
                    title = "0000" + str(time_step)

                elif time_step < 100:
                    title = "000" + str(time_step)
                elif time_step < 1000:
                    title = "00" + str(time_step)
                elif time_step < 10000:
                    title = "0" + str(time_step)
                else:
                    title = str(time_step)
                plt.savefig(f"results_2/{title}.png")
                plt.close()
        make_gif("results_2/",f"movie_{f_i}-{k_i}")
    #print(where_max)
    #plt.figure()
    #plt.imshow(maxims_v, aspect=0.008, vmax=1, vmin=0,origin="lower")
    #plt.savefig("maxims_v_2.png")





if __name__ == "__main__":

    
    main()
    