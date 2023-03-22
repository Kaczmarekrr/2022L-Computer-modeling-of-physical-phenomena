import numpy as np
from matplotlib import pyplot as plt
import imageio
import os
from tqdm import tqdm
from scipy.signal import argrelextrema

# constants


N = 100
##########################
def step(u, v):
    D_u = 2 * 10 ** (-5)
    D_v = 1 * 10 ** (-5)
    F = 0.037
    k = 0.06
    dt = 1
    dx = 0.02

    u_2nd = (np.roll(u, -1) + np.roll(u, +1) - 2 * u) / dx**2
    v_2nd = (np.roll(v, -1) + np.roll(v, +1) - 2 * v) / dx**2

    u_derivative = D_u * u_2nd - u * v**2 + F * (1 - u)
    v_derivative = D_v * v_2nd + u * v**2 - (F + k) * v

    u = u + u_derivative * dt
    v = v + v_derivative * dt
    return u, v


def main():
    u = np.ones(N)
    v = np.zeros(N)
    xs = np.arange(0, 2, 0.02)
    for i in range(int(N / 4), int(3 * N / 4)):
        u[i] = np.random.random() * 0.2 + 0.4
        v[i] = np.random.random() * 0.2 + 0.2

    maxims_v = np.zeros((5000, N))
    for time_step in tqdm(range(10000)):
        where_max = argrelextrema(v, np.greater)
        maxims_v[time_step,where_max] = 1
        u, v = step(u, v)
        if time_step % 1000 == 0:

            plt.figure()
            plt.title(str(time_step))
            plt.plot(xs, u)
            plt.plot(xs, v)

            if time_step < 10:
                title = "000" + str(time_step)

            elif time_step < 100:
                title = "00" + str(time_step)
            elif time_step < 1000:
                title = "0" + str(time_step)
            else:
                title = str(time_step)
            plt.savefig(f"results/{title}.png")
            plt.close()

    print(where_max)
    plt.figure()
    plt.imshow(maxims_v, aspect=0.008, vmax=1, vmin=0,origin="lower")
    plt.savefig("maxims_v.png")


def make_gif(path):
    filenames = os.listdir(path)
    images = []
    for filename in tqdm(filenames):
        images.append(imageio.v2.imread(path + filename))

    kargs = {"duration": 0.2}
    imageio.mimsave("movie.gif", images, **kargs)


if __name__ == "__main__":
    main()
    make_gif("results/")
