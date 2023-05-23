import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

# consts
x_n = 300
y_n = 200

I = 1
Dh = 10
b = 0.05
r = 200
raindrops = 5000

H = np.ones((x_n, y_n))

river = np.ones((H.shape))


for i in range(y_n):
    H[:, i] = I * i + 10e-6 * np.random.rand(x_n)

print(H[10:40, 50])
print(H[10:40, 51])
H2 = H.copy()
wet_spots = np.array(
    [
        np.random.randint(0, x_n, size=raindrops),
        np.random.randint(1, y_n - 1, size=raindrops),
    ],
    dtype=np.int32,
)
wet_spots = wet_spots.T
print(wet_spots.shape)
print(wet_spots[0])


# wetspot are goood
def get_cord_BD(x, y, transpons, x_n=300):
    new_x, new_y = x + transpons[0], y + transpons[1]
    if new_x >= x_n:  # new_x == 300 -> 0, with new_x = -1 there is no problem
        new_x = 0
    if new_y >= x_n:
        new_y = 0
    return new_x, new_y


def new_probab_1(H, wet_spot, x_n=300):
    transpons = [[1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]]
    beta = 0.05

    probs = np.zeros(len(transpons))
    x, y = wet_spot[0], wet_spot[1]
    for j, changes in enumerate(transpons):
        new_x, new_y = get_cord_BD(x, y, changes)
        if new_x > x_n - 1:  # new_x == 300 -> 0, with new_x = -1 there is no problem
            new_x = 0
        try:
            if H[x, y] - (H[new_x, new_y]) > 0:
                probs[j] = np.exp(beta * (H[x, y] - (H[new_x, new_y])))
        except IndexError:
            print(x, y, new_x, new_y)
    if np.sum(probs) > 0:
        probs = probs / np.sum(probs)
        wet_spot[0] += transpons[np.argmax(probs)][0]
        wet_spot[1] += transpons[np.argmax(probs)][1]
        if wet_spot[0] == 300:
            wet_spot[0] = 0
    elif wet_spot[1] > 0:
        wet_spot[1] += -1

    return wet_spot  # ,erode_cords

def find_river(river):
    mask = np.zeros((river.shape),dtype=np.int32)
    transpons = [[1, 0], [0, 1], [-1, 0], [1, 1], [-1, 1]]

    start_x, start_y = np.where(river[:,0]>100)[0], 0
    for x in start_x:
        y = start_y
        mask[x,start_y] = 1
        cond = True
        breaker = 500
        while cond is True and breaker> 0:
            for j, changes in enumerate(transpons):
                new_x, new_y = get_cord_BD(x, y, changes)
                if new_x > x_n - 1:  # new_x == 300 -> 0, with new_x = -1 there is no problem
                    new_x = 0
                if river[new_x,new_y] > 100:
                    mask[new_x,new_y] = 1
                    x,y = new_x,new_y
                    cond = True
                    break
                else: cond = False
            breaker-=1

    return mask

def avalange(H2, r):
    tmp_H2 = np.copy(H2)
    e = np.array(
        [[1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]]
    )
    e = np.array(
        [[1, 0], [0, 1], [-1, 0], [0, -1]]
    )
    for i in range(len(e)):
        diff = np.roll(np.roll(tmp_H2[:,:],e[i][0],axis=0),-e[i,1],axis=1)
        while np.any(H2 - diff > r):
            H2 = np.where(H2 - diff > r, H2 - 0.25 * (H2 - diff), H2)

    H2[:, -1] = tmp_H2[:, -1]  # to no to go from down to up

    return H2



for i in tqdm(range(raindrops)):
    points_to_erode = []
    points_to_erode.append([int(wet_spots[i, 0]), int(wet_spots[i, 1])])
    breaker = 1000
    while wet_spots[i][1] > 0 and breaker > 0:
        wet_spots[i] = new_probab_1(H, wet_spots[i])
        list_arr = [int(wet_spots[i, 0]), int(wet_spots[i, 1])]
        points_to_erode.append(list_arr)
        breaker -=1
        
        #print(wet_spots[i])
    unique_points = list({tuple(x) for x in points_to_erode})
    for x, y in unique_points:
        H[x, y] -= Dh
        river[x,y]+=1

    

    H = avalange(H, r)

    if i % 100 == 0:
        plt.imshow(H.T, origin="lower")
        plt.colorbar()
        plt.savefig(f"results/{i}.png")

        plt.close()

        
        
        
        plt.imshow(river.T, origin="lower",vmin=100,vmax=101.1)
        plt.colorbar()
        plt.savefig(f"results/river_{i}.png")
        plt.close()

from mpl_toolkits.mplot3d import Axes3D

# Set up grid and test data

x = range(x_n)
y = range(y_n)



hf = plt.figure()
ha = hf.add_subplot(111, projection='3d')

X, Y = np.meshgrid(x, y)  # `plot_surface` expects `x` and `y` data to be 2D


ha.plot_surface(X, Y, H.T)

plt.show()