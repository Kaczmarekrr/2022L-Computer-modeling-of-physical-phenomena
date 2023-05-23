import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

# consts
x_n = 300
y_n = 200

I = 1.0
Dh = 10
b = 0.05
r = 200
raindrops = 1000

def new_probab(H, wet_spots, Dh=Dh,x_n=300):
    transpons = [[-1, 0], [-1, -1], [0, -1], [1, -1], [1,0]]
    beta = 0.05

    erode_cords = []

    for i, spot in enumerate(wet_spots):
        probs = np.zeros(len(transpons))
        tmp_list = []
        if spot[1] > 0:
            for j, changes in enumerate(transpons):
                new_x, new_y = spot[0] + changes[0], spot[1] + changes[1]

                if new_x >= x_n:
                    new_x = 0

                if (
                    H[spot[0], spot[1]] - (H[new_x, new_y] - Dh)
                    > 0
                ):
                    probs[j] = np.exp(
                        beta
                        * (
                            H[spot[0], spot[1]]
                            - (H[new_x, new_y] - Dh)
                        )
                    )
                # tmp_list.append([wet_spots[i][0],wet_spots[i][1]])
            #print(np.sum(probs))
            probs = probs / np.sum(probs)

            if (wet_spots[i][0] > 0 and wet_spots[i][0] < 300) and (
                wet_spots[i][1] > 0 and wet_spots[i][1] < 200
            ):
                wet_spots[i][0] += transpons[np.argmax(probs)][0]
                wet_spots[i][1] += transpons[np.argmax(probs)][1]

                if wet_spots[i][0] >= 300:
                    wet_spots[i][0] = 0
        else:
            pass
        # erode_cords.append(tmp_list)

    return wet_spots  # ,erode_cords


def erode(H, wet_spots, Dh=Dh):
    delta_H = Dh

    for i, spot in enumerate(wet_spots):
        if cond_arr[i,spot[0], spot[1]] == 0:
            H[spot[0], spot[1]] -= 10
            cond_arr[i,spot[0], spot[1]] = 1
        #except IndexError: 
        #print(spot[0], spot[1])
        #else:
           # pass

    return H,cond_arr


# initial hight map
H = np.ones((x_n, y_n))


for i in range(y_n):
    H[:, i] = I * i + np.random.rand(x_n)

H2 = H.copy()
wet_spots = []

for i in range(raindrops):
    wet_spots.append([np.random.randint(15, x_n - 4), np.random.randint(1, y_n - 1)])



cond_arr = np.zeros((raindrops,H.shape[0],H.shape[1]))


H2,cond_arr = erode(H2, wet_spots, cond_arr)
# random starting point - it's working



for i in tqdm(range(50)):
    # print(wet_spots[0][0], wet_spots[0][1])
    if i % 5 == 0:
        plt.imshow(H2[:,:].T, origin="lower")
        #for spot in wet_spots:
        #    if spot[1] != 0:
        #        plt.plot(spot[0], spot[1], "r.")
        #        plt.plot(spot[0], spot[1]-1, "g.")
        plt.colorbar()
        plt.savefig(f"results/{i}.png")
        
        plt.close()
    wet_spots = new_probab(H2[:,:], wet_spots)
    H2,cond_arr = erode(H2, wet_spots, cond_arr)

    #erode is working!!!!
    