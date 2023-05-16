import numpy as np
from matplotlib import pyplot as plt

# consts
x_n = 300
y_n = 200

I = 1.0
Dh = 10
b = 0.05
r = 200


def new_probab(H, wet_spots, Dh=Dh):
    transpons = [[-1, 0], [-1, -1], [0, -1], [1, -1], [0, 1]]
    beta = 0.05

    erode_cords = []

    for i, spot in enumerate(wet_spots):
        probs = np.zeros(len(transpons))
        tmp_list = []
        if spot[1] > 0:
            for j, changes in enumerate(transpons):
                if (
                    H[spot[0], spot[1]] - H[spot[0] + changes[0], spot[1] + changes[1]]
                    > 0
                ):
                    probs[j] = np.exp(
                        beta
                        * (
                            H[spot[0], spot[1]]
                            - H[spot[0] + changes[0], spot[1] + changes[1]]
                        )
                    )
                # tmp_list.append([wet_spots[i][0],wet_spots[i][1]])
            print(np.sum(probs))
            probs = probs / np.sum(probs)

            if (wet_spots[i][0] > 0 and wet_spots[i][0] < 300) and (
                wet_spots[i][1] > 0 and wet_spots[i][1] < 200
            ):
                wet_spots[i][0] += transpons[np.argmax(probs)][0]
                wet_spots[i][1] += transpons[np.argmax(probs)][1]
        else:
            pass

        # erode_cords.append(tmp_list)

    return wet_spots  # ,erode_cords


def erode(H, wet_spots, Dh=Dh):
    delta_H = Dh

    for i, spot in enumerate(wet_spots):
        if cond_arr[spot[0], spot[1]] == 0:
            H[spot[0], spot[1]] -= 10
            cond_arr[spot[0], spot[1]] = 1
        else:
            pass

    return H


# initial hight map
H = np.ones((x_n, y_n)) * 100


for i in range(y_n):
    H[:, i] = I * i + np.random.rand(x_n)

H2 = np.copy(H)
wet_spots = []

for i in range(100):
    wet_spots.append([np.random.randint(15, x_n - 4), np.random.randint(40, y_n - 40)])


cond_arr = np.zeros((H.shape))

# random starting point - it's working
for i in range(100):
    # print(wet_spots[0][0], wet_spots[0][1])
    plt.imshow(H2.T, origin="lower")
    #for spot in wet_spots:
    #    if spot[1] != 0:
    #        plt.plot(spot[0], spot[1], "r.")
    plt.savefig(f"{i}.png")
    plt.close()
    wet_spots = new_probab(H, wet_spots)
    H2 = erode(H2, wet_spots, cond_arr)
