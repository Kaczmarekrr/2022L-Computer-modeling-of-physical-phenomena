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
raindrops = 10000


def new_probab(H, wet_spots,cond_arr, Dh=10, x_n=300):
    transpons = [[-1, 0], [0, -1], [0, 1], [1, 0],[-1,-1],[1,-1]]
    beta = 0.05

    erode_cords = []
    #plt.imshow(H_spot)
    #plt.show()
    for i, spot in enumerate(wet_spots):
        probs = np.zeros(len(transpons))
        tmp_list = []
        
        if spot[1] > 0:
            for j, changes in enumerate(transpons):
                new_x, new_y = spot[0] + changes[0], spot[1] + changes[1]

                if new_x >= x_n:
                    new_x = 0

                if H[spot[0], spot[1]]  - (H[new_x, new_y]) > 0:
                    probs[j] = np.exp(
                        beta * (H[spot[0], spot[1]] - (H[new_x, new_y]))
                    )
                # tmp_list.append([wet_spots[i][0],wet_spots[i][1]])
            # print(np.sum(probs))
            if np.sum(probs) > 0:
                probs = probs / np.sum(probs)
            else: pass

            if (wet_spots[i][0] > 0 and wet_spots[i][0] < 300) and (
                wet_spots[i][1] > 0 and wet_spots[i][1] < 200
            ):
                wet_spots[i][0] += transpons[np.argmax(probs)][0]
                wet_spots[i][1] += transpons[np.argmax(probs)][1]

                if wet_spots[i][0] >= 300:
                    wet_spots[i][0] = 0
        else:
            wet_spots[i][1] = 0
        # erode_cords.append(tmp_list)

    return wet_spots  # ,erode_cords


def new_probab_1(H, wet_spot,cond_arr, Dh=10, x_n=300):
    transpons = [[-1, 0], [0, -1], [0, 1], [1, 0],[-1,-1],[1,-1]]
    beta = 0.05

    erode_cords = []
    #plt.imshow(H_spot)
    #plt.show()
    
    probs = np.zeros(len(transpons))
    tmp_list = []
    return_wet_spot = [99,99]
    if wet_spot[1] > 0:
        for j, changes in enumerate(transpons):
            new_x, new_y = wet_spot[0] + changes[0], wet_spot[1] + changes[1]

            if new_x >= x_n:
                new_x = 0

            if H[wet_spot[0], wet_spot[1]]  - (H[new_x, new_y]) > 0:
                probs[j] = np.exp(
                    beta * (H[wet_spot[0], wet_spot[1]] - (H[new_x, new_y]))
                )
            # tmp_list.append([wet_spots[i][0],wet_spots[i][1]])
        # print(np.sum(probs))
        if np.sum(probs) > 0:
            probs = probs / np.sum(probs)
        else: pass

        if (wet_spot[0] > 0 and wet_spot[0] < 300) and (
            wet_spot[1] > 0 and wet_spot[1] < 200
        ):
            wet_spot[0] += transpons[np.argmax(probs)][0]
            wet_spot[1] += transpons[np.argmax(probs)][1]
            if wet_spot[0] >= 300:
                wet_spot[0] = 0
    else:
        wet_spot[1] = 0
    # erode_cords.append(tmp_list)
    return_wet_spot = wet_spot.copy()
    return return_wet_spot  # ,erode_cords


def erode(H, H2,wet_spots, delta_H=10):
    for i, spot in enumerate(wet_spots):
        if cond_arr[i, spot[0], spot[1]] == 0:
            cond_arr[i, spot[0], spot[1]] = 1
        # except IndexError:
        # print(spot[0], spot[1])
        # else:
        # pass

        if np.any(cond_arr[i, :, 0]==1) and np.all(cond_arr[i, :, 0]!=2) :
            H2[cond_arr[i,:,:]>0] -= 10
            cond_arr[i, :, 0] = 2

    return H,H2, cond_arr


def avalange(H2,r):
    tmp_H2 = np.copy(H2)
    roll_direction = np.array([[-1, 0], [1, 0], [-1, 1], [1, 1]])  # move, axis
    check_cond = False
    for i in range(4):
        while np.any(H2 - np.roll(tmp_H2, roll_direction[i, 0], axis=roll_direction[i, 1]) > r):
        
            H2 = np.where(
                H2 - np.roll(tmp_H2, roll_direction[i, 0], axis=roll_direction[i, 1]) > r,
                H2
                - 0.25 * (H2
                - np.roll(tmp_H2, roll_direction[i, 0], axis=roll_direction[i, 1])),
                H2,
            )
    
        
    H2[:,-1] = tmp_H2[:,-1] #to no to go from down to up

    return H2


# initial hight map
H = np.ones((x_n, y_n))


for i in range(y_n):
    H[:, i] = I * i + np.random.rand(x_n)

#H[:,0] = -5000
H2 = H.copy()
wet_spots = []

for i in range(raindrops):
    wet_spots.append([np.random.randint(15, x_n - 4), np.random.randint(1, y_n - 1)])

cond_arr = np.zeros((raindrops, H.shape[0], H.shape[1]))


H , H2, cond_arr = erode(H,H2, wet_spots, cond_arr)
# random starting point - it's working
for i in tqdm(range(len(wet_spots))):
    points_to_erode = []
    points_to_erode.append(wet_spots[i])
    breakaer = 0
    wet_spot = wet_spots[i]
    while wet_spot[1] > 0 and breakaer <200:
        wet_spot = new_probab_1(H, wet_spot,cond_arr, Dh=10, x_n=300)
        points_to_erode.append(wet_spot)
        breakaer+=1
    

    for x,y in points_to_erode:
        H[x,y] -= Dh

    H = avalange(H,r)
    #plt.imshow(np.sum(cond_arr[:,1:],axis=0).T, origin="lower",vmax=5)
    #plt.imshow(cond_arr[1].T, origin="lower")

    if i % 100==0:
        plt.imshow(H.T, origin="lower")
        #for spot in wet_spots:
        #   if spot[1] != 0:
        #       plt.plot(spot[0], spot[1], "r.")
        plt.colorbar()
        plt.savefig(f"results/{i}.png")

        plt.close()
    #wet_spots = new_probab(H2[:, :], wet_spots,cond_arr)
    #H, H2, cond_arr = erode(H,H2, wet_spots, cond_arr)
    #H2 = avalange(H2,r)


    # erode is working!!!!
