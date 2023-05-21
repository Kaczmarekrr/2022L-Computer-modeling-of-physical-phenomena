import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import imageio
import os
def get_payoff(CC, DD, DC, CD, b):
    cc_pay = 1
    dd_pay = 0
    dc_pay = b
    cd_pay = 0

    payoff_matrix = np.zeros((201, 201))
    payoff_matrix = CC * cc_pay + DD * dd_pay + DC * dc_pay + CD * cd_pay

    return payoff_matrix


def payoff_matrix(strat_matrix, b):
    """get payoff matrix for chosen points
    This function get the idxs of points in one of 4 class of payoff

    Args:
        strat_matrix (np.ndarray): stategy matrix
        b (int): weight for get_payoff
    Returns:
        _type_: iter over all nearbours
    """

    CC = np.zeros((201, 201))
    DC = np.zeros((201, 201))
    CD = np.zeros((201, 201))
    DD = np.zeros((201, 201))

    for i in range(-1, 2):
        for j in range(-1, 2):
            CC[
                np.where(
                    (strat_matrix == np.roll(np.roll(strat_matrix, -j, 1), -i, 0)) & (strat_matrix == 0)
                )
            ] += 1

            DD[
                np.where(
                    (strat_matrix == np.roll(np.roll(strat_matrix, -j, 1), -i, 0)) & (strat_matrix == 1)
                )
            ] += 1

            DC[
                np.where(
                    (strat_matrix != np.roll(np.roll(strat_matrix, -j, 1), -i, 0)) & (strat_matrix == 1)
                )
            ] += 1

            CD[
                np.where(
                    (strat_matrix != np.roll(np.roll(strat_matrix, -j, 1), -i, 0)) & (strat_matrix == 0)
                )
            ] += 1
    payoff = get_payoff(CC, DD, DC, CD, b)
    return payoff

def change_choices(strat_matrix,payoff_matrix):
    new_strat = np.zeros((201,201))
    tmp_pay = np.zeros((201,201))
    for i in range(-1, 2):
        for j in range(-1, 2):
            tmp_pay = np.where(np.roll(np.roll(payoff_matrix, -j, 1), -i, 0)>tmp_pay,np.roll(np.roll(payoff_matrix, -j, 1), -i, 0),tmp_pay)
        
    for i in range(-1, 2):
        for j in range(-1, 2):
            new_strat = np.where(np.roll(np.roll(payoff_matrix, -j, 1), -i, 0)==tmp_pay,np.roll(np.roll(strat_matrix, -j, 1), -i, 0),new_strat)

    return new_strat


def plot_changes(last_strat,choices_matrix,i,b):
    plot_matrix = np.zeros((201,201,3),dtype=np.int32)
    plot_matrix[np.where((last_strat==0)&(choices_matrix==1))] = (0,0,0) # C to D -> black
    plot_matrix[np.where((last_strat==1)&(choices_matrix==0))] = (255,255,255) # D to C -> white
    plot_matrix[np.where((last_strat==0)&(choices_matrix==0))] = (0,255,0) # C to C -> green
    plot_matrix[np.where((last_strat==1)&(choices_matrix==1))] = (255,0,0) # D to D -> red

    plt.figure()
    plt.title(f"i = {i}, b = {b}")
    plt.imshow(plot_matrix)
    if i < 10:
        name = f"results_2/{b:.3f}_img_00{i}.png"
    elif i < 100:
        name = f"results_2/{b:.3f}_img_0{i}.png"
    elif i < 1000:
        name = f"results_2/{b:.3f}_img_{i}.png"
    plt.savefig(name)
    plt.close()

def main_1():
    # matrix for current choices
    #
    b = 2.08
    choices_matrix = np.zeros((201, 201))
    choices_matrix[100, 100] = 1  # D - 1, C - 0
    n_games = 120

    for i in tqdm(range(n_games)):
        payoff = payoff_matrix(choices_matrix, b)
        last_strat = np.copy(choices_matrix)
        choices_matrix = change_choices(choices_matrix,payoff)


        if i%5==0:
            plot_changes(last_strat,choices_matrix,i,b)


def main_2():
    #B = np.arange(1.99,2.01,0.0001)
    B = [1.98,1.99,1.995,1.999,2.001,2.005,2.01,2.02]
    n_games = 150
    D_sum = []
    choices_matrix_original = np.random.randint(2,size=(201,201)) #D - 1, C - 0
    plt.figure()
    plt.imshow(choices_matrix_original)
    plt.savefig("starting_choices.jpg")
    for b in tqdm(B):
        choices_matrix = np.copy(choices_matrix_original)
        for i in range(n_games):
            payoff = payoff_matrix(choices_matrix, b)
            last_strat = np.copy(choices_matrix)
            choices_matrix = change_choices(choices_matrix,payoff)


            if i%20==0:
                plot_changes(last_strat,choices_matrix,i,b)
        D_sum.append(np.sum(choices_matrix)/len(choices_matrix)**2)

    plt.figure()
    plt.plot(B,D_sum)
    plt.savefig("f_b.jpg")
    plt.close()



def change_strat(choices_matrix):
    return None
def change_choices(strategies,choises_matrix,payoff_matrix):
    new_strat = np.zeros((201,201))
    tmp_pay = np.zeros((201,201))
    for i in range(-1, 2):
        for j in range(-1, 2):
            tmp_pay = np.where(np.roll(np.roll(payoff_matrix, -j, 1), -i, 0)>tmp_pay,np.roll(np.roll(payoff_matrix, -j, 1), -i, 0),tmp_pay)
        

    for strat in strategies:
        for i in range(-1, 2):
            for j in range(-1, 2):
                if strat == 0:
                    new_strat = np.where(strat==0,0,new_strat)
                elif strat == 1:
                    new_strat = np.where(strat==0,1,new_strat)
                elif strat == 2:
                    new_strat = np.where(strat==0,1,new_strat)
                elif strat == 3:
                    pass
    
    return new_strat


def main_3():
    #B = np.arange(1.99,2.01,0.0001)
    B = [1.9]
    n_games = 150
    M = 5

    
    stat_matrix =np.random.randint(4,size=(201,201)) # 0 - always C, 1 - always D, 2 - tit for tat, 3- pavlov
    choices_matrix_original = np.random.randint(2,size=(201,201)) #D - 1, C - 0
    plt.figure()
    plt.imshow(choices_matrix_original)
    plt.savefig("starting_choices.jpg")
    
    for i in range(n_games):
        payoff = np.zeros((201,201))
        for b in tqdm(B):
            payoff += payoff_matrix(choices_matrix, b)
            choices_matrix = change_choices(choices_matrix,payoff)

        choices_matrix = change_strat(choices_matrix)

        if i%20==0:
            plot_changes(last_strat,choices_matrix,i,b)

    plt.figure()
    plt.plot(B,D_sum)
    plt.savefig("f_b.jpg")
    plt.close()


def make_gif(path):
    filenames = os.listdir(path)
    images = []
    for filename in tqdm(filenames):
        images.append(imageio.v2.imread(path + filename))

    kargs = {"duration": 0.5}
    imageio.mimsave("movie.gif", images, **kargs)
if __name__ == "__main__":
    #main_1()
    #make_gif("results_12/")
    main_2()
    make_gif("results_2/")
