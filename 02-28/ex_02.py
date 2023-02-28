import numpy as np
from matplotlib import pyplot as plt


def n_kick(n:int,k_const:float):
    """_summary_

    Args:
        n (int): numbers of kicks

    Returns:
        np.ndarray: results for the plot
    """    

    r, g, b = np.random.random(3)
    color = [r,g,b]
    k_const = 1.2
    x_n, p_n = np.random.random(2)*2*np.pi

    result_arr = np.zeros((n+1,2))
    result_arr[0] = [p_n, x_n] %(2*np.pi)

    for i in range(1,n+1):
        p_n = (p_n + k_const*np.sin(x_n))%(2*np.pi)
        x_n = (x_n + p_n) % (2*np.pi)

        
        result_arr[i,0] = p_n
        result_arr[i,1] = x_n


    return result_arr, color
        
def kicks_k_trajectories(n,k,k_const):
    result_arr = np.zeros((k,n+1,2))
    results_colors = np.zeros((k,3))
    for i in range(k):
        result_arr[i,:,:],results_colors[i] = n_kick(n,k_const=k_const)
        

    return result_arr,results_colors

def plot(data:np.ndarray,colors:np.ndarray,save_fig,k_const):
    """_summary_

    Args:
        data (np.ndarray): _description_
    """    

    for i in range(len(data)):
        data_0 = data[i]
        plt.plot(data_0[:,1],data_0[:,0],".",color=colors[i],label="p")
    plt.grid()
    plt.xlabel("x")
    plt.ylabel("p")
    plt.title(f"Results for {len(data)} trajectories, {len(data_0[:,1])-1} kicks, K = {k_const}")

    if save_fig:
        plt.savefig(f"results/ex_02_{k_const}.png")

    else:plt.show()

if __name__ == '__main__':

    k_trajectories = 100
    n_kicks = 1000
    k_consts = [1.2,2.1,5.5,7]
    for k_const in k_consts:
        save_fig = True
        results_arr_1,colors = kicks_k_trajectories(n_kicks,k_trajectories,k_const)
        plot(results_arr_1,colors,save_fig,k_const)
