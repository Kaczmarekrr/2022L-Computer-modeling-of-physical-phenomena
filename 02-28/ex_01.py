import numpy as np
from matplotlib import pyplot as plt


def n_kick(n:int,x_n,p_n):
    """_summary_

    Args:
        n (int): numbers of kicks

    Returns:
        np.ndarray: results for the plot
    """    
    k = 1.2

    result_arr = np.zeros((n+1,2))
    result_arr[0] = [x_n, p_n]

    for i in range(1,n+1):
        p_n = p_n + k*np.sin(x_n)
        x_n = (x_n + p_n) % (2*np.pi)

        result_arr[i,0] = x_n
        result_arr[i,1] = p_n


    return result_arr
        

def plot(data_0:np.ndarray,data_1:np.ndarray,params:list[int]):
    """_summary_

    Args:
        data (np.ndarray): _description_
    """    
    plt.figure()

    plt.subplot(2,1,1)
    plt.plot(data_0[:,0],"b.",label="p")
    plt.plot(data_0[:,1],"r.",label="x",)
    
    plt.grid()
    plt.xlabel("n")
    plt.ylabel("x")
    plt.title(f"Results for x_0 = {params[0]}, p_0 = {params[1]}")

    plt.subplot(2,1,2)
    plt.plot(data_1[:,0],"b.",label="p")
    plt.plot(data_1[:,1],"r.",label="x")
    plt.grid()
    plt.xlabel("n")
    plt.ylabel("p")
    plt.title(f"Results for x_0 = {params[2]}, p_0 = {params[3]}")
    plt.legend(loc="best")
    plt.tight_layout()


    #plt.show()
    plt.savefig("results/ex_01.png")

if __name__ == '__main__':
    n = 50
    x_n = 3.0
    p_n = 1.9
    params = [x_n,p_n]
    results_arr_1 = n_kick(n,x_n,p_n)
    x_n = 3.0
    p_n = 1.8999
    params.append(x_n)
    params.append(p_n)
    results_arr_2 = n_kick(n,x_n,p_n)

    
    print(np.max(results_arr_1[:,1]),np.min(results_arr_1[:,1]))
    print(np.max(results_arr_2[:,1]),np.min(results_arr_2[:,1]))
    plot(results_arr_1,results_arr_2,params)
