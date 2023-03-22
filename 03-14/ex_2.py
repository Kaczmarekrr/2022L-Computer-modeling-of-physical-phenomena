import numpy as np
from matplotlib import pyplot as plt


params = {0:[6,20_000],1:[20,10_000],2:[200,500]}


def Wigner_surmise_GOE(bins):
    wig = np.zeros(len(bins),dtype = np.float64)
    wig = 0.5*np.pi * bins * np.exp(-0.25*np.pi*bins**2)
    return wig

def Wigner_surmise_GUE(bins):
    wig = np.zeros(len(bins),dtype = np.float64)
    wig = (32/np.pi**2) * np.exp((-4/np.pi)*bins**2) * bins**2
    return wig


def GOE_sim(N,n_samples):
    result_list = np.zeros((n_samples, N-1))

    for i in range(n_samples):
        h = np.random.randn(N, N)
        h = (h + h.T) / 2
        eigen_values, _ = np.linalg.eigh(h)
        result_list[i, :] = np.diff(np.sort(eigen_values))/np.mean(np.diff(eigen_values))


    eigen_diffs = result_list


    eigen_diffs = eigen_diffs.flatten()
    return eigen_diffs


def GUE_sim(N,n_samples):
    result_list = np.zeros((n_samples, N-1))

    for i in range(n_samples):
        h_1 = np.random.randn(N,N)
        h_2 = np.random.randn(N,N)
        h = (h_1 + h_2 * 1j)/np.sqrt(2)
        h = (h + (np.conjugate(h)).T) / 2

        eigen_values, _ = np.linalg.eigh(h)
        eigen_values = np.sort(eigen_values)
        #print(eigen_values
        result_list[i, :] = np.diff(eigen_values)/np.mean(np.diff(eigen_values))

        # TODO need to change to use only middle indexes to of eigen diffs


    eigen_diffs = result_list/np.mean(result_list)


    eigen_diffs = eigen_diffs.flatten()
    return eigen_diffs


params = ([8,100],[200,100],[500,100])
bins_per = [20,80,120]
for i,x in enumerate(params):
    GOE_eigen_diffs = GOE_sim(x[0],x[1])
    GUE_eigen_diffs = GUE_sim(x[0],x[1])



    plt.figure()
    plt.subplot(1,2,1)
    n, bins, _ = plt.hist(
            GOE_eigen_diffs, bins_per[i], density=True, facecolor="cyan", alpha=0.75
        )
    Wigner_GOE = Wigner_surmise_GOE(bins)
    plt.title("GOE")
    plt.ylim(0,1)
    plt.xlim(0,3.5)
    plt.plot(bins,Wigner_GOE,"r-")

    plt.subplot(1,2,2)
    n, bins, _ = plt.hist(
            GUE_eigen_diffs, bins_per[i], density=True, facecolor="cyan", alpha=0.75
        )
    GUE_wigner = Wigner_surmise_GUE(bins)
    plt.title("GUE")
    plt.plot(bins,GUE_wigner,"r-")
    plt.ylim(0,1)
    plt.xlim(0,4)
    plt.savefig(f"GOE-GUE comparition_{x[0]}-{x[1]}.png")