"""Calculate eigenvalues of nsample random matrices drawn from GOE
ensamble (N × N matrix size). Make a histogram (normalized) of
eigenvalues and compare with the analytical Wigner semicircle law. For this
task we accumulate all eigenvalues from all generated matrices.
"""

import numpy as np
from matplotlib import pyplot as plt



def compute_and_plot():
    """_summary_

    Args:

    Returns:
        _type_: _description_
    """

    params = {0:[6,20_000],1:[20,10_000],2:[200,500]}

    for test_case in range(3):
        N, n_sample = params[test_case]

        def wigner(bins, N):
            wig = np.zeros(len(bins))
            wig = (2 / (np.pi * 2 * N)) * np.sqrt(2 * N - bins**2,where=bins**2<=2*N)
            wig = np.where(bins**2>2*N,0,wig)
            return wig

        plt.subplot(1,3,test_case+1)
        plt.ylim(0,0.2)
        plt.xlim(-21,21)
        result_list = np.zeros((n_sample, N))

        for i in range(n_sample):
            h = np.random.randn(N, N)
            h = (h + h.T) / 2

            eigen_values, _ = np.linalg.eigh(h)
            result_list[i, :] = eigen_values


        


        result_list = result_list.reshape(n_sample * N)
        n, bins, _ = plt.hist(
            result_list, 50, density=True, facecolor="cyan", alpha=0.75
        )
        
        wigners = wigner(bins, N)
        plt.plot(bins, wigners, "r-", linewidth=2)
        plt.title(f"N = {N}, n_sample = {n_sample}")
    plt.suptitle("Comparision of wigners semi-circles with histogram of eigen values of Hamiltonian from GOE")
    plt.show()


compute_and_plot()






"""Calculate the histogram of energy spacings for GOE and GUE ensembles.
Normalize the accumulated energy spacings to unit mean level spacing
(divide by the average)"""



