import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
def get_chromosom():
    ret_string = ""
    np.random.seed(100)
    gene = np.random.randint(0, 2, size=512)
    return gene


def fit_function(actual_fit, matrix):

    cond_1 = np.sum(np.where(np.roll(matrix, -1, 0) == matrix)) * (-3) + np.sum(
        np.where(np.roll(matrix, -1, 1) == matrix)
    ) * (-3)

    matrix_left = np.where(np.roll(matrix, -1, 0) & np.roll(matrix, -1, 0) != matrix)
    copy_matrix = np.copy(matrix)
    copy_matrix = np.where(matrix_left == 1,copy_matrix,0)
    cond_2nd = (
        np.sum(np.where(np.roll(np.roll(copy_matrix, -1, 1), -1, 0) == matrix))
        * 8
        - np.sum(
            np.where(np.roll(np.roll(copy_matrix, -1, 1), -1, 0) != matrix)
        )
        * 5
    )

    cond_3rd = (
        np.sum(np.where(np.roll(np.roll(copy_matrix, -1, 1), -1, 0) == matrix))
        * 8
        - np.sum(
            np.where(np.roll(np.roll(copy_matrix, -1, 1), -1, 0) != matrix)
        )
        * 5
    )

    value = actual_fit + cond_1 + cond_2nd + cond_3rd
    return value


def coded_neighbors(matrix):
    """_summary_

    Args:
        matrix (_type_): _description_

    Returns:
        _type_: should working, in test for only 1 in 0 it works
    """
    coded_matrix = np.zeros(np.shape(matrix), dtype=int)
    n = 0
    for i in range(-1, 2):
        for j in range(-1, 2):
            coded_matrix += np.roll(np.roll(matrix, -j, 1), -i, 0) * 2**n
            n += 1

    return coded_matrix


def main():
    steps = 100
    fit = 0

    matrix_result = np.random.randint(0, 2, size=(50, 50))
    N = coded_neighbors(matrix_result)

    Rule = get_chromosom()
    for step in tqdm(range(steps)):
        fit = fit_function(fit, matrix_result)
        N = coded_neighbors(matrix_result)
        matrix_result = Rule[N]
        plt.imshow(matrix_result)
        plt.savefig(f"results/fig_{step}.png")
        plt.close()

    print(fit/steps)

if __name__ == "__main__":
    main()
