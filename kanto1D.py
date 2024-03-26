import sys

import numpy as np

import pandas as pd

from ot import dist, sinkhorn2

from itertools import combinations

from scipy.stats import wasserstein_distance



# ====================================================

# code

def kanto_1d(sample1: np.ndarray,

             sample2: np.ndarray,

             reduce=sum) -> float:

    """

    Computes the sum of a series of kantorovich distances on cpu.


    Args:

        sample1: 2D array where the first dimension is samples per distribution (e.g. cells)

            and the second dimension is distributions (e.g. genes)

        sample2: 2D array where the first dimension is samples per distribution (e.g. cells)

            and the second dimension is distributions (e.g. genes)

        reduce (optional): function to use to reduce the per-gene distances. Defaults to sum.



    Returns:

        the sum of element-wise kantorovich distances

    """

    assert sample1.ndim == 2 and sample2.ndim == 2, "expects 2D arrays"

    assert sample1.shape[1] == sample2.shape[1], "nb of distributions should be the same"



    nb_genes = sample1.shape[1]



    result = [wasserstein_distance(sample1[:, i], sample2[:, i]) for i in range(nb_genes)]



    if reduce is None:

        return result



    return reduce(result)


if __name__ == "__main__":

    sample1 = pd.read_csv(sys.argv[1], sep='\t').to_numpy()

    sample2 = pd.read_csv(sys.argv[2], sep='\t').to_numpy()



    result = kanto_1d(sample1, sample2, reduce=None)

    print(pd.DataFrame(result).to_csv())

