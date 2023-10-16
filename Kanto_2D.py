# Compute Kanto 2D distances between experimental and CARDAMOM-simulated distributions

import multiprocessing as mp
from tqdm.auto import tqdm
from math import factorial
from ot import dist, sinkhorn2
from dataclasses import dataclass
from itertools import combinations, repeat
from scipy.stats import wasserstein_distance
import time
from sys import argv 
import os
import numpy as np
import matplotlib.pyplot as plt

# ====================================================
# code

@dataclass
class Parameters:
    nb_cells_1: int
    nb_cells_2: int
    sample_1: np.ndarray
    sample_2: np.ndarray
    reg: int
    stop_thr: float

def kanto_core(genes, params):
    subsample1 = params.sample_1[:, genes]
    subsample2 = params.sample_2[:, genes]
    distances = dist(subsample1, subsample2, metric="sqeuclidean")
    distances /= distances.max()
    return sinkhorn2(np.ones(params.nb_cells_1), np.ones(params.nb_cells_2), distances, 1 / params.reg, stopThr=params.stop_thr)

def kanto_2d(sample1: np.ndarray, 
             sample2: np.ndarray, 
             reg=10,
             stop_thr=1e-9,
             nb_cores=1,
             reduce=np.mean) -> float:
    """
    Computes the sum for all pairs of genes of sinkhorn divergences between the provided #cells*#genes samples.

    Uses the POT sinkhorn implementation.

    Args:
        sample1: first sample (#cells * #genes).
        sample2: first sample (#cells * #genes).
        result: array of sinkhorn divergences (#gene_pairs).
        reg (optional): regularization parameter. Defaults to 10. Caution: inverse parametrization compared to POT.
        stop_thr (optional): stop threashold. Defaults to 1e-9.
        nb_cores: number of cores to use (if greater than 1, uses multiprocessing)
        reduce (optional): function to use to reduce the per-gene distances. Defaults to sum.

    Returns:
        float: the mean of sinkhorn divergences.
    """
    nb_genes = sample1.shape[1]
    nb_cells_1 = sample1.shape[0]
    nb_cells_2 = sample2.shape[0]
    assert sample1.shape[1] == sample2.shape[1]
    total = int(factorial(nb_genes) / (2 * factorial(nb_genes - 2)))
    if nb_cores == 1:
        print('1')
        result = []
        for genes in tqdm(combinations(range(nb_genes), 2), total=total):
            subsample1 = sample1[:, genes]
            subsample2 = sample2[:, genes]
            distances = dist(subsample1, subsample2, metric="sqeuclidean")
            distances /= distances.max()
            result.append(sinkhorn2(np.ones(nb_cells_1), np.ones(nb_cells_2), distances, 1 / reg, stopThr=stop_thr))
    else:
        print('multi')
        parameters = Parameters(nb_cells_1, nb_cells_2, sample1, sample2, reg, stop_thr)
        genes = combinations(range(nb_genes), 2)
        with mp.Pool(nb_cores) as pool:
            result = pool.starmap(kanto_core, tqdm(zip(genes, repeat(parameters)), total=total))  
    if reduce is None:
        return result
    return reduce(result)
# ====================================================   

def main():

    D=argv[1]
    P=argv[2]
    cwd=argv[3]

    os.chdir(str(cwd)+"/OG"+str(D)+"/"+str(P)+"/Data")

    # Open dataset
    data_real = np.loadtxt('panel_real.txt', dtype=float, delimiter='\t')[1:, 1:]
    data_real[0, :] = np.loadtxt('panel_real.txt', dtype=float, delimiter='\t')[0, 1:]
    data_netw = np.loadtxt('panel_simulated.txt', dtype=float, delimiter='\t')[1:, 1:]
    data_netw[0, :] = np.loadtxt('panel_simulated.txt', dtype=float, delimiter='\t')[0, 1:]
    
    # Genes
    names = np.loadtxt('panel_genes.txt', dtype='str')[:, 1]  # Load  the names of the genes

    # Sort genes
    order = np.argsort(names[1:])
    names = [names[0]] + list(names[order+1])

    # Order the data
    order = [0] + list(order+1)
    data_real = data_real[order,:]
    data_netw = data_netw[order,:]

    # Times
    t_real = list(set(data_real[0, :]))  # Time of each sample for the reference datasets
    t_real.sort()
    t_netw = list(set(data_netw[0, :]))
    t_netw.sort()

    k=np.array([])

    for i in range (0 , np.shape(t_real)[0]):
        z1r=np.where(data_real[0]==t_real[i])
        d1r=data_real[:,z1r]
        z1n=np.where(data_netw[0]==t_netw[i])
        d1n=data_netw[:,z1n]
        d1r=d1r[:,0,:]
        d1n=d1n[:,0,:]
        d1r=np.transpose(d1r)
        d1n=np.transpose(d1n)
        k=np.append(k, round(kanto_2d(d1r,d1n),3))
    #print(k)

    os.chdir(str(cwd)+"/OG"+str(D)+"/"+str(P)+"/Results")
    # Plot and save the result
    fig=plt.bar(range(k.shape[0]),k) 
    ti='Kanto.2D.pdf'
    plt.savefig(ti)

    with open('kanto_distances', 'w') as f:
        for line in k:
           f.write(str(line))
           f.write('\n')

if __name__ == "__main__":
    main()
