import numpy as np
import numpy.typing as npt
import pandas as pd
import plotly.express as px
import vdata as vd
import os
from sys import argv 


from typing import Callable, TypeVar

_R = TypeVar("_R")

# Distances functions ---------------------------------------------------------
def kanto_1d(
    u: npt.NDArray[np.float32], v: npt.NDArray[np.float32], p: int = 1
) -> float:
    """Computes the Kantorovich distance between two 1D distributions.

    Args:
        u: first 1D distribution
        v: second 1D distribution
        p (optional): the p-norm to apply. Defaults to 1.

    Returns:
        The Kantorovich distance in 1D between u and v.
    """
    assert u.ndim == 1 and v.ndim == 1
    all_values = np.concatenate((u, v))
    all_values.sort(kind="mergesort")

    # Compute the differences between pairs of successive values of u and v.
    deltas = np.diff(all_values)

    # Get the respective positions of the values of u and v among the values of both distributions.
    cdf_indices_u = np.sort(u).searchsorted(all_values[:-1], "right")
    cdf_indices_v = np.sort(v).searchsorted(all_values[:-1], "right")

    # Calculate the CDFs of u and v using their weights, if specified.
    cdf_u = cdf_indices_u / u.size
    cdf_v = cdf_indices_v / v.size

    return float(np.power(np.sum(np.multiply(np.abs(cdf_u - cdf_v), deltas)), p))

def NoReduce(arr: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    return arr

def multigene_kanto_1d(
    u: npt.NDArray[np.float32],
    v: npt.NDArray[np.float32],
    reduce: Callable[[npt.NDArray[np.float32]], _R] = np.sum,
    p: int = 1,
    normalization_max= 10,
) -> _R:
    """Computes 1D Kantorovich distances over each column of 2D distributions.

    Args:
        u: 2D array where the first dimension is samples per distribution (e.g. cells)
            and the second dimension is distributions (e.g. genes)
        v: 2D array where the first dimension is samples per distribution (e.g. cells)
            and the second dimension is distributions (e.g. genes)
        reduce (optional): function to use to reduce the per-gene distances. Defaults to sum.
        p (optional): the p-norm to apply. Defaults to 1.
        normalization_max (optional): if set to a floating point value, sample values for each dimension will be
            rescaled to fit in the range [0, normalization_max]

    Returns:
        The (reduced) series of 1D Kantorovich distances.
    """
    assert u.ndim == 2 and v.ndim == 2, "expects 2D arrays"
    assert u.shape[1] == v.shape[1], "nb of distributions should be the same"

    if normalization_max is not None:
        u *= normalization_max / u.max(axis=0)
        v *= normalization_max / v.max(axis=0)

    nb_genes = u.shape[1]
    return reduce(np.array([kanto_1d(u[:, i], v[:, i], p=p) for i in range(nb_genes)]))


def main():

    D=argv[1]
    P=argv[2]
    cwd=argv[3]
    percent_valid=float(argv[4])
    m=float(argv[5])


    os.chdir(str(cwd)+"/OG"+str(D)+"/"+str(P)+"/Data")

    # Load data -------------------------------------------------------------------
    gene_names = pd.read_csv("panel_genes.txt", sep=" ", header=None, names=["id", "name"]).name
    
    real_data = pd.read_csv("panel_real.txt", sep="\t", header=None, index_col=0)
    real_data.index = pd.Index(pd.concat((pd.Series(["timepoint"]), gene_names)))
    real_data = real_data.drop(index="Stimulus")
    real_data = vd.TemporalDataFrame(real_data.T, time_col_name="timepoint", name="real")

    simulated_data = pd.read_csv("panel_simulated.txt", sep="\t", header=None, index_col=0)
    simulated_data.index = pd.Index(pd.concat((pd.Series(["timepoint"]), gene_names)))
    simulated_data = simulated_data.drop(index="Stimulus")
    simulated_data = vd.TemporalDataFrame(simulated_data.T, time_col_name="timepoint", name="real")

    # Compute distances -----------------------------------------------------------
    distances_1D = []

    for tp in real_data.timepoints:
        distances_1D.append(
            multigene_kanto_1d(
                real_data[tp].values.copy(),
                simulated_data[tp].values.copy(),
                reduce=NoReduce,
            )
        )

    # Plot and save the result
    os.chdir(str(cwd)+"/OG"+str(D)+"/"+str(P)+"/Results")
   
    np.save('distances_1D.csv', distances_1D)

    # Plot summed distances -------------------------------------------------------
    fig = px.bar(
        pd.DataFrame(
            np.sum(distances_1D, axis=1),
            index=real_data.timepoints.astype(str),
            columns=["distance 1D"],
        )
    )
    fig.write_image("Sum_distances_1d.png", width=600, height=400)


    # # Plot distances per gene -----------------------------------------------------
    df = pd.DataFrame(distances_1D, columns=gene_names[1:])
    # sort columns (gene names) alphabetically
    df = df.reindex(sorted(df.columns), axis=1)
    max_dist = df.max(axis=None)
    print(max_dist)

    if m !=0 :
        df.iloc[[0],[0]]=m

    max_valid_distance = percent_valid*max_dist # percentage of values to be considered as correct
    good_fit_dist = max_valid_distance / max_dist  # borne couleur à KD = max_valid_distance
    bad_fit_dist = (max_valid_distance + (max_dist - max_valid_distance) / 2) / max_dist  
    # borne couleur à la moitié de l'intervalle [max_valid_distance, max KD]
    cmap = [
        (0.0, "lightgreen"),
        (good_fit_dist, "lightgreen"),
        (good_fit_dist, "sandybrown"),
        (bad_fit_dist, "sandybrown"),
        (bad_fit_dist, "red"),
        (1.0, "red"),
    ]


    fig = px.imshow(df, color_continuous_scale=cmap)

    fig.write_image("Gene_distances_1d.WT.png", width=len(gene_names)*16, height=400)


if __name__ == "__main__":
    main()




