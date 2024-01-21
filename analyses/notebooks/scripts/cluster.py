#!/usr/bin/env python3
"""
This file provides for creating a 9 context SBS and cluster it down.
And calculates the perplexity for each file.
"""
import sys
from pathlib import Path
from mubelnet.utils import holdout_split
from jax import random
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from GenomeSigInfer.sbs import SBSMatrixGenerator
from mubelnet.utils import perplexity
from statkit.non_parametric import bootstrap_score
from statkit.types import Estimate

_PATH = Path(__file__).parent.parent.parent.parent
_NUCL_MAP = {
    "nucl_strength": {"A": "W", "T": "W", "C": "S", "G": "S"},
    "amino": {"A": "M", "C": "M", "G": "K", "T": "K"},
    "structure": {"A": "R", "C": "Y", "G": "R", "T": "Y"},
}
_NMF_INIT = "nndsvdar"
_BETA_LOSS = "kullback-leibler"
_SIGNATURES = 48


def create_9_context_df() -> pd.DataFrame:
    """
    Create a 9-context dataframe.

    Returns:
        pd.DataFrame: The 9-context dataframe.
    """
    files = (
        _PATH / "data" / "vcf" / "WES_Other.20180327.simple",
        _PATH / "data" / "vcf" / "WGS_Other.20180413.simple",
    )
    sbs_folder = _PATH / "analyses" / "SBS"
    ref_genome = _PATH / "analyses" / "ref_genome" / "GRCh37"
    genome = "GRCh37"
    sbs_9_context = SBSMatrixGenerator.generate_single_sbs_matrix(
        sbs_folder,
        files,
        ref_genome,
        genome,
        max_context=9,
    )
    return sbs_9_context


def cluster_df(df: pd.DataFrame, cluster: dict[str, str], cluster_type: str) -> None:
    """
    Clusters the mutation types in the given DataFrame based on the provided cluster dictionary.
    Saves the clustered DataFrame as a parquet file.

    Args:
        df (pd.DataFrame): The DataFrame containing mutation data.
        cluster (dict[str, str]): A dictionary mapping individual characters to their corresponding cluster.
        cluster_type (str): The type of the DataFrame.

    Returns:
        None
    """
    df_copy = df.copy()
    df_copy["MutationType"] = df_copy["MutationType"].apply(
        lambda x: "".join(cluster.get(c, c) for c in x[:2])
        + x[2:-2]
        + "".join(cluster.get(c, c) for c in x[-2:])
    )
    df_copy = df_copy.groupby("MutationType").sum()
    filename = (
        _PATH / "analyses" / "SBS" / f"sbs.{df_copy.shape[0]}.{cluster_type}.parquet"
    )
    df_copy.to_parquet(filename)


def cluster_sbs_9_context(sbs_9_context: pd.DataFrame) -> None:
    """
    Clusters the given sbs_9_context DataFrame and saves the results to a parquet file.

    Args:
        sbs_9_context (pd.DataFrame): The input DataFrame containing SBS 9-context data.

    Returns:
        None
    """
    sbs_9_context.to_parquet(
        _PATH / "analyses" / "SBS" / f"sbs.{sbs_9_context.shape[0]}.parquet"
    )
    for cluster_type in _NUCL_MAP:
        cluster_df(sbs_9_context, _NUCL_MAP[cluster_type], cluster_type)


from typing import List, Tuple
import numpy as np

def calculate_bootstrap_score(
    X_train: np.ndarray, X_test: np.ndarray, signatures: int, nmf_init: str, beta_loss: str
) -> float:
    """
    Calculate the bootstrap score for a given set of data.

    Args:
        X_train (np.ndarray): The training data.
        X_test (np.ndarray): The test data.
        signatures (int): The number of signatures.
        nmf_init (str): The initialization method for NMF.
        beta_loss (str): The beta loss function.

    Returns:
        float: The bootstrap score.
    """
    results = {
        "Train perplexity": None,
        "Test perplexity": None,
        "Test subset perplexity": None,
    }
    nmf = NMF(
        n_components=signatures,
        init=nmf_init,
        beta_loss=beta_loss,
        solver="cd" if beta_loss == "frobenius" else "mu",
        max_iter=10_000,
        tol=1e-15,
        random_state=43,
    ).fit(X_train)
    h = nmf.transform(X_train)
    unnormed_probs = h @ nmf.components_
    probs_sigprof_xtrct = unnormed_probs / unnormed_probs.sum(axis=-1, keepdims=True)
    # Compute performance on test set.
    is_inf = (probs_sigprof_xtrct == 0) & (X_train > 0)
    zero_rows = np.where(is_inf)[0]
    print(f"Deleting zero rows: {zero_rows}")
    X_train_subset = np.delete(X_train, zero_rows, axis=0)
    probs_sigprof_xtrct_subset = np.delete(probs_sigprof_xtrct, zero_rows, axis=0)
    pp_train = bootstrap_score(
        X_train_subset, probs_sigprof_xtrct_subset, metric=perplexity, random_state=43
    )
    results["Train perplexity"] = pp_train
    is_inf = (probs_sigprof_xtrct == 0) & (X_test > 0)
    X_test_eps = np.where(is_inf, np.finfo(np.float64).tiny, X_test)
    perplexity_score = bootstrap_score(
        X_test_eps, probs_sigprof_xtrct, metric=perplexity, random_state=43
    )
    return perplexity_score


def calculate_perplexity_per_file() -> None:
    """
    Calculate perplexity score for each file in the list and write the results to a file.

    Returns:
        None
    """
    files = [
        "sbs.24576.parquet",
        "sbs.24576.amino.parquet",
        "sbs.24576.nucl_strength.parquet",
        "sbs.24576.structure.parquet",
    ]
    perplexity_file = _PATH / "analyses" / "results" / "perplexity.txt"
    with open(perplexity_file, "w") as f:
        for file in files:
            print(file)
            sbs_file = _PATH / "analyses" / "SBS" / file
            df = pd.read_parquet(sbs_file)
            df = df[[df.columns[0]] + sorted(df.columns[1:])]
            all_genomes = np.array(df.iloc[:, 1:])
            X_train, X_test = holdout_split(random.PRNGKey(43), all_genomes)
            perplexity_score = calculate_bootstrap_score(
                X_train, X_test, _SIGNATURES, _NMF_INIT, _BETA_LOSS
            )
            f.write(f"{file}\t{perplexity_score.latex()}\n")
            print(f"Done for {file=}")


def main() -> None:
    """
    This is the main function that performs the clustering analysis.

    It calls the following functions:
    - create_9_context_df: Creates a dataframe with 9-context information.
    - cluster_sbs_9_context: Performs clustering on the 9-context dataframe.
    - calculate_perplexity_per_file: Calculates perplexity for each file.
    """
    sbs_9_context = create_9_context_df()
    cluster_sbs_9_context(sbs_9_context)
    calculate_perplexity_per_file()


if __name__ == "__main__":
    sys.exit(main())
