#!/usr/bin/env python3
"""
Generate a 9-substitution context matrix and perform clustering
"""
import sys
from pathlib import Path
import pandas as pd
from GenomeSigInfer.sbs import SBSMatrixGenerator

_PATH = Path(__file__).parent.parent.parent / "data" / "vcf"
_NUCL_MAP = {
    "nucl_strength": {"A": "W", "T": "W", "C": "S", "G": "S"},
    "amino": {"A": "M", "C": "M", "G": "K", "T": "K"},
    "structure": {"A": "R", "C": "Y", "G": "R", "T": "Y"},
}


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
        Path(__file__).parent.parent
        / "sbs"
        / f"sbs.{df_copy.shape[0]}.{cluster_type}.parquet"
    )
    df_copy.to_parquet(filename)


def main() -> None:
    """
    Generate a 9-substitution context matrix and perform clustering.

    This function generates a 9-substitution context matrix using the specified input files and reference genome.
    It then saves the matrix as a Parquet file and performs clustering on the matrix using different cluster types.

    Args:
        None

    Returns:
        None
    """
    files = (_PATH / "WES_Other.20180327.simple", _PATH / "WGS_Other.20180413.simple")
    sbs_9_context = SBSMatrixGenerator.generate_single_sbs_matrix(
        "./analyses/SBS/",
        files,
        "./analyses/ref_genome/GRCh37",
        "GRCh37",
        max_context=9,
    )
    sbs_9_context.to_parquet(
        Path(__file__).parent.parent / "sbs" / f"sbs.{sbs_9_context.shape[0]}.parquet"
    )
    for cluster_type in _NUCL_MAP:
        cluster_df(sbs_9_context, _NUCL_MAP[cluster_type], cluster_type)


if __name__ == "__main__":
    sys.exit(main())
