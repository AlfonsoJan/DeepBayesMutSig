#!/usr/bin/env python3
"""
This module provides functions for loading and processing mutation spectrum data.
"""
import sys
from pathlib import Path
import pandas as pd
from gabenet.utils import holdout_split
import jax.numpy as jnp
import haiku as hk


_PATH_COSMIC_FOLDER = Path(__file__).parent.parent.parent / "data"
_COSMIC_PATH = _PATH_COSMIC_FOLDER / "COSMIC_v3.4_SBS_GRCh37.txt"
COSMIC_WEIGHTS = pd.read_csv(_COSMIC_PATH, sep="\t", index_col=0).T
_PATH_SBS = Path(__file__).parent.parent / "SBS"


def _read_sbs_file(filename: str, folder: Path = _PATH_SBS) -> pd.DataFrame:
    """
    Read the SBS file and return a DataFrame.

    Args:
        filename (str): The name of the file to read.
        folder (Union[str, Path], optional): The folder path where the file is located. Defaults to _PATH_SBS.

    Returns:
        pd.DataFrame: The DataFrame containing the data from the SBS file.
    """
    df = pd.read_parquet(folder / filename)
    df.rename(columns={"MutationType": "Type"}, inplace=True)
    df = df.set_index("Type").sort_index()
    df = df.transpose()
    assert jnp.all(df.columns == COSMIC_WEIGHTS.columns)
    return df


def _build_mutation_spectrum(random_state: int = 42, context: int = 96) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Build the mutation spectrum.

    Args:
        random_state (int): Random seed for reproducibility. Default is 42.
        context (int): Context size. Default is 96.

    Returns:
        tuple[jnp.ndarray, jnp.ndarray]: A tuple containing the training and testing data.
    """
    key_seq = hk.PRNGSequence(random_state)
    df = _read_sbs_file(f"sbs.{context}.parquet")
    X_train, X_test = holdout_split(next(key_seq), df.to_numpy(), test_size=0.5)
    return X_train, X_test


def load_mutation_spectrum(random_state: int = 42, context: int = 96, folder: Path = _PATH_SBS) -> tuple[jnp.ndarray, jnp.ndarray, int]:
    """
    Load the mutation spectrum data.

    Args:
        random_state (int): Random seed for reproducibility. Default is 42.
        context (int): Context size. Default is 96.
        folder (Path): Path to the folder containing the data files. Default is _PATH_SBS.

    Returns:
        tuple: A tuple containing the training data, testing data, and context size.
    """
    if not (folder / f"train.{context}.npy").exists():
        X_train, X_test = _build_mutation_spectrum(random_state=random_state)
        jnp.save(folder / f"train.{context}.npy", X_train)
        jnp.save(folder / f"test.{context}.npy", X_test)
    else:
        X_train = jnp.load(folder / f"train.{context}.npy")
        X_test = jnp.load(folder / f"test.{context}.npy")
    return X_train, X_test, context


def main():
    """
    The main function of the script.
    """
    print("Running for 96")
    load_mutation_spectrum(context=96)


if __name__ == "__main__":
    sys.exit(main())
