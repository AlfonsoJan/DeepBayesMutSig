#!/usr/bin/env python3
"""
Create nmf files and deconmpose mutational signatures from NMF results.

And calculates the cosine similarity between each file's signature data and a reference column.

This script takes input parameters for the smallest context SBS file (--sbs-96),
larger context SBS files (--sbs-lg), the number of signatures (--sigs), the cosmic
file (--cosmic), NMF initialization method (--nmf-init), and the beta loss function
for NMF (--beta-loss). It then prints the provided parameters.
"""
import sys
import click
from pathlib import Path
from GenomeSigInfer.nmf import NMFMatrixGenerator

def unique_paths(ctx, param, value):
    """
    Validate that the provided paths are unique.

    Args:
        ctx (click.Context): The Click context.
        param (click.Parameter): The Click parameter.
        value (List[Path]): The provided paths.

    Returns:
        List[Path]: The validated unique paths.

    Raises:
        click.BadParameter: If the paths are not unique.
    """
    # Check if the provided paths are unique
    if len(value) != len(set(value)):
        raise click.BadParameter("Paths must be unique.")
    return value


def positive_number(ctx, param, value):
    """
    Validate that the provided number is positive.

    Args:
        ctx (click.Context): The Click context.
        param (click.Parameter): The Click parameter.
        value (int): The provided number.

    Returns:
        int: The validated positive number.

    Raises:
        click.BadParameter: If the number is not positive.
    """
    if value is not None and value <= 0:
        raise click.BadParameter("The number of signatures must be a positive integer.")
    return value

@click.command()
@click.option(
    "--sbs-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path, exists=True),
    default="project/NMF",
    help="The location of the SBS files",
)
@click.option("--sigs", type=click.INT, help="The number of signatures", required=True, callback=positive_number)
@click.option(
    "--cosmic",
    type=click.Path(file_okay=True, dir_okay=False, exists=True, path_type=Path),
    help="The cosmic file",
    required=True,
)
@click.option(
    "--nmf-init",
    type=click.Choice(["random", "nndsvd", "nndsvda", "nndsvdar", "custom"]),
    default="nndsvda",
    help="NMF initialization method. Choose from 'random', 'nndsvd', 'nndsvda', 'nndsvdar', or 'custom'.",
)
@click.option(
    "--beta-loss",
    type=click.Choice(['frobenius', 'kullback-leibler', 'itakura-saito']),
    default='frobenius',
    help="Beta loss function for NMF. Choose from 'frobenius', 'kullback-leibler', or 'itakura-saito'.",
)
@click.option(
    "--nmf-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default="project/NMF",
    help="The location for the NMF files",
)
@click.option(
    "--res-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default="project/results",
    help="The location for the NMF files",
)
def main(sbs_dir: Path, sigs: int, cosmic: Path, nmf_init: str, beta_loss: str, nmf_dir: Path, res_dir: Path) -> int:
    """
    Main function for the script.

    Args:
        sbs_96 (Path): Path to the smallest context SBS file.
        sbs_lg (tuple[Path]): Paths to the larger context SBS files.
        sigs (int): Number of signatures.
        cosmic (Path): Path to the cosmic file.
        nmf_init (str): NMF initialization method.
        beta_loss (str): Beta loss function for NMF.
        nmf_dir (Path): The location for the NMF files.
        res_dir (Path): The location for the figures.

    Returns:
        int: Exit code.
    """
    NMFMatrixGenerator.generate_nmf_matrix(
        sbs_folder=sbs_dir,
        signatures=sigs,
        cosmic=cosmic,
        nmf_folder=nmf_dir,
        nmf_init=nmf_init,
        beta_los=beta_loss,
        result_folder=res_dir
    )

if __name__ == "__main__":
    sys.exit(main())
