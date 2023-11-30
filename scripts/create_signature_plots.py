#!/usr/bin/env python3
"""
Create nmf files and deconmpose mutational signatures from NMF results.
And calculates the `cosine similarity` between each file's signature data and a reference column.
"""
import sys
import click
from pathlib import Path
from GenomeSigInfer.figures import signature_plots


@click.command()
@click.option(
    "--nmf-folder",
    type=click.Path(path_type=Path, dir_okay=True, file_okay=False, exists=True),
    default="project/NMF",
    help="The location of the NMF files",
)
@click.option(
    "--result-folder",
    type=click.Path(path_type=Path, dir_okay=True, file_okay=False),
    default="project/results",
    help="The location for the plots",
)
def main(nmf_folder: Path, result_folder: Path) -> int:
    """
    Main entry point of the script.

    Args:
        project (Path): Path of the project folder.
        vcf (tuple): Tuple pf the vcf files list.
        genome (str): Reference genome.
        bash (bool): If you want to download using bash.

    Returns:
        int: Exit status (0 for success).
    """
    # Create SBS files
    sig_plots = signature_plots.SigPlots(nmf_folder, result_folder)
    sig_plots.create_plots()
    sig_plots.create_expected_plots()


if __name__ == "__main__":
    sys.exit(main())
