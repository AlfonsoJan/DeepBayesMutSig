#!/usr/bin/env python3
"""
Create a plot for each signature identified that depicts the proportion of the mutations for that signatur
"""
import sys
from pathlib import Path
import click
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
        nmf_folder (Path): Path of the NMF files".
        result_folder (Path): Path of the plots folder.

    Returns:
        int: Exit status (0 for success).
    """
    # Create SBS files
    sig_plots = signature_plots.SigPlots(nmf_folder, result_folder)
    sig_plots.create_plots()
    sig_plots.create_expected_plots()


if __name__ == "__main__":
    sys.exit(main())
