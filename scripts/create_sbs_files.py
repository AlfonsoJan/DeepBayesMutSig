#!/usr/bin/env python3
"""
Create mutliple SBS files. With increasing context.

The sbs.96.txt file contains all of the following the pyrimidine single nucleotide variants, N[{C > A, G, or T} or {T > A, G, or C}]N.
*4 possible starting nucleotides x 6 pyrimidine variants x 4 ending nucleotides = 96 total combinations.*

The sbs.1536.txt file contains all of the following the pyrimidine single nucleotide variants, NN[{C > A, G, or T} or {T > A, G, or C}]NN.
*16 (4x4) possible starting nucleotides x 6 pyrimidine variants x 16 (4x4) possible ending nucleotides = 1536 total combinations.*

The sbs.24576.txt file contains all of the following the pyrimidine single nucleotide variants, NNN[{C > A, G, or T} or {T > A, G, or C}]NNN.
*16 (4x4) possible starting nucleotides x 16 (4x4) nucleotides x 6 pyrimidine variants x 16 (4x4) nucleotides x 16 (4x4) possible ending dinucleotides = 24576 total combinations.*
"""
import sys
import click
from pathlib import Path
from GenomeSigInfer.sbs import SBSMatrixGenerator


@click.command()
@click.option(
    "--sbs-out",
    type=click.Path(path_type=Path),
    default="project/SBS",
    prompt="The location for the SBS files",
)
@click.option(
    "--ref-genome",
    type=click.Path(path_type=Path),
    prompt="The parent folder of the reference genome",
)
@click.option(
    "--vcf",
    type=click.Path(file_okay=True, dir_okay=False, exists=True, path_type=Path),
    multiple=True,
    prompt="The VCF file(s)",
    required=True,
    help="The VCF file(s). At least one file is required.",
)
@click.option(
    "--genome",
    type=click.Choice(["GRCh37", "GRCh38"]),
    default="GRCh37",
    help="Choose genome version.",
)
def main(sbs_out: Path, ref_genome: Path, vcf: tuple, genome: str) -> int:
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
    SBSMatrixGenerator.generate_sbs_matrix(
        project=sbs_out, ref_genome=ref_genome, vcf=vcf, genome=genome
    )


if __name__ == "__main__":
    sys.exit(main())
