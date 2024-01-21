# Building Mutational Signatures in Cancer using Deep Bayesian Neural Nets

## SBS

This folder is organized into the following files:

### `sbs`

The `sbs.96.parquet` file contains all of the following the pyrimidine single nucleotide variants, N[{C > A, G, or T} or {T > A, G, or C}]N. 4 possible starting nucleotides x 6 pyrimidine variants x 4 ending nucleotides = 96 total combinations.

The `sbs.1536.parquet` file contains all of the following the pyrimidine single nucleotide variants, NN[{C > A, G, or T} or {T > A, G, or C}]NN. 16 (4x4) possible starting nucleotides x 6 pyrimidine variants x 16 (4x4) possible ending nucleotides = 1536 total combinations.

The `sbs.24576.parquet` file contains all of the following the pyrimidine single nucleotide variants, NNN[{C > A, G, or T} or {T > A, G, or C}]NNN. 64 (4x4x4) possible starting nucleotides x 6 pyrimidine variants x 64 (4x4x4) possible ending dinucleotides = 24576 total combinations.

The `sbs.24576.amino.parquet` file contains all of the following the pyrimidine single nucleotide variants {M,K}{M,K}NN[C > {A,G,T}]NN{M,K}{M,K} or {M,K}{M,K}NN[T > {A,C,G}]NN{M,K}{M,K} 2x2x4x4x6x4x4x2x2 = 24576 total combinations.ons

The `sbs.24576.nucl_strength.parquet` file contains all of the following the pyrimidine single nucleotide variants {W,S}{W,S}NN[C > {A,G,T}]NN{W,S}{W,S} or {W,S}{W,S}NN[T > {A,C,G}]NN{W,S}{W,S} 2x2x4x4x6x4x4x2x2 = 24576 total combinations.

The `sbs.24576.structure.parquet` file contains all of the following the pyrimidine single nucleotide variants {P,R}{P,R}NN[C > {A,G,T}]NN{P,R}{P,R} or {P,R}{P,R}NN[T > {A,C,G}]NN{P,R}{P,R} 2x2x4x4x6x4x4x2x2 = 24576 total combinations.

The `sbs.393216.parquet` file contains all of the following the pyrimidine single nucleotide variants, NNNN[{C > A, G, or T} or {T > A, G, or C}]NNNN. 256 (4x4x4x4) possible starting nucleotides x 6 pyrimidine variants x 256 (4x4x4x4) possible ending dinucleotides = 393216 total combinations.
