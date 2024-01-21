[![Docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://osf.io/t6j7u/wiki/home/) 
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Building Mutational Signatures in Cancer using Deep Bayesian Neural Nets

<sub><sub>**J.A. Busker**</sub></sub>    
<sub><sub>**Hanze University of Applied Sciences, Groningen**</sub></sub>    
<sub><sub>**UMCG department of Epidemiology, Groningen**</sub></sub>

This project aims to refine the statistical model and the current representation of mutations in building mutational signatures in cancer using deep Bayesian neural nets. Cancer is characterized by uncontrolled growth of cells, which is primarily acquired by mutation of the genome. These mutations have accrued by exposure to DNA-damaging processes coming from within (endogenous) and from outside (exogenous) throughout the course of life. By looking at the direct context of the mutations, we can infer the way these mutations were created.

## Table of Contents

- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Installation](#installation)

## Introduction

Cancer, characterized by uncontrolled cell growth, accumulate mutations. The mutations that occur in the context of cancer development are a result of exposure to various DNA-damaging processes and accumulate throughout life. The sources of these DNA-damaging processes include both endogenous and exogenous factors. These genetic variations result in unique "mutational signatures" within the DNA sequence.

This project aims to refine the statistical model and the current representation of mutations by building mutational signatures of cancer using deep bayesian neural networks. Additionally, there is a plan to expand the representation to capture more context. Increasing the context involves subdividing mutations. By looking at an extra nucleotide on each side. This expansion aimed to reveal contextual imprints associated with surrounding nucleotides, employing techniques like Latent Dirichlet Allocation (LDA). The methodology centered on curating and quality controlling variant calling samples and replicating mutational signatures using Non-negative Matrix Factorization (NMF).

Significant advancements were achieved in this study, particularly in the analysis of Single Base Substitutions (SBS) such as SBS7a, SBS22a, SBS10a, SBS13, and SBS17b, which provided a deeper understanding of their complex etiology through an expanded contextual framework. A novel single-run NMF approach for decomposing mutational signatures was introduced, marking a departure from traditional multi-iteration methods. The future vision involves enhancing the reliability of this single-run approach and integrating user-interactive clustering mechanisms, aiming to connect advanced genomic analysis with personalized cancer treatment strategies.

The significance of this project lies not only in advancing our understanding of cancer mutational signatures but also in its implications for cancer diagnosis, treatment, and prevention.

## Project Structure

The project is organized into the following folders:

### `analyses`

The `analyses` folder is intended for storing all the results, notebooks, created data and more.

### `data`

The `data` folder is intended for storing datasets or data files used in the project.

### `documents`

The `documents` folder contains important documents related to school, research, or any relevant materials. These documents may include academic papers, project proposals, and other resources that contribute to the understanding and development of the project.

### `download_script.py`

The `download_script.py` file downloads and unzips the VCF files used for this study, into the data\vcf folder.


## References

- Alexandrov, Ludmil B., et al. "The repertoire of mutational signatures in human cancer." Nature 578.7793 (2020): 94-101.
- Bergstrom, Erik N., et al. "SigProfilerMatrixGenerator: a tool for visualizing and exploring patterns of small mutational events." BMC genomics 20.1 (2019): 1-12.
- Degasperi, Andrea, et al. "Substitution mutational signatures in whole-genome–sequenced cancers in the UK population." Science 376.6591 (2022): abl9283.
- Kim, Yoo-Ah, et al. "Mutational signatures: From methods to mechanisms." Annual Review of Biomedical Data Science 4 (2021): 189-206.
- Zhou, Mingyuan, Yulai Cong, and Bo Chen. "Augmentable gamma belief networks." The Journal of Machine Learning Research 17.1 (2016): 5656-5699.
