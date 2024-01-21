# Building Mutational Signatures in Cancer using Deep Bayesian Neural Nets

## notebooks

This folder is organized into the following files:

### `notebooks`

The `best_nmf_combinations.ipynb` is about finding the best NMF paramterrs combinations.

The `create_nmf_files.ipynb` is about creating NMF files and decomposing.

The `create_sbs_files.ipynb` is about creating SBS files and compressing them stepwise.

The `create_signature_plots.ipynb` is about finding the best NMF paramterrs combinations.

The `data_discovery.ipynb` is about exploring the VCF files.

The `meta_signatures.ipynb` is about the meta signatures and the results from the neural network.

The `scripts` fodler contains the following files:

* `cluster`: This file provides for creating a 9 context SBS and cluster it down. And calculates the perplexity for each file.
* `dataset` Provides functions for loading and processing mutation spectrum data.
* `plot`: Module to create plots for the meta signatures.
* `train`: Module to train the network.
