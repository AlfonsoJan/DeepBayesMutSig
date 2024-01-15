import os, itertools
from pathlib import Path
from mubelnet.utils import holdout_split
from jax import random
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
# from mubelnet.utils import perplexity
from statkit.non_parametric import bootstrap_score

_PATH = Path(__file__).parent.parent/ "data" / "vcf"
_FILE_PATH = Path(__file__).parent.parent / "results" / "cluster.txt"
NMF_INIT = "nndsvdar"
BETA_LOSS = "kullback-leibler"
SIGNATURES = 48

def perplexity(X, probs):
    n_words = X.sum(axis=1, keepdims=True)
    log_probs = np.log(probs)
    log_probs = np.where(np.isneginf(log_probs) & (X == 0.0), 0.0, log_probs)
    log_likel = X * log_probs
    return np.exp(-np.nanmean(np.sum(log_likel / n_words, axis=1))) / 10

def calculate_bootstrap_score(X_train, X_test, signatures, nmf_init, beta_loss):
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
    pp_train = bootstrap_score(X_train_subset, probs_sigprof_xtrct_subset, metric=perplexity, random_state=43)
    results["Train perplexity"] = pp_train
    is_inf = (probs_sigprof_xtrct == 0) & (X_test > 0)
    X_test_eps  = np.where(is_inf, np.finfo(np.float64).tiny, X_test)
    pp_test = bootstrap_score(X_test_eps, probs_sigprof_xtrct, metric=perplexity, random_state=43)
    results["Test perplexity"] = pp_test
    zero_rows = np.where(is_inf)[0]
    print('Deleting rows', zero_rows)
    X_test_subset = np.delete(X_test, zero_rows, axis=0)
    probs_sigprof_xtrct_subset = np.delete(probs_sigprof_xtrct, zero_rows, axis=0)
    pp_test_sub = bootstrap_score(X_test_subset, probs_sigprof_xtrct_subset, metric=perplexity, random_state=43)
    results["Test subset perplexity"] = pp_test_sub
    return results


files = [
    "sbs.96.parquet",
    "sbs.1536.parquet",
    "sbs.24576.parquet",
    "sbs.24576.amino.parquet",
    "sbs.24576.nucl_strength.parquet",
    "sbs.24576.structure.parquet"
]


with open(_FILE_PATH, "w") as f:
    for file in files:
        print(file)
        sbs_file = _PATH.parent.parent / "SBS" / file
        df = pd.read_parquet(sbs_file)
        df = df[[df.columns[0]] + sorted(df.columns[1:])]
        all_genomes = np.array(df.iloc[:, 1:])
        X_train, X_test = holdout_split(random.PRNGKey(43), all_genomes)
        pp_dict = calculate_bootstrap_score(X_train, X_test, SIGNATURES, NMF_INIT, BETA_LOSS)
        for key, value in pp_dict.items():
            f.write(f"{file}\t{key}\t{value.latex()}\n")
        print(f"Done for {file=}")
        # break
