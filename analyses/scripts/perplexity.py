from pathlib import Path
from mubelnet.utils import holdout_split
from jax import random
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from mubelnet.utils import perplexity
from statkit.non_parametric import bootstrap_score

_PATH = Path(__file__).parent
signatures = 48
beta_loss = "frobenius"

print("RUNNING PERPLEXITY ANALYSIS")

def calculate_bootstrap_score(all_genomes, signatures, nmf_init, beta_loss):
    X_train, X_test = holdout_split(random.PRNGKey(42), all_genomes)
    nmf = NMF(
        n_components=signatures,
        init=nmf_init,
        beta_loss=beta_loss,
        solver="cd",
        max_iter=10_000,
        tol=1e-15,
    ).fit(X_train)
    h = nmf.transform(X_train)
    unnormed_probs = h @ nmf.components_
    probs_nmf = unnormed_probs / unnormed_probs.sum(axis=-1, keepdims=True)
    # Compute performance on test set.
    is_inf = (probs_nmf == 0) & (X_test > 0)

    # 2) Delete zero rows.
    zero_rows = np.where(is_inf)[0]
    print('Deleting rows', zero_rows)
    X_test_subset = np.delete(X_test, zero_rows, axis=0)
    probs_sigprof_xtrct_subset = np.delete(probs_nmf, zero_rows, axis=0)

    pp_nmf = bootstrap_score(X_test_subset, probs_sigprof_xtrct_subset, metric=perplexity, random_state=43)
    print('Perplexity on unseen observations (removed samples causing infinite perplexity)', pp_nmf)
    print(pp_nmf.latex())
    return pp_nmf

def calculate_perplexity(context: list):
    for c in context:
        print(f"Doing {c} context")
        df = pd.read_parquet(_PATH / f"sbs.{c}.parquet")
        all_genomes = np.array(df.iloc[:, 1:])
        pp_nmf = calculate_bootstrap_score(all_genomes, signatures, None, beta_loss)
        print(pp_nmf)
        print(f"Finished {c} context")


calculate_perplexity(context=["96", "1536", "24576"])
