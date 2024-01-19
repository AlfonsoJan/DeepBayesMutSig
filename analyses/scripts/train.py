from functools import partial
import os
from pathlib import Path
import pickle
from typing import Literal, NamedTuple

from mubelnet.nets import MultinomialBelief
from mubelnet.utils import perplexity
import haiku as hk
import jax
from jax import random
import jax.numpy as jnp

from dataset import load_mutation_spectrum, COSMIC_WEIGHTS

_ARTEFACT_DIR = Path(os.environ.get("ARTEFACT_DIR", "/mnt/output/"))
_ARTEFACT_DIR.mkdir(parents=True, exist_ok=True)

# Training hyperparameters
# Pseudo-random number generator sequence.
RANDOM_SEED = 43
N_BURNIN = 200
LOG_EVERY = 200
N_SAMPLES = 2_000
CONTEXT = 96

# Print out training hyperparameters for logging.
print(f"ARTEFACT_DIR = {_ARTEFACT_DIR}")
print(f"CONTEXT = {CONTEXT}")
print(f"RANDOM_SEED = {RANDOM_SEED}")
print(f"N_BURNIN = {N_BURNIN}")
print(f"LOG_EVERY = {LOG_EVERY}")
print(f"N_SAMPLES = {N_SAMPLES}")

# Model hyperparameters.
MODEL: Literal["multinomial_belief", "poisson_gamma_believe"] = "multinomial_belief"
MODEL = "multinomial_belief"
n_topics = len(COSMIC_WEIGHTS)
n_chains = jax.device_count()
# Network config after pruning.
# Updated from 78 to 86 in 3.4
HIDDEN_LAYER_SIZES = [41, n_topics]
GAMMA_0 = 10.0
_bottom_layer_name = f"{MODEL}/~/multinomial_layer"
# Print out model hyperparameters for logging.
print(f"MODEL = {MODEL}")
print(f"n_chains = {n_chains}")
print(f"n_topics = {n_topics}")
print(f"HIDDEN_LAYER_SIZES = {HIDDEN_LAYER_SIZES}")
print(f"GAMMA_0 = {GAMMA_0}")

X_train, X_test, n_features = load_mutation_spectrum(
    random_state=RANDOM_SEED, context=CONTEXT
)


i_checkpoint = 0


class TrainState(NamedTuple):
    params: hk.Params
    state: hk.State
    key: jax.Array  # type: ignore
    step: int
    model_name: Literal["multinomial_belief", "poisson_gamma_belief"]
    hidden_layer_sizes: tuple[int]


if jax.device_count == 1:
    print("ERROR: Only one visible device!")
    exit(1)


def infer_last_checkpoint_number(checkpoint_dir: Path) -> int:
    """Look in checkpoint_dir and find largest checkpoint number."""
    # List all pickle files, sort by number and load last one.
    files = sorted(checkpoint_dir.glob("checkpoint_*.pkl"))
    if len(files) == 0:
        return -1
    return int(files[-1].stem.split("_")[-1])


def save_states(state: TrainState, samples, target_dir=_ARTEFACT_DIR):
    """Extract and dump last state to disk."""
    architecture = "-".join(map(str, state.hidden_layer_sizes))
    checkpoint_dir = target_dir / state.model_name / architecture / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    sample_dir = target_dir / state.model_name / architecture / "samples"
    sample_dir.mkdir(exist_ok=True, parents=True)

    i_checkpoint = infer_last_checkpoint_number(checkpoint_dir) + 1

    print("Dumping samples to disk.")
    with open(sample_dir / f"sample_{i_checkpoint:04d}.pkl", "wb") as fo:
        pickle.dump(samples, fo)

    print(f"Saving checkpoint i={i_checkpoint}.")
    # Add leading zeros to checkpoint number.
    name = f"checkpoint_{i_checkpoint:04d}.pkl"
    with open(checkpoint_dir / name, "wb") as fo:
        pickle.dump(state, fo)

    i_checkpoint += 1


def load_last_checkpoint(
    model_name, hidden_layer_sizes, source_dir=_ARTEFACT_DIR
) -> TrainState | None:
    """Load last state from disk."""
    architecture = "-".join(map(str, hidden_layer_sizes))
    checkpoint_dir = source_dir / model_name / architecture / "checkpoints"
    files = sorted(checkpoint_dir.glob("checkpoint_*.pkl"))
    if len(files) == 0:
        print("No checkpoints found.")
        return None
    with open(files[-1], "rb") as fi:
        state = pickle.load(fi)
    i_checkpoint = int(files[-1].stem.split("_")[-1])
    print(f"Loaded checkpoint i={i_checkpoint}.")
    return state


@hk.transform_with_state
def kernel(n_hidden_units=HIDDEN_LAYER_SIZES, X=X_train):
    """Advance the Markov chain by one step."""
    model = MultinomialBelief(n_hidden_units, n_features, gamma_0=GAMMA_0)
    # Do one Gibbs sampling step.
    model(X)


def probability(params, state):
    bottom_params = params.get(_bottom_layer_name, {})
    bottom_state = state[_bottom_layer_name]
    phi = bottom_params.get("phi", bottom_state.get("phi"))
    theta = bottom_state["copy[theta(1)]"]
    probs = theta @ phi
    probs /= probs.sum(axis=-1, keepdims=True)
    return probs


def initialise(key, model, hidden_layer_sizes) -> TrainState:
    """Initialise training state."""
    key, subkey = random.split(key)
    keys = random.split(subkey, jax.device_count())
    params, state = jax.pmap(kernel.init)(keys)
    return TrainState(
        params,
        state,
        key,
        step=0,
        model_name=model,
        hidden_layer_sizes=hidden_layer_sizes,
    )


def evaluate(params, states, X, axis=[0, 1]):
    """Compute perplexity over chains and samples by default (axis=[0, 1])."""
    probs = probability(params, states).mean(axis)
    return perplexity(X, probs)


def train_step(kernel_fn, train_state: TrainState, n_steps) -> TrainState:
    """Do a set of Markov chain monte carlo steps and save checkpoint."""
    key_seq = hk.PRNGSequence(train_state.key)
    state = train_state.state
    # 1) Burn-in model.
    for i in range(n_steps):
        _, state = kernel_fn(
            train_state.params, state, random.split(next(key_seq), n_chains)
        )
        print(".", end="")

    train_state = TrainState(
        train_state.params,
        state,
        next(key_seq),
        train_state.step + n_steps,
        MODEL,
        train_state.hidden_layer_sizes,
    )
    return train_state


# When starting from scratch, initialize the Markov chain and run with burn in.
if (train_state := load_last_checkpoint(MODEL, HIDDEN_LAYER_SIZES)) is None:
    key = jax.random.PRNGKey(RANDOM_SEED)
    train_state = initialise(key, MODEL, HIDDEN_LAYER_SIZES)
    loss_test = evaluate(train_state.params, train_state.state, X_test, axis=[0])
    print("Initial perplexity on test set", loss_test)

n_start = train_state.step // LOG_EVERY
n_stop = N_BURNIN // LOG_EVERY
kernel_fn = jax.pmap(
    partial(kernel.apply, n_hidden_units=train_state.hidden_layer_sizes),
    in_axes=(None, 0, 0),
)

for _ in range(n_start, n_stop):
    train_state = train_step(kernel_fn, train_state, n_steps=LOG_EVERY)
    save_states(train_state, {})
    print("Burn in", train_state.step)

for _ in range(10):
    trace = []
    for i in range(LOG_EVERY):
        train_state = train_step(kernel_fn, train_state, n_steps=1)
        trace.append(train_state.state)

    states = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=1), *trace)

    save_states(train_state, states)
