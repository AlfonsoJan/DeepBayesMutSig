from functools import partial
import os
from pathlib import Path
import pickle
from typing import Literal, NamedTuple

from gabenet.mcmc import sample_markov_chain
from gabenet.nets import MultinomialDirichletBelieve, PoissonGammaBelieve
from gabenet.utils import freeze_trainable_states, perplexity
import haiku as hk
import jax
from jax import random
import jax.numpy as jnp

from dataset import load_mutation_spectrum, COSMIC_WEIGHTS

ARTEFACT_DIR = Path(__file__).parent.parent

# Training hyperparameters
# Pseudo-random number generator sequence.
RANDOM_SEED = 43
N_BURNIN = 200
LOG_EVERY = 200
N_SAMPLES = 2_000
CONTEXT = 96

# Print out training hyperparameters for logging.
print(f"CONTEXT = {CONTEXT}")
print(f"RANDOM_SEED = {RANDOM_SEED}")
print(f"N_BURNIN = {N_BURNIN}")
print(f"LOG_EVERY = {LOG_EVERY}")
print(f"N_SAMPLES = {N_SAMPLES}")

# Model hyperparameters.
MODEL: Literal[
    "multinomial_dirichlet_believe", "poisson_gamma_believe"
] = "multinomial_dirichlet_believe"
n_topics = len(COSMIC_WEIGHTS)
# Network config after pruning.
# Updated from 78 to 86 in 3.4
HIDDEN_LAYER_SIZES = [41, 86]
GAMMA_0 = 10.0
_bottom_layer_name = (
    f"{MODEL}/~/multinomial_layer"
    if MODEL == "multinomial_dirichlet_believe"
    else f"{MODEL}/~/poisson_layer"
)
# Print out model hyperparameters for logging.
print(f"MODEL = {MODEL}")
print(f"n_topics = {n_topics}")
print(f"HIDDEN_LAYER_SIZES = {HIDDEN_LAYER_SIZES}")
print(f"GAMMA_0 = {GAMMA_0}")

X_train, X_test, n_features = load_mutation_spectrum(context=CONTEXT)

print(X_train)

# i_checkpoint = 0

# class TrainState(NamedTuple):
#     params: hk.Params
#     state: hk.State
#     key: jax.Array  # type: ignore
#     step: int


# if jax.device_count == 1:
#     print("ERROR: Only one visible device!")
#     exit(1)


# def save_states(state: TrainState, samples):
#     """Extract and dump last state to disk."""
#     global i_checkpoint
#     architecture = "-".join(map(str, HIDDEN_LAYER_SIZES))
#     checkpoint_dir = ARTEFACT_DIR / MODEL / architecture / "checkpoints"
#     checkpoint_dir.mkdir(exist_ok=True, parents=True)
#     sample_dir = ARTEFACT_DIR / MODEL / architecture / "samples"
#     sample_dir.mkdir(exist_ok=True, parents=True)

#     print("Dumping samples to disk.")
#     with open(sample_dir / f"sample_{i_checkpoint:04d}.pkl", "wb") as fo:
#         pickle.dump(samples, fo)

#     print(f"Saving checkpoint i={i_checkpoint}.")
#     # Add leading zeros to checkpoint number.
#     name = f"checkpoint_{i_checkpoint:04d}.pkl"
#     with open(checkpoint_dir / name, "wb") as fo:
#         pickle.dump(state, fo)

#     i_checkpoint += 1

# def load_last_checkpoint() -> TrainState | None:
#     """Load last state from disk."""
#     global i_checkpoint
#     # List all pickle files, sort by number and load last one.
#     architecture = "-".join(map(str, HIDDEN_LAYER_SIZES))
#     checkpoint_dir = ARTEFACT_DIR / MODEL / architecture / "checkpoints"
#     files = sorted(checkpoint_dir.glob("checkpoint_*.pkl"))
#     if len(files) == 0:
#         print("No checkpoints found.")
#         return None
#     with open(files[-1], "rb") as fi:
#         state = pickle.load(fi)
#     i_checkpoint = int(files[-1].stem.split("_")[-1])
#     print(f"Loaded checkpoint i={i_checkpoint}.")
#     i_checkpoint += 1
#     return state


# @hk.transform_with_state
# def kernel(X=X_train, freeze_phi=True):
#     """Advance the Markov chain by one step."""
#     if MODEL == "multinomial_dirichlet_believe":
#         model = MultinomialDirichletBelieve(
#             HIDDEN_LAYER_SIZES, n_features, gamma_0=GAMMA_0
#         )
#     else:
#         model = PoissonGammaBelieve(HIDDEN_LAYER_SIZES, n_features, gamma_0=GAMMA_0)
#     if freeze_phi:
#         model.layers.layers[-1].set_training(False)
#     # Do one Gibbs sampling step.
#     model(X)


# def probability(params, state):
#     bottom_params = params.get(_bottom_layer_name, {})
#     bottom_state = state[_bottom_layer_name]
#     phi = bottom_params.get("phi", bottom_state.get("phi"))
#     theta = bottom_state["copy[theta(1)]"]
#     probs = theta @ phi
#     probs /= probs.sum(axis=-1, keepdims=True)
#     return probs


# def initialise(key) -> TrainState:
#     """Initialise training state."""
#     key, subkey = random.split(key)
#     keys = random.split(subkey, jax.device_count())
#     params, state = jax.vmap(partial(kernel.init, freeze_phi=False), in_axes=[0, None])(
#         keys, X_train
#     )
#     params, state = freeze_trainable_states(state, variable_names=["phi"])
#     params[_bottom_layer_name]["phi"] = jnp.array(COSMIC_WEIGHTS)
#     return TrainState(params, state, key, 0)


# def evaluate(params, states, X, axis=[0, 1]):
#     """Compute perplexity over chains and samples by default (axis=[0, 1])."""
#     probs = probability(params, states).mean(axis)
#     return perplexity(X, probs)


# def train_step(train_state: TrainState, n_burnin=0) -> TrainState:
#     """Do a set of Markov chain monte carlo steps and save checkpoint."""
#     key_seq = hk.PRNGSequence(train_state.key)
#     _, states = sample_markov_chain(
#         next(key_seq),
#         kernel=kernel,
#         n_samples=LOG_EVERY,
#         n_burnin_steps=n_burnin,
#         initial_state=train_state.state,
#         params=train_state.params,
#     )
#     loss_train = evaluate(train_state.params, states, X_train)
#     print(f"Training set perplexity: {loss_train:.2f}")
#     loss_test = evaluate(train_state.params, states, X_test)
#     print(f"Test set perplexity: {loss_test:.2f}")

#     last_state = jax.tree_util.tree_map(lambda x: x[:, -1, ...], states)
#     new_step = train_state.step + n_burnin + LOG_EVERY
#     new_train_state = TrainState(
#         train_state.params, last_state, next(key_seq), new_step
#     )
#     save_states(new_train_state, states)
#     return new_train_state


# n_checkpoint_iterations = N_SAMPLES // LOG_EVERY
# # When starting from scratch, initialize the Markov chain and run with burn in.
# if (train_state := load_last_checkpoint()) is None:
#     key = jax.random.PRNGKey(RANDOM_SEED)
#     train_state = initialise(key)
#     loss_test = evaluate(train_state.params, train_state.state, X_test, axis=[0])
#     print("Initial perplexity on test set", loss_test)
#     train_state = train_step(train_state, n_burnin=N_BURNIN)
#     n_checkpoint_iterations -= 1

# for _ in range(n_checkpoint_iterations):
#     train_state = train_step(train_state)
