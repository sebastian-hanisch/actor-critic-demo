"""Analyse und das Kernexperiment dieses Stuecks: Actor-Critic gegen reines REINFORCE (Stück 7), auf denselben Seeds, demselben Vehikel und
denselben Trainingsepisoden - der einzige Unterschied ist, WOMIT aktualisiert wird (TD-Fehler des Kritikers vs. volle Monte-Carlo-Rückgabe)."""

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

import ac_agent as A
import ac_constants as C
import ac_grid as G
import ac_reference as R
from ac_features import one_hot

METHODS = ("actor_critic", "reinforce")
METHOD_LABEL = {"actor_critic": "Actor-Critic", "reinforce": "REINFORCE (ohne Basislinie)"}


@dataclass(frozen=True)
class Settings:
    rows: int = C.DEFAULT_ROWS
    cols: int = C.DEFAULT_COLS
    slip: float = C.DEFAULT_SLIP
    gamma: float = C.DEFAULT_GAMMA
    lr_actor: float = C.DEFAULT_ACTOR_LR
    lr_critic: float = C.DEFAULT_CRITIC_LR
    episodes: int = C.DEFAULT_EPISODES
    seed: int = 0

    @property
    def grid(self):
        return G.Grid(self.rows, self.cols, self.slip, self.gamma)


@lru_cache(maxsize=64)
def _reference(rows, cols, slip, gamma):
    grid = G.Grid(rows, cols, slip, gamma)
    P, Rw = G.build_model(grid)
    V_star, Q_star, pi_star = R.value_iteration(P, Rw, gamma)
    return grid, P, Rw, V_star, Q_star, pi_star


def _feature_fn(grid):
    return lambda s: one_hot(s, grid.n_states)


@dataclass
class Analysis:
    settings: Settings
    method: str
    grid: G.Grid
    theta: np.ndarray
    returns: np.ndarray
    lengths: np.ndarray
    falls: np.ndarray
    action_probs: np.ndarray
    V_star: np.ndarray
    pi_star: np.ndarray
    V_pi: np.ndarray
    gap: float
    env_steps: int
    snapshots: dict = field(default_factory=dict)


def analyse(s, method="actor_critic", checkpoints=()):
    grid, P, Rw, V_star, Q_star, pi_star = _reference(s.rows, s.cols, s.slip, s.gamma)
    feature_fn = _feature_fn(grid)
    if method == "actor_critic":
        policy, critic, returns, lengths, falls, snapshots = A.train_actor_critic(
            grid, feature_fn, grid.n_states, s.lr_actor, s.lr_critic, s.episodes, s.seed, checkpoints=checkpoints)
    else:
        policy, returns, lengths, falls, snapshots = A.train_reinforce(
            grid, feature_fn, grid.n_states, s.lr_actor, s.episodes, s.seed, checkpoints=checkpoints)
    action_probs = A.action_probs_table(policy, feature_fn, grid.n_states)
    V_pi = R.policy_evaluation_stochastic(P, Rw, action_probs, grid.gamma)
    start_s = grid.state_of(grid.start)
    gap = float(V_star[start_s] - V_pi[start_s])
    return Analysis(s, method, grid, policy.theta, returns, lengths, falls, action_probs, V_star, pi_star, V_pi, gap, int(lengths.sum()), snapshots)


def _gap_for_seed(rows, cols, slip, gamma, lr_actor, lr_critic, episodes, seed, method):
    grid, P, Rw, V_star, _, _ = _reference(rows, cols, slip, gamma)
    feature_fn = _feature_fn(grid)
    start_s = grid.state_of(grid.start)
    if method == "actor_critic":
        policy, critic, returns, lengths, falls, _ = A.train_actor_critic(grid, feature_fn, grid.n_states, lr_actor, lr_critic, episodes, seed)
    else:
        policy, returns, lengths, falls, _ = A.train_reinforce(grid, feature_fn, grid.n_states, lr_actor, episodes, seed)
    probs = A.action_probs_table(policy, feature_fn, grid.n_states)
    V_pi = R.policy_evaluation_stochastic(P, Rw, probs, gamma)
    return float(V_star[start_s] - V_pi[start_s]), float(returns[-20:].mean())


def seed_variance_experiment(seeds=None, base=None):
    """Fuer jede Methode (Actor-Critic, REINFORCE) ein eigenstaendiges Training je Seed (gleiche Hyperparameter, gleiche Seeds). Rueckgabe:
    Wert-Abstand je Seed und Methode, plus zusammenfassende Kennzahlen."""
    seeds = range(C.EXP_SEEDS) if seeds is None else list(seeds)
    base = Settings() if base is None else base
    rows = []
    for method in METHODS:
        gaps = []
        for seed in seeds:
            gap, _ = _gap_for_seed(base.rows, base.cols, base.slip, base.gamma, base.lr_actor, base.lr_critic, base.episodes, seed, method)
            gaps.append(gap)
        gaps = np.array(gaps)
        rows.append({
            "method": method,
            "gaps": gaps,
            "frac_near_optimal": float(np.mean(gaps < C.NEAR_OPTIMAL_GAP)),
            "frac_stuck": float(np.mean(gaps > C.STUCK_GAP_THRESHOLD)),
            "median_gap": float(np.median(gaps)),
        })
    return {"seeds": list(seeds), "rows": rows}
