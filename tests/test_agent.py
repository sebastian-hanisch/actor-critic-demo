"""REINFORCE (Vergleichsmassstab) und Actor-Critic: Rückgabe-Berechnung von Hand, Training läuft und aktualisiert beide Parametersätze,
Schnappschuss-Tests für beide Trainingsschleifen."""

import numpy as np
import pytest

import ac_agent as A
import ac_constants as C
from ac_features import one_hot
from ac_grid import Grid


def test_returns_from_rewards_matches_the_discounted_sum_by_hand():
    rewards = [1.0, 2.0, 3.0]
    gamma = 0.5
    G = A.returns_from_rewards(rewards, gamma)
    assert G[2] == pytest.approx(3.0)
    assert G[1] == pytest.approx(2.0 + 0.5 * 3.0)
    assert G[0] == pytest.approx(1.0 + 0.5 * (2.0 + 0.5 * 3.0))


def test_train_reinforce_runs_and_updates_theta():
    g = Grid(rows=3, cols=4, slip=0.0, gamma=0.9)
    feature_fn = lambda s: one_hot(s, g.n_states)
    policy, returns, lengths, falls, snaps = A.train_reinforce(g, feature_fn, g.n_states, lr=0.2, episodes=20, seed=0)
    assert returns.shape == (20,) and lengths.shape == (20,)
    assert not np.allclose(policy.theta, 0.0)


def test_train_reinforce_snapshot_at_a_checkpoint_equals_a_separately_truncated_run():
    g = Grid(rows=3, cols=4, slip=0.0, gamma=0.9)
    feature_fn = lambda s: one_hot(s, g.n_states)
    _, _, _, _, snaps = A.train_reinforce(g, feature_fn, g.n_states, lr=0.2, episodes=15, seed=3, checkpoints=(7,))
    truncated, *_ = A.train_reinforce(g, feature_fn, g.n_states, lr=0.2, episodes=7, seed=3)
    assert np.allclose(snaps[7].theta, truncated.theta)


def test_train_actor_critic_runs_and_updates_both_actor_and_critic():
    g = Grid(rows=3, cols=4, slip=0.0, gamma=0.9)
    feature_fn = lambda s: one_hot(s, g.n_states)
    policy, critic, returns, lengths, falls, snaps = A.train_actor_critic(
        g, feature_fn, g.n_states, lr_actor=0.05, lr_critic=0.1, episodes=20, seed=0)
    assert returns.shape == (20,) and lengths.shape == (20,)
    assert not np.allclose(policy.theta, 0.0)
    assert not np.allclose(critic.w, 0.0)


def test_train_actor_critic_snapshot_at_a_checkpoint_equals_a_separately_truncated_run():
    g = Grid(rows=3, cols=4, slip=0.0, gamma=0.9)
    feature_fn = lambda s: one_hot(s, g.n_states)
    _, _, _, _, _, snaps = A.train_actor_critic(g, feature_fn, g.n_states, lr_actor=0.05, lr_critic=0.1, episodes=15, seed=3, checkpoints=(7,))
    truncated, *_ = A.train_actor_critic(g, feature_fn, g.n_states, lr_actor=0.05, lr_critic=0.1, episodes=7, seed=3)
    assert np.allclose(snaps[7].theta, truncated.theta)


def test_action_probs_table_covers_every_state_and_sums_to_one():
    g = Grid(rows=3, cols=4, slip=0.1, gamma=0.9)
    feature_fn = lambda s: one_hot(s, g.n_states)
    from ac_policy import SoftmaxPolicy
    policy = SoftmaxPolicy(g.n_states, 4, seed=0)
    table = A.action_probs_table(policy, feature_fn, g.n_states)
    assert table.shape == (g.n_states, 4)
    assert np.allclose(table.sum(axis=1), 1.0)
