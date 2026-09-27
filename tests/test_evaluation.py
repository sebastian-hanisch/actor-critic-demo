"""Analyse und das Methoden-Vergleichs-Experiment: Aufbau und Grundeigenschaften (grosse Seed-Zahlen gehoeren in test_claims.py)."""

import numpy as np
import pytest

import ac_evaluation as E


def test_analyse_wiring_for_both_methods():
    for method in E.METHODS:
        a = E.analyse(E.Settings(episodes=50, seed=0), method=method)
        assert a.action_probs.shape == (a.grid.n_states, 4)
        assert np.allclose(a.action_probs.sum(axis=1), 1.0)
        assert a.returns.shape == (50,) and a.lengths.shape == (50,)
        assert a.env_steps == int(a.lengths.sum())


def test_analyse_is_deterministic_given_the_same_seed():
    s = E.Settings(episodes=30, seed=5)
    a1, a2 = E.analyse(s, method="actor_critic"), E.analyse(s, method="actor_critic")
    assert np.array_equal(a1.theta, a2.theta)


def test_reference_is_cached_across_settings_with_the_same_grid():
    assert E._reference(3, 4, 0.10, 0.95) is E._reference(3, 4, 0.10, 0.95)


def test_seed_variance_experiment_shape():
    exp = E.seed_variance_experiment(seeds=range(4), base=E.Settings(episodes=50))
    assert len(exp["rows"]) == 2
    methods = {r["method"] for r in exp["rows"]}
    assert methods == set(E.METHODS)
    for row in exp["rows"]:
        assert row["gaps"].shape == (4,)


def test_actor_critic_has_a_lower_stuck_fraction_than_reinforce_at_a_short_budget():
    # Kleine Stichprobe fuer Testgeschwindigkeit, grosszuegige Schwelle (das echte Verhaeltnis, siehe test_claims.py, ist viel deutlicher).
    exp = E.seed_variance_experiment(seeds=range(8), base=E.Settings(episodes=300))
    rows = {r["method"]: r for r in exp["rows"]}
    assert rows["actor_critic"]["frac_stuck"] <= rows["reinforce"]["frac_stuck"]
