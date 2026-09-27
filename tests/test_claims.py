"""Jede Zahl im README gegen den tatsaechlichen Code, nicht nur behauptet. Der Methoden-Vergleich ist der Befund dieses Stuecks und wird deshalb
mit der VOLLEN Seed-Zahl nachgerechnet (wie im README), nicht mit einer kleineren, schnelleren Stichprobe."""

import pytest

import ac_constants as C
import ac_evaluation as E


def test_seed5_reinforce_is_stuck_but_actor_critic_recovers():
    # Derselbe Seed, der REINFORCE (policy-gradient-demo, Stück 7) bei identischen Hyperparametern dauerhaft in einer schlechten Politik
    # haengen liess (Wert-Abstand 114,4) - hier zusaetzlich mit Actor-Critic auf demselben Seed gegengeprueft.
    re_gap, _ = E._gap_for_seed(C.DEFAULT_ROWS, C.DEFAULT_COLS, C.DEFAULT_SLIP, C.DEFAULT_GAMMA, C.DEFAULT_ACTOR_LR, C.DEFAULT_CRITIC_LR,
                                 C.DEFAULT_EPISODES, seed=5, method="reinforce")
    ac_gap, _ = E._gap_for_seed(C.DEFAULT_ROWS, C.DEFAULT_COLS, C.DEFAULT_SLIP, C.DEFAULT_GAMMA, C.DEFAULT_ACTOR_LR, C.DEFAULT_CRITIC_LR,
                                 C.DEFAULT_EPISODES, seed=5, method="actor_critic")
    assert re_gap == pytest.approx(114.4, abs=5.0)
    assert re_gap > C.STUCK_GAP_THRESHOLD
    assert ac_gap < C.NEAR_OPTIMAL_GAP


def test_method_comparison_matches_the_readme():
    # Volle Seed-Zahl wie im README - deterministisch bei festen Seeds, aber ueber 2000 Trainings-Episoden koennen winzige Gleitkomma-
    # Unterschiede zwischen Plattformen einen knapp an einer Schwelle liegenden Seed kippen lassen (siehe policy-gradient-demo, Stück 7) -
    # deshalb grosszuegige statt exakte Baender (feedback_ci_platform_robust_tests). Laufzeit: mehrere Minuten.
    exp = E.seed_variance_experiment(seeds=range(C.EXP_SEEDS_FULL))
    rows = {r["method"]: r for r in exp["rows"]}
    ac_row, re_row = rows["actor_critic"], rows["reinforce"]
    assert ac_row["frac_near_optimal"] == pytest.approx(0.967, abs=0.07)
    assert ac_row["frac_stuck"] == pytest.approx(0.0, abs=0.07)
    assert ac_row["median_gap"] == pytest.approx(0.73, abs=0.5)
    assert re_row["frac_near_optimal"] == pytest.approx(0.1667, abs=0.07)
    assert re_row["frac_stuck"] == pytest.approx(0.1667, abs=0.07)
    # Der eigentliche Befund: Actor-Critic bleibt in DEUTLICH weniger Faellen haengen als reines REINFORCE, bei denselben Seeds.
    assert ac_row["frac_stuck"] < re_row["frac_stuck"]
