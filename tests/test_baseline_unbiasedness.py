"""Der zentrale Korrektheits-Check dieses Stuecks (PLAN.md Hook 8): eine von der Aktion UNABHAENGIGE Basislinie b(s) aendert den ERWARTUNGSWERT
des Politikgradienten NICHT, egal wie sie gewaehlt ist (Lehrbuch-Identitaet E[b(s)*grad_log_pi(a|s)] = 0, weil E_a[grad_log_pi(a|s)] = 0 - der
Gradient einer normierten Wahrscheinlichkeitsverteilung summiert sich zu null). Exakt nachgerechnet durch Aufsummieren ueber alle Aktionen einer
winzigen Politik (keine Stichprobe, kein Zufall) - nicht nur "nahe genug" statistisch, sondern bis auf Gleitkomma-Genauigkeit exakt."""

import numpy as np
import pytest

from ac_policy import SoftmaxPolicy


def _expected_gradient(policy, x, rewards, baseline):
    """E_{a~pi(.|x)}[(r(a) - baseline) * grad_log_pi(a|x)] - exakt durch Aufsummieren ueber alle Aktionen, kein Sampling."""
    probs = policy.probs(x[None, :])[0]
    total = np.zeros_like(policy.theta)
    for a, (p_a, r_a) in enumerate(zip(probs, rewards)):
        total += p_a * (r_a - baseline) * policy.grad_log_prob(x, a)
    return total


@pytest.mark.parametrize("baseline", [0.0, 5.0, -3.7, 100.0])
def test_any_action_independent_baseline_leaves_the_expected_gradient_unchanged(baseline):
    policy = SoftmaxPolicy(n_features=2, n_actions=4, seed=3)
    x = np.array([0.4, -0.9])
    rewards = np.array([10.0, -100.0, -1.0, -1.0])
    grad_no_baseline = _expected_gradient(policy, x, rewards, baseline=0.0)
    grad_with_baseline = _expected_gradient(policy, x, rewards, baseline=baseline)
    assert np.allclose(grad_no_baseline, grad_with_baseline, atol=1e-10)


def test_the_mean_reward_itself_is_a_valid_unbiased_baseline():
    # Insbesondere die vom Kritiker geschaetzte Politik-Rueckgabe selbst (V(s) ~ E[r]) ist ein gueltiger Spezialfall.
    policy = SoftmaxPolicy(n_features=2, n_actions=4, seed=7)
    x = np.array([-0.2, 0.5])
    rewards = np.array([3.0, -50.0, 2.0, 0.0])
    probs = policy.probs(x[None, :])[0]
    mean_reward = float(np.dot(probs, rewards))
    grad_no_baseline = _expected_gradient(policy, x, rewards, baseline=0.0)
    grad_with_mean_baseline = _expected_gradient(policy, x, rewards, baseline=mean_reward)
    assert np.allclose(grad_no_baseline, grad_with_mean_baseline, atol=1e-10)


def test_expected_grad_log_prob_alone_is_zero():
    # Der eigentliche Grund fuer die beiden Tests oben: der Gradient einer normierten Verteilung summiert sich im Erwartungswert zu null.
    policy = SoftmaxPolicy(n_features=3, n_actions=5, seed=1)
    x = np.array([0.3, -0.1, 0.7])
    probs = policy.probs(x[None, :])[0]
    total = np.zeros_like(policy.theta)
    for a, p_a in enumerate(probs):
        total += p_a * policy.grad_log_prob(x, a)
    assert np.allclose(total, 0.0, atol=1e-10)
