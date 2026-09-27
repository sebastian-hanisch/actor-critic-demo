"""Der Kritiker von Hand: Werteschätzung, TD(0)-Update rührt nur den besuchten Zustand an, und Konvergenz auf einer Kette mit geschlossener
Lösung (unabhängig von der Politik - reine Politik-Auswertung mit Bootstrapping)."""

import numpy as np
import pytest

from ac_critic import LinearCritic


def test_value_is_the_dot_product():
    c = LinearCritic(n_features=3)
    c.w = np.array([1.0, 2.0, 3.0])
    assert c.value(np.array([0.0, 1.0, 0.0])) == pytest.approx(2.0)


def test_td_step_only_touches_the_visited_state():
    c = LinearCritic(n_features=4)
    x = np.array([0.0, 1.0, 0.0, 0.0])
    c.td_step(x, td_error=2.0, lr=0.5)
    assert np.allclose(c.w, [0.0, 1.0, 0.0, 0.0])


def test_td_zero_converges_to_the_exact_value_on_a_two_state_chain():
    # s0 -(r=1)-> s1 (terminal, V(s1)=0), deterministisch, gamma=0.9. Geschlossene Loesung: V(s0) = 1.0 (ein Schritt, dann Ende).
    x0, x1 = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    c = LinearCritic(n_features=2)
    gamma = 0.9
    for _ in range(2000):
        td_error = 1.0 + gamma * 0.0 - c.value(x0)
        c.td_step(x0, td_error, lr=0.1)
    assert c.value(x0) == pytest.approx(1.0, abs=1e-3)
