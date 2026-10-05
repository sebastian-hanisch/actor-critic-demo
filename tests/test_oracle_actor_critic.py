"""Orakel-Tests mit anderen Rechenwegen: Übergänge von `step` gegen `build_model` per Aufzählung der drei Zufallsfälle, Value Iteration gegen Politik-Iteration mit exakten linearen Lösungen
(und ein lineares Programm, falls scipy da ist), beide Trainingsschleifen gegen eine tabellarische Neuschreibung mit demselben Zufallsstrom, Politikgradienten-Satz (erwartetes Update = Gradient
des exakten Werts per finiter Differenz)."""

import numpy as np
import pytest

import ac_agent as A
import ac_constants as C
import ac_grid as G
import ac_reference as R
from ac_features import one_hot
from ac_policy import SoftmaxPolicy


def _softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


def _solve_policy(P, Rw, probs, gamma):
    S = P.shape[0]
    return np.linalg.solve(np.eye(S) - gamma * np.einsum("sa,sap->sp", probs, P), np.einsum("sa,sa->s", probs, Rw))


class _FixedU:
    def __init__(self, u):
        self.u = u

    def random(self):
        return self.u


def test_step_matches_the_model_by_enumerating_the_three_random_cases():
    rng = np.random.default_rng(0)
    for _ in range(15):
        g = G.Grid(int(rng.integers(2, 5)), int(rng.integers(3, 7)), float(rng.choice([0.0, 0.1, 0.3])), 0.9)
        P, Rw = G.build_model(g)
        for s in range(g.n_states):
            for a in G.ACTIONS:
                Pe, Re = np.zeros(g.n_states), 0.0
                for u, w in ((0.0, 1 - g.slip), (1 - 0.75 * g.slip, g.slip / 2), (0.999999999, g.slip / 2)):
                    if w > 0:
                        s2, r, _ = G.step(g, s, a, _FixedU(u))
                        Pe[s2] += w
                        Re += w * r
                assert np.allclose(Pe, P[s, a]) and abs(Re - Rw[s, a]) < 1e-12


def test_value_iteration_and_policy_evaluation_match_exact_linear_solves():
    rng = np.random.default_rng(1)
    for _ in range(15):
        g = G.Grid(int(rng.integers(2, 5)), int(rng.integers(3, 7)), float(rng.choice([0.0, 0.1, 0.2])), float(rng.choice([0.8, 0.95, 0.99])))
        P, Rw = G.build_model(g)
        S = g.n_states
        V, Q, pol = R.value_iteration(P, Rw, g.gamma)
        pi = np.zeros(S, int)
        for _ in range(100):                                                                 # Politik-Iteration mit exakten Lösungen
            Vp = _solve_policy(P, Rw, np.eye(4)[pi], g.gamma)
            Qp = Rw + g.gamma * np.einsum("sap,p->sa", P, Vp)
            new = np.where(Qp.max(1) > Qp[np.arange(S), pi] + 1e-12, Qp.argmax(1), pi)
            if (new == pi).all():
                break
            pi = new
        assert np.abs(Vp - V).max() < 1e-5
        assert np.abs(_solve_policy(P, Rw, np.eye(4)[pol], g.gamma) - V).max() < 1e-5          # die gierige Politik ist optimal
        probs = rng.dirichlet(np.ones(4), S)
        assert np.abs(R.policy_evaluation_stochastic(P, Rw, probs, g.gamma) - _solve_policy(P, Rw, probs, g.gamma)).max() < 1e-5
        det = rng.integers(0, 4, S)
        assert np.abs(R.policy_evaluation(P, Rw, det, g.gamma) - _solve_policy(P, Rw, np.eye(4)[det], g.gamma)).max() < 1e-5


def test_value_iteration_matches_the_linear_program_of_the_mdp():
    opt = pytest.importorskip("scipy.optimize")
    g = G.Grid(3, 5, 0.1, 0.95)
    P, Rw = G.build_model(g)
    S = g.n_states
    rows, rhs = [], []
    for s in range(S):
        for a in range(4):
            row = g.gamma * P[s, a].copy()
            row[s] -= 1.0
            rows.append(row)
            rhs.append(-Rw[s, a])
    lp = opt.linprog(np.ones(S), A_ub=np.array(rows), b_ub=np.array(rhs), bounds=[(None, None)] * S, method="highs")           # min sum V  s.t.  V >= R + gamma P V
    assert np.abs(lp.x - R.value_iteration(P, Rw, g.gamma)[0]).max() < 1e-5


def _tabular_reinforce(g, lr, episodes, seed):
    rng = np.random.default_rng(seed)
    theta = np.random.default_rng(seed).normal(0.0, 0.01, size=(g.n_states, 4))
    for _ in range(episodes):
        s, traj = g.state_of(g.start), []
        for _ in range(C.MAX_STEPS_PER_EPISODE):
            a = int(rng.choice(4, p=_softmax(theta[s])))
            s2, r, done = G.step(g, s, a, rng)
            traj.append((s, a, r))
            s = s2
            if done:
                break
        for t, (st, a, _) in enumerate(traj):
            Gt = sum(g.gamma ** k * traj[t + k][2] for k in range(len(traj) - t))
            grad = -_softmax(theta[st])
            grad[a] += 1.0
            theta[st] += lr * g.gamma ** t * Gt * grad
    return theta


def _tabular_actor_critic(g, lr_a, lr_c, episodes, seed):
    rng = np.random.default_rng(seed)
    theta = np.random.default_rng(seed).normal(0.0, 0.01, size=(g.n_states, 4))
    V = np.zeros(g.n_states)
    for _ in range(episodes):
        s, discount = g.state_of(g.start), 1.0
        for _ in range(C.MAX_STEPS_PER_EPISODE):
            p = _softmax(theta[s])
            a = int(rng.choice(4, p=p))
            s2, r, done = G.step(g, s, a, rng)
            delta = r + (0.0 if done else g.gamma * V[s2]) - V[s]
            V[s] += lr_c * delta
            grad = -p
            grad[a] += 1.0
            theta[s] += lr_a * discount * delta * grad
            discount *= g.gamma
            s = s2
            if done:
                break
    return theta, V


def test_both_training_loops_equal_a_tabular_rewrite_with_the_same_random_stream():
    rng = np.random.default_rng(2)
    for _ in range(4):
        g = G.Grid(int(rng.integers(3, 5)), int(rng.integers(4, 7)), float(rng.choice([0.0, 0.1, 0.2])), float(rng.choice([0.9, 0.95])))
        seed = int(rng.integers(0, 1000))
        feat = lambda s: one_hot(s, g.n_states)
        pol, *_ = A.train_reinforce(g, feat, g.n_states, 0.005, 40, seed)
        assert np.allclose(pol.theta, _tabular_reinforce(g, 0.005, 40, seed), atol=1e-8)
        pol, critic, *_ = A.train_actor_critic(g, feat, g.n_states, 0.01, 0.1, 40, seed)
        theta, V = _tabular_actor_critic(g, 0.01, 0.1, 40, seed)
        assert np.allclose(pol.theta, theta, atol=1e-8) and np.allclose(critic.w, V, atol=1e-8)


def test_expected_actor_update_with_the_exact_advantage_is_the_gradient_of_the_exact_value():
    """Politikgradienten-Satz: sum_s d(s) sum_a pi (Q - V) grad log pi (d = diskontierte Besuchsverteilung) = Gradient von V^pi(Start) per finiter Differenz; prüft Vorzeichen und Lage von grad_log_prob."""
    rng = np.random.default_rng(3)
    for _ in range(4):
        g = G.Grid(int(rng.integers(2, 4)), int(rng.integers(3, 5)), float(rng.choice([0.0, 0.1, 0.2])), float(rng.choice([0.8, 0.95])))
        P, Rw = G.build_model(g)
        S, s0 = g.n_states, g.state_of(g.start)
        theta = rng.normal(0, 0.8, (S, 4))
        pol = SoftmaxPolicy(S, 4)
        pol.theta = theta.copy()

        def J(th):
            return _solve_policy(P, Rw, np.array([_softmax(r) for r in th]), g.gamma)[s0]

        fd = np.zeros_like(theta)
        for s in range(S):
            for a in range(4):
                up, dn = theta.copy(), theta.copy()
                up[s, a] += 1e-6
                dn[s, a] -= 1e-6
                fd[s, a] = (J(up) - J(dn)) / 2e-6
        probs = np.array([_softmax(r) for r in theta])
        V = _solve_policy(P, Rw, probs, g.gamma)
        Q = Rw + g.gamma * np.einsum("sap,p->sa", P, V)
        d = np.linalg.solve((np.eye(S) - g.gamma * np.einsum("sa,sap->sp", probs, P)).T, np.eye(S)[s0])
        grad = sum(d[s] * probs[s, a] * (Q[s, a] - V[s]) * pol.grad_log_prob(one_hot(s, S), a) for s in range(S) for a in range(4))
        assert np.abs(grad - fd).max() < 1e-5
