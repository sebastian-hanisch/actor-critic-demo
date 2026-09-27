"""Die Softmax-Politik des Akteurs von Hand: Wahrscheinlichkeiten, Sampling, analytischer Gradient gegen einen numerischen Gradienten-Check."""

import numpy as np
import pytest

from ac_policy import SoftmaxPolicy


def test_probs_sum_to_one_and_are_positive():
    p = SoftmaxPolicy(n_features=3, n_actions=4, seed=0)
    probs = p.probs(np.eye(3))
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert (probs > 0).all()


def test_grad_log_prob_matches_a_numerical_gradient_check():
    p = SoftmaxPolicy(n_features=2, n_actions=3, seed=1)
    x = np.array([0.6, -0.3])
    action = 1
    analytic = p.grad_log_prob(x, action)
    eps = 1e-6
    numeric = np.zeros_like(p.theta)
    for i in range(p.theta.shape[0]):
        for j in range(p.theta.shape[1]):
            orig = p.theta[i, j]
            p.theta[i, j] = orig + eps
            logp_plus = np.log(p.probs(x[None, :])[0, action])
            p.theta[i, j] = orig - eps
            logp_minus = np.log(p.probs(x[None, :])[0, action])
            p.theta[i, j] = orig
            numeric[i, j] = (logp_plus - logp_minus) / (2 * eps)
    assert np.allclose(analytic, numeric, atol=1e-5)


def test_step_moves_theta_in_the_gradient_direction():
    p = SoftmaxPolicy(2, 2, seed=0)
    theta0 = p.theta.copy()
    grad = np.ones_like(p.theta)
    p.step(grad, lr=0.1)
    assert np.allclose(p.theta, theta0 + 0.1 * grad)
