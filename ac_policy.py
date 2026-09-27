"""Der Akteur: eine lineare Softmax-Politik pi(a|s) = softmax(theta^T x(s))_a - byte-gleich zu `policy-gradient-demo`s Politik (Stück 7). Der
Unterschied dieses Stuecks liegt NICHT im Akteur, sondern darin, WOMIT er trainiert wird (siehe `ac_critic.py`, `ac_agent.py`)."""

import numpy as np


class SoftmaxPolicy:
    def __init__(self, n_features, n_actions, seed=0):
        rng = np.random.default_rng(seed)
        self.theta = rng.normal(0.0, 0.01, size=(n_features, n_actions))

    def probs(self, X):
        scores = X @ self.theta
        scores = scores - scores.max(axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        return exp_scores / exp_scores.sum(axis=1, keepdims=True)

    def sample(self, x, rng):
        p = self.probs(x[None, :])[0]
        return int(rng.choice(len(p), p=p))

    def grad_log_prob(self, x, action):
        p = self.probs(x[None, :])[0]
        n_actions = p.shape[0]
        onehot_a = np.zeros(n_actions)
        onehot_a[action] = 1.0
        return np.outer(x, onehot_a - p)

    def step(self, grad, lr):
        self.theta = self.theta + lr * grad
