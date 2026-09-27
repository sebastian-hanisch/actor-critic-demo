"""Der Kritiker: eine lineare Zustandswert-Schaetzung V(s) = w^T x(s). Mit One-Hot-Merkmalen ist w[s] direkt der geschaetzte Wert von Zustand s -
ein Update per semi-gradientem TD(0) ruehrt NUR den besuchten Zustand an, alle anderen bleiben unveraendert (kein Verallgemeinerungs-Anspruch,
wie schon beim Akteur)."""

import numpy as np


class LinearCritic:
    def __init__(self, n_features, seed=0):
        self.w = np.zeros(n_features)

    def value(self, x):
        return float(x @ self.w)

    def td_step(self, x, td_error, lr):
        self.w = self.w + lr * td_error * x
