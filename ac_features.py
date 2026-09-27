"""Zustands-Darstellung: One-Hot, fuer Akteur UND Kritiker (jeder Zustand hat seine eigene, isolierte Spalte)."""

import numpy as np


def one_hot(state, n_states):
    x = np.zeros(n_states)
    x[state] = 1.0
    return x
