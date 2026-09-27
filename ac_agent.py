"""Zwei Trainingsschleifen auf demselben Vehikel: `train_reinforce` (episodisches Monte-Carlo-REINFORCE, byte-gleich zu `policy-gradient-demo`s
Agent - der Vergleichsmassstab) und `train_actor_critic` (Sutton & Barto 2018, Algorithmus 13.5: One-step Actor-Critic). Der Unterschied ist NICHT
der Akteur (identische Softmax-Politik), sondern WOMIT er aktualisiert wird: REINFORCE wartet auf die volle Episoden-Rueckgabe, Actor-Critic
nutzt bei JEDEM Schritt den TD-Fehler eines gelernten Kritikers als Schaetzung des Vorteils - kein Warten auf das Episodenende noetig."""

import numpy as np

import ac_constants as C
from ac_critic import LinearCritic
from ac_grid import ACTIONS, step
from ac_policy import SoftmaxPolicy

N_ACTIONS = len(ACTIONS)


def returns_from_rewards(rewards, gamma):
    T = len(rewards)
    G = np.zeros(T)
    running = 0.0
    for t in reversed(range(T)):
        running = rewards[t] + gamma * running
        G[t] = running
    return G


def _run_episode_reinforce(grid, policy, feature_fn, rng, max_steps):
    s = grid.state_of(grid.start)
    states, actions, rewards = [], [], []
    for _ in range(max_steps):
        x = feature_fn(s)
        a = policy.sample(x, rng)
        s_next, r, done = step(grid, s, a, rng)
        states.append(s)
        actions.append(a)
        rewards.append(r)
        s = s_next
        if done:
            break
    return states, actions, rewards


def train_reinforce(grid, feature_fn, n_features, lr, episodes, seed, max_steps=C.MAX_STEPS_PER_EPISODE, checkpoints=()):
    """Episodisches Monte-Carlo-REINFORCE ohne Basislinie - der Vergleichsmassstab (byte-gleich zu policy-gradient-demo, Stück 7)."""
    rng = np.random.default_rng(seed)
    policy = SoftmaxPolicy(n_features, N_ACTIONS, seed)
    returns = np.zeros(episodes)
    lengths = np.zeros(episodes, dtype=int)
    falls = np.zeros(episodes, dtype=int)
    snapshots = {}
    checkpoint_set = set(checkpoints)
    for e in range(episodes):
        states, actions, rewards = _run_episode_reinforce(grid, policy, feature_fn, rng, max_steps)
        G = returns_from_rewards(rewards, grid.gamma)
        for t in range(len(states)):
            x = feature_fn(states[t])
            grad = policy.grad_log_prob(x, actions[t])
            policy.step(grad, lr * (grid.gamma ** t) * G[t])
        returns[e] = float(sum(rewards))
        lengths[e] = len(rewards)
        falls[e] = sum(1 for r in rewards if r == C.CLIFF_PENALTY)
        if (e + 1) in checkpoint_set:
            snap = SoftmaxPolicy(n_features, N_ACTIONS, seed)
            snap.theta = policy.theta.copy()
            snapshots[e + 1] = snap
    return policy, returns, lengths, falls, snapshots


def train_actor_critic(grid, feature_fn, n_features, lr_actor, lr_critic, episodes, seed, max_steps=C.MAX_STEPS_PER_EPISODE, checkpoints=()):
    """One-step Actor-Critic (Sutton & Barto 2018, Algorithmus 13.5): TD-Fehler des Kritikers als Vorteils-Schaetzung, Aktualisierung bei
    JEDEM Schritt statt erst am Episodenende."""
    rng = np.random.default_rng(seed)
    policy = SoftmaxPolicy(n_features, N_ACTIONS, seed)
    critic = LinearCritic(n_features, seed)
    returns = np.zeros(episodes)
    lengths = np.zeros(episodes, dtype=int)
    falls = np.zeros(episodes, dtype=int)
    snapshots = {}
    checkpoint_set = set(checkpoints)
    for e in range(episodes):
        s = grid.state_of(grid.start)
        discount = 1.0
        total_reward, n_steps, n_falls = 0.0, 0, 0
        for _ in range(max_steps):
            x = feature_fn(s)
            a = policy.sample(x, rng)
            s_next, r, done = step(grid, s, a, rng)
            x_next = feature_fn(s_next)
            v_next = 0.0 if done else critic.value(x_next)
            td_error = r + grid.gamma * v_next - critic.value(x)
            critic.td_step(x, td_error, lr_critic)
            grad = policy.grad_log_prob(x, a)
            policy.step(grad, lr_actor * discount * td_error)
            total_reward += r
            n_steps += 1
            if r == C.CLIFF_PENALTY:
                n_falls += 1
            discount *= grid.gamma
            s = s_next
            if done:
                break
        returns[e] = total_reward
        lengths[e] = n_steps
        falls[e] = n_falls
        if (e + 1) in checkpoint_set:
            snap = SoftmaxPolicy(n_features, N_ACTIONS, seed)
            snap.theta = policy.theta.copy()
            snapshots[e + 1] = snap
    return policy, critic, returns, lengths, falls, snapshots


def action_probs_table(policy, feature_fn, n_states):
    X = np.stack([feature_fn(s) for s in range(n_states)])
    return policy.probs(X)
