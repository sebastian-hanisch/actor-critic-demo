"""Konstanten der Demo "Actor-Critic" (Stück 8, letztes Stück der Reinforcement-Learning-Linie). Dasselbe kleine 3x4-Raster wie bei
`policy-gradient-demo` (Stück 7) - REINFORCE lernt hier weiterhin episodisch/Monte-Carlo, der Kontrast ist NICHT das Vehikel, sondern ob eine
gelernte Zustandswert-Basislinie (der "Kritiker") die Varianz senkt."""

EPS = 1e-9
SEED_MAX = 999999

# --- Das Raster (wie bei policy-gradient-demo - Stück 7 hat gemessen, dass das groessere Geschwister-Raster fuer episodisches Lernen kaum
# terminiert; derselbe Grund gilt hier fuer den REINFORCE-Vergleichslauf) -------------------------------------------------------------------
STEP_COST = -1.0
CLIFF_PENALTY = -100.0
GOAL_REWARD = 10.0

ROWS_MIN, ROWS_MAX, DEFAULT_ROWS = 3, 6, 3
COLS_MIN, COLS_MAX, DEFAULT_COLS = 4, 12, 4
SLIP_MIN, SLIP_MAX, SLIP_STEP, DEFAULT_SLIP = 0.0, 0.30, 0.02, 0.10
GAMMA_MIN, GAMMA_MAX, GAMMA_STEP, DEFAULT_GAMMA = 0.80, 0.99, 0.01, 0.95

# --- Referenzloesung (Value Iteration, nur zur Gegenprobe) -------------------------------------------------------------------------------------
VI_TOL = 1e-8
VI_MAX_ITER = 5000

# --- Akteur (Softmax-Politik, One-Hot-Eingabe, wie in policy-gradient-demo) UND Kritiker (lineare Zustandswert-Schaetzung, mit One-Hot exakt
# eine Tabelle) -----------------------------------------------------------------------------------------------------------------------------
ACTOR_LR_MIN, ACTOR_LR_MAX, ACTOR_LR_STEP, DEFAULT_ACTOR_LR = 0.001, 0.05, 0.001, 0.005
CRITIC_LR_MIN, CRITIC_LR_MAX, CRITIC_LR_STEP, DEFAULT_CRITIC_LR = 0.01, 0.50, 0.01, 0.10
MAX_STEPS_PER_EPISODE = 300

EPISODES_MIN, EPISODES_MAX, EPISODES_STEP, DEFAULT_EPISODES = 100, 3000, 100, 2000

NEAR_OPTIMAL_GAP = 1.0
STUCK_GAP_THRESHOLD = 50.0

# --- Experiment: Streuung zwischen Zufalls-Seeds, REINFORCE gegen Actor-Critic ---------------------------------------------------------------
EXP_SEEDS = 20
EXP_SEEDS_FULL = 30
