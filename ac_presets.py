"""SETTING_SPECS-Permalink-Muster, Presets und Regler-Grenzen (Standardmuster des Portfolios)."""

import math
from dataclasses import dataclass
from typing import Callable, Optional

import streamlit as st

import ac_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: Callable
    default: object
    lo: Optional[float] = None
    hi: Optional[float] = None


def _method_caster(x):
    return x if x in ("actor_critic", "reinforce") else "actor_critic"


SETTING_SPECS = {
    "rows_slider": SettingSpec("rows", int, C.DEFAULT_ROWS, C.ROWS_MIN, C.ROWS_MAX),
    "cols_slider": SettingSpec("cols", int, C.DEFAULT_COLS, C.COLS_MIN, C.COLS_MAX),
    "slip_slider": SettingSpec("slip", float, C.DEFAULT_SLIP, C.SLIP_MIN, C.SLIP_MAX),
    "gamma_slider": SettingSpec("gamma", float, C.DEFAULT_GAMMA, C.GAMMA_MIN, C.GAMMA_MAX),
    "lr_actor_slider": SettingSpec("lr_actor", float, C.DEFAULT_ACTOR_LR, C.ACTOR_LR_MIN, C.ACTOR_LR_MAX),
    "lr_critic_slider": SettingSpec("lr_critic", float, C.DEFAULT_CRITIC_LR, C.CRITIC_LR_MIN, C.CRITIC_LR_MAX),
    "episodes_slider": SettingSpec("episodes", int, C.DEFAULT_EPISODES, C.EPISODES_MIN, C.EPISODES_MAX),
    "seed_slider": SettingSpec("seed", int, 0, 0, C.SEED_MAX),
    "method_select": SettingSpec("method", _method_caster, "actor_critic"),
}
PRESET_KEYS = {
    "rows": "rows_slider", "cols": "cols_slider", "slip": "slip_slider", "gamma": "gamma_slider",
    "lr_actor": "lr_actor_slider", "lr_critic": "lr_critic_slider", "episodes": "episodes_slider",
    "seed": "seed_slider", "method": "method_select",
}
STEPS = {
    "slip_slider": C.SLIP_STEP, "gamma_slider": C.GAMMA_STEP, "lr_actor_slider": C.ACTOR_LR_STEP,
    "lr_critic_slider": C.CRITIC_LR_STEP, "episodes_slider": C.EPISODES_STEP,
}


def _p(**kw):
    base = {
        "rows": C.DEFAULT_ROWS, "cols": C.DEFAULT_COLS, "slip": C.DEFAULT_SLIP, "gamma": C.DEFAULT_GAMMA,
        "lr_actor": C.DEFAULT_ACTOR_LR, "lr_critic": C.DEFAULT_CRITIC_LR, "episodes": C.DEFAULT_EPISODES,
        "seed": 0, "method": "actor_critic",
    }
    base.update(kw)
    return base


# Seed 5 ist derselbe Seed, der REINFORCE (policy-gradient-demo, Stück 7) bei identischen Hyperparametern dauerhaft in einer schlechten Politik
# haengen liess (Wert-Abstand 114,4) - hier zeigt derselbe Seed unter Actor-Critic UND als direkter REINFORCE-Vergleich denselben Bruch.
PRESETS = {
    "Standardfall (Actor-Critic)": _p(),
    "REINFORCE zum Vergleich": _p(method="reinforce"),
    "REINFORCEs unglücklicher Seed": _p(seed=5, method="reinforce"),
    "Derselbe Seed mit Actor-Critic": _p(seed=5, method="actor_critic"),
    "Kürzer trainiert": _p(episodes=500),
    "Mit Rutschen": _p(slip=0.20),
}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def load_permalink_settings():
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
                if isinstance(value, float) and not math.isfinite(value):
                    continue
                if spec.lo is not None and not isinstance(value, str):
                    value = max(spec.lo, min(spec.hi, value))
                st.session_state[state_key] = value
            except (ValueError, TypeError):
                pass
    for key, step in STEPS.items():
        if key in st.session_state:
            spec = SETTING_SPECS[key]
            snapped = spec.lo + round((st.session_state[key] - spec.lo) / step) * step
            snapped = min(spec.hi, max(spec.lo, snapped))
            st.session_state[key] = round(float(snapped), 4)
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        st.session_state[state_key] = PRESETS[name][key]


PRESET_HELP = {
    "Standardfall (Actor-Critic)": "Seed 0 mit Actor-Critic - der TD-Fehler eines gelernten Kritikers ersetzt die volle Monte-Carlo-Rückgabe.",
    "REINFORCE zum Vergleich": "Dieselben Hyperparameter, aber ohne Kritiker (reines REINFORCE aus Stück 7) - zum direkten Vergleich.",
    "REINFORCEs unglücklicher Seed": "Seed 5: bei reinem REINFORCE bleibt dieser Seed dauerhaft in einer schlechten Politik hängen (siehe policy-gradient-demo, Stück 7).",
    "Derselbe Seed mit Actor-Critic": "Derselbe Seed 5, aber mit Actor-Critic statt reinem REINFORCE - vergleichen Sie das Ergebnis mit dem Preset links.",
    "Kürzer trainiert": "500 statt 2000 Episoden.",
    "Mit Rutschen": "Rutsch-Wahrscheinlichkeit 0,20 statt 0,10.",
}
