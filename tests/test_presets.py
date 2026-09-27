"""Presets und Permalink-Werte: Vollständigkeit, gültige Werte, Grenzen und Schrittweiten - reine Datenprüfungen ohne Streamlit-Session."""

import ac_constants as C
import ac_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(P.PRESETS) == set(P.PRESET_HELP)
    for name, p in P.PRESETS.items():
        assert set(p) == set(P.PRESET_KEYS) and P.PRESET_HELP[name]


def test_preset_values_are_valid_and_on_the_slider_grid():
    for p in P.PRESETS.values():
        for key, state_key in P.PRESET_KEYS.items():
            spec = P.SETTING_SPECS[state_key]
            if isinstance(p[key], str):
                continue
            spec.caster(p[key])
            assert spec.lo <= p[key] <= spec.hi, (key, p[key])
        for key, state_key in P.PRESET_KEYS.items():
            if state_key not in P.STEPS:
                continue
            spec, step = P.SETTING_SPECS[state_key], P.STEPS[state_key]
            k = (p[key] - spec.lo) / step
            assert abs(k - round(k)) < 1e-6, (key, p[key])


def test_standard_preset_equals_the_default_settings():
    p = P.PRESETS["Standardfall (Actor-Critic)"]
    assert p["rows"] == C.DEFAULT_ROWS and p["cols"] == C.DEFAULT_COLS and p["lr_actor"] == C.DEFAULT_ACTOR_LR
    assert p["episodes"] == C.DEFAULT_EPISODES and p["seed"] == 0 and p["method"] == "actor_critic"


def test_same_seed_presets_differ_only_by_method():
    reinforce_bad = P.PRESETS["REINFORCEs unglücklicher Seed"]
    ac_same_seed = P.PRESETS["Derselbe Seed mit Actor-Critic"]
    assert reinforce_bad["seed"] == ac_same_seed["seed"] == 5
    assert reinforce_bad["method"] == "reinforce" and ac_same_seed["method"] == "actor_critic"


def test_bounds_steps_and_unique_url_params():
    assert P.bounds("slip_slider") == (C.SLIP_MIN, C.SLIP_MAX) and P.bounds("rows_slider") == (C.ROWS_MIN, C.ROWS_MAX)
    assert P.bounds("lr_actor_slider") == (C.ACTOR_LR_MIN, C.ACTOR_LR_MAX)
    assert len({spec.url_param for spec in P.SETTING_SPECS.values()}) == len(P.SETTING_SPECS)
