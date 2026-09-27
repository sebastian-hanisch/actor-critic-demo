import numpy as np
import streamlit as st

import ac_constants as C
from ac_evaluation import METHOD_LABEL, Settings, _reference, analyse, seed_variance_experiment
from ac_presets import PRESET_HELP, PRESETS, apply_preset, bounds, init_session_state_defaults, load_permalink_settings, sync_query_params
from ac_visualization import build_grid, build_learning_curve, build_method_comparison

st.set_page_config(page_title="Actor-Critic", page_icon="🎭", layout="wide")

init_session_state_defaults()
load_permalink_settings()


def de(x, digits=2):
    return f"{x:.{digits}f}".replace(".", ",")


@st.cache_data(show_spinner=False)
def _analyse(settings, method, checkpoints):
    return analyse(settings, method=method, checkpoints=checkpoints)


@st.cache_data(show_spinner=False)
def _seed_variance_exp(base):
    return seed_variance_experiment(base=base)


st.title("🎭 Actor-Critic")
st.markdown(
    """
Letztes Stück der Reinforcement-Learning-Linie. Derselbe Lagerroboter, dieselbe Softmax-Politik wie bei **REINFORCE** (Stück 7) - aber statt auf
die volle Episoden-Rückgabe zu warten, lernt hier zusätzlich ein **Kritiker** eine Zustandswert-Schätzung $V(s)$ und liefert bei **jedem
einzelnen Schritt** einen TD-Fehler als Vorteils-Schätzung. Der Akteur (die Politik) nutzt diesen TD-Fehler statt der rohen Monte-Carlo-Rückgabe.
Alle Daten sind erzeugt.
"""
)
st.caption(
    "Achtes und letztes Stück der **Reinforcement-Learning-Linie** der \"Konzepte\"-Reihe. **Bezug:** direkte Fortsetzung von REINFORCE "
    "(Stück 7, identischer Akteur) - die Frage hier ist, ob ein gelernter Kritiker die Varianz senkt, ohne den Erwartungswert des "
    "Politikgradienten zu ändern."
)

with st.expander("So funktioniert Actor-Critic", expanded=True):
    st.markdown(
        r"""
1. **Der Akteur** ist dieselbe Softmax-Politik wie bei REINFORCE: $\pi(a|s) = \operatorname{softmax}(\theta^\top x(s))_a$.
2. **Der Kritiker** ist eine lineare Zustandswert-Schätzung $V(s) = w^\top x(s)$ - mit One-Hot-Merkmalen ist $w_s$ direkt der geschätzte Wert von Zustand $s$.
3. **Bei jedem Schritt** (nicht erst am Episodenende): TD-Fehler $\delta = r + \gamma V(s') - V(s)$, Kritiker-Update $w \leftarrow w + \alpha_w\,\delta\,x(s)$, Akteur-Update $\theta \leftarrow \theta + \alpha_\theta\,\gamma^t\,\delta\,\nabla_\theta\log\pi(a|s)$.
4. **Der Korrektheits-Check:** eine von der Aktion unabhängige Basislinie $b(s)$ (wie der Kritiker) ändert den *Erwartungswert* des Politikgradienten NICHT - nur seine Varianz. Das folgt aus $\mathbb{E}_{a\sim\pi}[\nabla_\theta\log\pi(a|s)] = 0$ und wird exakt (nicht nur stichprobenweise) nachgerechnet.
        """
    )

st.caption("🎯 Schnellstart – ein Beispiel laden:")
preset_names = list(PRESETS.keys())
for row in (preset_names[:3], preset_names[3:]):
    cols = st.columns(len(row))
    for col, name in zip(cols, row):
        with col:
            if st.button(name, key=f"preset_{name}", help=PRESET_HELP[name], width="stretch"):
                apply_preset(name)
st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.subheader("Die Methode")
    method = st.radio("Trainingsmethode", options=["actor_critic", "reinforce"], key="method_select",
                       format_func=lambda m: METHOD_LABEL[m])
    st.subheader("Das Raster")
    rows = st.slider("Zeilen", *bounds("rows_slider"), key="rows_slider")
    cols = st.slider("Spalten", *bounds("cols_slider"), key="cols_slider")
    slip = st.slider("Rutsch-Wahrscheinlichkeit", *bounds("slip_slider"), key="slip_slider", step=C.SLIP_STEP, format="%.2f")
    gamma = st.slider("Diskontfaktor γ", *bounds("gamma_slider"), key="gamma_slider", step=C.GAMMA_STEP, format="%.2f")
    st.subheader("Das Training")
    lr_actor = st.slider("Lernrate Akteur", *bounds("lr_actor_slider"), key="lr_actor_slider", step=C.ACTOR_LR_STEP, format="%.3f")
    lr_critic = st.slider("Lernrate Kritiker", *bounds("lr_critic_slider"), key="lr_critic_slider", step=C.CRITIC_LR_STEP, format="%.2f",
                           disabled=(method == "reinforce"), help="Nur für Actor-Critic - REINFORCE hat keinen Kritiker.")
    episodes = st.slider("Trainingsepisoden", *bounds("episodes_slider"), key="episodes_slider", step=C.EPISODES_STEP)
    seed = st.slider("Seed", *bounds("seed_slider"), key="seed_slider")

sync_query_params({
    "rows_slider": int(rows), "cols_slider": int(cols), "slip_slider": round(float(slip), 3), "gamma_slider": round(float(gamma), 3),
    "lr_actor_slider": round(float(lr_actor), 4), "lr_critic_slider": round(float(lr_critic), 4), "episodes_slider": int(episodes),
    "seed_slider": int(seed), "method_select": method,
})

settings = Settings(int(rows), int(cols), round(float(slip), 3), round(float(gamma), 3), round(float(lr_actor), 4),
                     round(float(lr_critic), 4), int(episodes), int(seed))
frames = tuple(sorted({0} | {int(round(x)) for x in np.linspace(1, settings.episodes, min(settings.episodes, 11))}))
with st.spinner(f"{METHOD_LABEL[method]} trainiert ..."):
    a = _analyse(settings, method, frames)
grid = a.grid
s0 = grid.state_of(grid.start)

# --- Episode für Episode -----------------------------------------------------------------------------------------------------------------------

st.markdown(f"## 🎯 Episode für Episode zur gelernten Politik ({METHOD_LABEL[method]})")
if "ac_frame_idx" not in st.session_state or st.session_state.get("ac_frame_owner") != (settings, method):
    st.session_state["ac_frame_idx"] = len(frames) - 1
    st.session_state["ac_frame_owner"] = (settings, method)
idx = st.slider("Trainingsstand", 0, len(frames) - 1, key="ac_frame_idx", help="0 = noch untrainiert (uniforme Zufallspolitik).")
ep = frames[idx]
from ac_agent import action_probs_table
from ac_features import one_hot
from ac_reference import policy_evaluation_stochastic
feature_fn = lambda s: one_hot(s, grid.n_states)
if ep == 0:
    probs_snap = np.full((grid.n_states, 4), 0.25)
elif ep == settings.episodes:
    probs_snap = a.action_probs
else:
    probs_snap = action_probs_table(a.snapshots[ep], feature_fn, grid.n_states)
_, P_ref, Rw_ref, *_ = _reference(settings.rows, settings.cols, settings.slip, settings.gamma)
V_snap = policy_evaluation_stochastic(P_ref, Rw_ref, probs_snap, settings.gamma)
head = "Vor dem Training (uniforme Zufallspolitik)" if ep == 0 else f"Nach {ep} von {settings.episodes} Episoden"
c1, c2 = st.columns([3, 2])
c1.markdown(f"**{head}**")
c1.plotly_chart(build_grid(grid, V_snap, probs_snap), width="stretch", key=f"ac_grid_{ep}")
c2.markdown("**Ertrag je Episode**")
c2.plotly_chart(build_learning_curve(a.returns[:ep] if ep > 0 else np.array([0.0])), width="stretch", key=f"ac_curve_{ep}")

st.markdown("---")

# --- Kernfrage ---------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Senkt der Kritiker die Varianz, ohne den Erwartungswert zu ändern?")
mcols = st.columns(3)
mcols[0].metric("Wert-Abstand zu V*(Start)", de(a.gap, 2))
mcols[1].metric("Umgebungsschritte", f"{a.env_steps:,}".replace(",", "."))
mcols[2].metric("Methode", METHOD_LABEL[method])
if a.gap < C.NEAR_OPTIMAL_GAP:
    st.success(f"✅ Bei Seed {settings.seed} lernt {METHOD_LABEL[method]} hier eine nahezu optimale Politik (Wert-Abstand {de(a.gap,2)}).")
elif a.gap > C.STUCK_GAP_THRESHOLD:
    st.error(f"🛑 Bei Seed {settings.seed} bleibt {METHOD_LABEL[method]} hier in einer schlechten, festgefahrenen Politik hängen (Wert-Abstand {de(a.gap,2)}). Probieren Sie dieselbe Einstellung mit der jeweils anderen Methode (Regler links).")
else:
    st.warning(f"⚠️ Bei Seed {settings.seed} lernt {METHOD_LABEL[method]} eine brauchbare, aber nicht optimale Politik (Wert-Abstand {de(a.gap,2)}).")

st.markdown("---")

# --- Experiment: Actor-Critic gegen REINFORCE ---------------------------------------------------------------------------------------------------

st.subheader("🔬 Actor-Critic gegen REINFORCE: wie oft bleibt welche Methode hängen?")
st.caption(
    f"Standardraster, {C.EXP_SEEDS} unabhängige Trainingsläufe je Methode, dieselben Seeds (Zeilen/Spalten/Rutschen/Episoden von den Reglern "
    "oben). Gezeigt: Wert-Abstand zu V*(Start) je Seed, je Methode aufsteigend sortiert. Dauer: gut zwei Minuten."
)
if st.button("Vergleich durchrechnen (gut zwei Minuten)", key="comparison_start"):
    st.session_state["comparison_on"] = True
if st.session_state.get("comparison_on"):
    base = Settings(int(rows), int(cols), round(float(slip), 3), round(float(gamma), 3), round(float(lr_actor), 4), round(float(lr_critic), 4), int(episodes))
    with st.spinner(f"Trainiert {2 * C.EXP_SEEDS} unabhängige Politiken (Actor-Critic UND REINFORCE) ..."):
        exp = _seed_variance_exp(base)
    st.plotly_chart(build_method_comparison(exp), width="stretch", key="comparison_chart")
    rows_by_method = {r["method"]: r for r in exp["rows"]}
    ac_row, re_row = rows_by_method["actor_critic"], rows_by_method["reinforce"]
    st.warning(
        f"**Befund:** von {len(exp['seeds'])} unabhängigen Trainingsläufen erreicht Actor-Critic in {de(100*ac_row['frac_near_optimal'],0)} % "
        f"eine nahezu optimale Politik und bleibt in {de(100*ac_row['frac_stuck'],0)} % dauerhaft hängen (Median {de(ac_row['median_gap'],2)}). "
        f"Reines REINFORCE erreicht nur {de(100*re_row['frac_near_optimal'],0)} % nahezu optimal und bleibt in {de(100*re_row['frac_stuck'],0)} % "
        f"hängen (Median {de(re_row['median_gap'],2)}) - bei DENSELBEN Seeds, DENSELBEN sonstigen Hyperparametern. Der Kritiker liefert bei "
        "jedem Schritt ein Lernsignal statt erst am Episodenende auf die oft extreme volle Rückgabe zu warten - das senkt die Varianz spürbar, "
        "ohne (laut der Basislinien-Identität oben) den Erwartungswert des Gradienten zu verzerren."
    )

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Der Kritiker lernt schnell genug** | Ein noch sehr ungenauer Kritiker liefert einen verzerrten TD-Fehler - anders als eine reine Basislinie (siehe Korrektheits-Check) führt Bootstrapping mit einem UNGENAUEN $V$ zu einer leicht verzerrten Schätzung, die sich aber mit dem Training auflöst. | Mehrere Kritiker-Updates je Aktualisierung, ein separates, langsameres Lernraten-Verhältnis |
| **Diskrete, kleine Zustands-/Aktionsräume** | Der Kritiker hier ist eine Tabelle (One-Hot); bei sehr großen oder stetigen Räumen bräuchte er selbst ein Netz mit Verallgemeinerung. | Deep-RL-Actor-Critic-Varianten (A3C, PPO, SAC) |
| **Ein einzelner, lernender Agent** | Bei mehreren gleichzeitig lernenden Agenten wird die Umgebung aus Sicht jedes Einzelnen nicht-stationär. | Multi-Agenten-Actor-Critic mit zentralem Kritiker (`mappo-demo`) |
| **Endlicher Diskontfaktor/endliche Episoden** | Diese Linie bleibt beim tabellarischen/kleinen-Netz-Fundament (Sutton & Barto, Kapitel 1-13), auf dem PPO, SAC, Rainbow und modellbasierte Verfahren (Dreamer, MuZero) aufbauen - keines davon wird hier nachgebaut. PPO ist mechanisch Actor-Critic plus ein geklemmtes Ersatzziel (schon in `mappo-demo` gebaut); SAC ist der Standard für stetige Aktionsräume (passt nicht zu vier diskreten Richtungen). | PPO, SAC, Rainbow-DQN, Dreamer/MuZero, GRPO/RLHF |
"""
)
st.caption("Die Linie: Bandit → Value Iteration und Policy Iteration → Q-Learning → SARSA → Dyna-Q → DQN / Funktionsapproximation → Policy Gradient / REINFORCE → **Actor-Critic** (dieses Stück, letztes der Linie).")

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Akteur:** $\pi_\theta(a|s) = \operatorname{softmax}(\theta^\top x(s))_a$ (identisch zu REINFORCE, Stück 7).

**Kritiker:** $V_w(s) = w^\top x(s)$, TD-Fehler $\delta_t = r_{t+1} + \gamma V_w(s_{t+1}) - V_w(s_t)$ (mit $V_w(\text{terminal})=0$).

**Kritiker-Update (semi-gradientes TD(0)):** $w \leftarrow w + \alpha_w\,\delta_t\,x(s_t)$.

**Akteur-Update (One-step Actor-Critic, Sutton & Barto 2018, Algorithmus 13.5):** $\theta \leftarrow \theta + \alpha_\theta\,\gamma^t\,\delta_t\,\nabla_\theta\log\pi_\theta(a_t|s_t)$.

**Basislinien-Identität (Korrektheits-Check):** für jede von der Aktion unabhängige Funktion $b(s)$ gilt $\mathbb{E}_{a\sim\pi}[b(s)\,\nabla_\theta\log\pi(a|s)] = b(s)\,\mathbb{E}_{a\sim\pi}[\nabla_\theta\log\pi(a|s)] = 0$, da $\sum_a \pi(a|s) = 1$ konstant ist und daher $\nabla_\theta \sum_a \pi(a|s) = \sum_a \nabla_\theta\pi(a|s) = 0$.
"""
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning. "
    "Interesse an einer maßgeschneiderten Lösung für Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
