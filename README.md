# 🎭 Actor-Critic

Achtes und letztes Stück der **Reinforcement-Learning-Linie** der "Konzepte"-Reihe im Portfolio von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning. Direkte Fortsetzung von [Policy Gradient / REINFORCE](https://github.com/sebastian-hanisch/policy-gradient-demo) (Stück 7): derselbe Akteur (eine Softmax-Politik), aber zusätzlich ein gelernter **Kritiker** – eine Zustandswert-Schätzung $V(s)$, die bei jedem einzelnen Schritt einen TD-Fehler als Vorteils-Schätzung liefert, statt auf die volle Episoden-Rückgabe zu warten.

## Kernfrage

**Senkt der Kritiker die Varianz des Politikgradienten messbar, ohne seinen Erwartungswert zu ändern – und konvergiert Actor-Critic zuverlässiger als reines REINFORCE?** Eine von der Aktion unabhängige Basislinie $b(s)$ ändert den Erwartungswert des Gradienten nicht (Lehrbuch-Identität), nur seine Varianz.

## Modell

- **Vehikel:** dasselbe 3×4-Standardraster wie bei REINFORCE (Stück 7) – aus demselben Grund (episodisches Lernen braucht ein Raster, auf dem Episoden zuverlässig enden).
- **Der Akteur** (`ac_policy.py`): byte-gleich zu Stück 7 – eine lineare Softmax-Regression über One-Hot-Zustandsmerkmalen.
- **Der Kritiker** (`ac_critic.py`): eine lineare Zustandswert-Schätzung $V(s) = w^\top x(s)$ – mit One-Hot-Merkmalen ist $w_s$ direkt der geschätzte Wert von Zustand $s$.
- **One-step Actor-Critic** (`ac_agent.py`, Sutton & Barto 2018, Algorithmus 13.5): bei jedem Schritt TD-Fehler $\delta = r + \gamma V(s') - V(s)$, Kritiker-Update $w \leftarrow w + \alpha_w\delta\,x(s)$, Akteur-Update $\theta \leftarrow \theta + \alpha_\theta\gamma^t\delta\,\nabla_\theta\log\pi(a|s)$.
- **Korrektheits-Check:** die Basislinien-Identität $\mathbb{E}_{a\sim\pi}[b(s)\nabla_\theta\log\pi(a|s)] = 0$ für jede aktionsunabhängige Funktion $b(s)$ – exakt nachgerechnet durch Aufsummieren über alle Aktionen einer winzigen Politik (kein Sampling, keine Stichprobenabweichung).

## Der sauberste Befund der ganzen Linie

Anders als DQN (Stück 6, verrauschte Ablation) und REINFORCE selbst (Stück 7, ~17 % dauerhaft hängengebliebene Seeds) liefert dieser direkte Methodenvergleich ein klares, unmissverständliches Ergebnis: bei **denselben 30 Seeds, denselben Hyperparametern**, unterscheidet sich nur, ob der TD-Fehler eines gelernten Kritikers oder die volle Monte-Carlo-Rückgabe für das Politik-Update verwendet wird.

| | Actor-Critic | REINFORCE (Stück 7) |
|---|---|---|
| Nahezu optimal (Wert-Abstand < 1) | **96,7 %** | 16,7 % |
| Dauerhaft hängengeblieben (Wert-Abstand > 50) | **0,0 %** | 16,7 % |
| Median Wert-Abstand | **0,73** | 5,26 |
| Größter Wert-Abstand über alle Seeds | 1,00 | 942,52 |

Seed 5 – der Seed, der REINFORCE in Stück 7 dauerhaft in einer schlechten Politik hängen ließ (Wert-Abstand 114,4) – lernt unter Actor-Critic bei identischen sonstigen Einstellungen eine nahezu optimale Politik. Der Grund: der Kritiker liefert bei **jedem** Schritt ein Lernsignal, statt erst am Episodenende auf eine oft extreme, unbeschnittene Rückgabe zu warten (dieselbe Klippen-Strafe von −100, die REINFORCEs Politik gelegentlich sättigt, wird hier über viele kleine TD-Updates verteilt statt in einem einzigen riesigen Gradientenschritt).

## Befunde (gemessen, keine Behauptungen)

| Frage | Befund | Test |
|---|---|---|
| **Korrektheits-Check** | Für jede aktionsunabhängige Basislinie (getestet: 0, 5, −3,7, 100, der Erwartungswert der Rückgabe selbst) bleibt der erwartete Politikgradient exakt unverändert (Toleranz 1e-10). | `test_any_action_independent_baseline_leaves_the_expected_gradient_unchanged`, `test_the_mean_reward_itself_is_a_valid_unbiased_baseline` |
| **Seed 5: derselbe Seed, zwei Methoden** | REINFORCE: Wert-Abstand 114,4 (hängengeblieben). Actor-Critic: Wert-Abstand < 1 (nahezu optimal) – identischer Seed, identische sonstige Hyperparameter. | `test_seed5_reinforce_is_stuck_but_actor_critic_recovers` |
| **Methodenvergleich über 30 Seeds** | Actor-Critic: 96,7 % nahezu optimal, 0,0 % hängengeblieben. REINFORCE: 16,7 % / 16,7 %. | `test_method_comparison_matches_the_readme` |

## Ehrliche Grenzen

| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Der Kritiker lernt schnell genug** | Ein noch ungenauer Kritiker liefert einen leicht verzerrten TD-Fehler – anders als eine reine Basislinie führt Bootstrapping mit einem ungenauen $V$ zu einer Verzerrung, die sich mit dem Training auflöst. | Mehrere Kritiker-Updates je Aktualisierung, ein langsameres Lernraten-Verhältnis |
| **Diskrete, kleine Zustands-/Aktionsräume** | Der Kritiker ist hier eine Tabelle (One-Hot); bei sehr großen oder stetigen Räumen bräuchte er selbst Funktionsapproximation. | Deep-RL-Actor-Critic (A3C, PPO, SAC) |
| **Ein einzelner, lernender Agent** | Bei mehreren gleichzeitig lernenden Agenten wird die Umgebung aus Sicht jedes Einzelnen nicht-stationär. | Multi-Agenten-Actor-Critic mit zentralem Kritiker ([`mappo-demo`](https://github.com/sebastian-hanisch/mappo-demo)) |
| **Diese Linie bleibt beim Fundament stehen** | PPO ist mechanisch Actor-Critic plus ein geklemmtes Ersatzziel (schon in `mappo-demo` gebaut); SAC ist der Standard für stetige Aktionsräume (passt nicht zu vier diskreten Richtungen); Rainbow-DQN, modellbasierte Verfahren (Dreamer/MuZero) und RLHF/GRPO gehen alle über das hier gezeigte Fundament hinaus. | PPO, SAC, Rainbow-DQN, Dreamer/MuZero, GRPO/RLHF |

## Tests

`tests/` prüft das Vehikel, die Referenzlösung, den Akteur (inkl. des numerischen Gradienten-Checks, byte-gleich zu Stück 7), den Kritiker (TD-Update rührt nur den besuchten Zustand an, Konvergenz auf einer Kette mit geschlossener Lösung), die zentrale Basislinien-Identität (exakt, kein Sampling), beide Trainingsschleifen (REINFORCE und Actor-Critic, je ein struktureller Schnappschuss-Test), die Auswertung, die Presets/Permalinks, die Plotly-Achsen, die App (AppTest: jedes Preset, Trainingsstand-Slider, Permalink-Grenzen, Extremwerte, Experiment auf Abruf) und jede Zahl dieses READMEs (`test_claims.py`, mit der vollen Seed-Zahl – großzügige statt exakte Bänder wegen möglicher Windows/Linux-Gleitkomma-Unterschiede über 2000 Trainings-Episoden, siehe Stück 7). Gesamtlaufzeit einige Minuten statt der sonst üblichen knappen Minute (der Methodenvergleich braucht 60 vollständig unabhängige Trainings); die CI läuft bei jedem Push und wöchentlich.

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Oberfläche: Methodenwahl, Episode-für-Episode-Ansicht, Kernfrage, Methodenvergleich auf Abruf, Grenzen, Formeln |
| `ac_constants.py` | Regler-Grenzen, feste Rewards, Lernparameter für Akteur und Kritiker, Experiment-Konstanten |
| `ac_grid.py` | Das Vehikel: Raster, `step`, `build_model` (nur für die Referenz) |
| `ac_features.py` | One-Hot-Zustandsmerkmale |
| `ac_policy.py` | Der Akteur: Softmax-Politik (byte-gleich zu `policy-gradient-demo`) |
| `ac_critic.py` | Der Kritiker: lineare Zustandswert-Schätzung, TD(0)-Update |
| `ac_agent.py` | Zwei Trainingsschleifen: `train_reinforce` (Vergleichsmaßstab) und `train_actor_critic` |
| `ac_reference.py` | Value Iteration und (deterministische UND stochastische) Politik-Auswertung – nur zur Gegenprobe |
| `ac_evaluation.py` | Analyse, Methodenvergleichs-Experiment |
| `ac_visualization.py` | Plotly-Abbildungen |
| `ac_presets.py` | Presets, Permalink |
| `tests/` | Tests (siehe oben) |

## Bewusst nicht umgesetzt

- **PPO, SAC, Rainbow-DQN, modellbasierte Verfahren, RLHF/GRPO** – die Linie endet bewusst beim Ein-Agenten-Actor-Critic-Fundament, auf dem all diese Verfahren aufbauen (siehe Grenzen-Tabelle); PPOs Mechanik ist bereits in `mappo-demo` (Multi-Agenten-Koordination) gebaut.
- **Eligibility Traces** – der klassische nächste Schritt für Actor-Critic (TD(λ) statt TD(0)), aber ein eigenes Konzept jenseits dieses Stücks.
- **Ein Kritiker mit Funktionsapproximation** – bliebe bei diesem kleinen, endlichen Raster ohne Zusatznutzen (siehe `dqn-demo`, Stück 6, für die Funktionsapproximations-Story).

## Lokal ausführen

```bash
pip install -r requirements-dev.txt
streamlit run app.py
python -m pytest tests/ -q
```

Gebaut mit Streamlit, Plotly und numpy.
