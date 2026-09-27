"""Rauchtest der Plotly-Abbildungen: explizite Keys, zwei Balkenreihen (eine je Methode)."""

import ac_evaluation as E
from ac_visualization import build_grid, build_learning_curve, build_method_comparison


def test_method_comparison_chart_has_one_trace_per_method():
    exp = E.seed_variance_experiment(seeds=range(4), base=E.Settings(episodes=50))
    fig = build_method_comparison(exp)
    assert len(fig.data) == 2
    assert {len(trace.y) for trace in fig.data} == {4}


def test_grid_and_curve_render_without_error():
    a = E.analyse(E.Settings(episodes=20, seed=0), method="actor_critic")
    fig1 = build_grid(a.grid, a.V_pi, a.action_probs)
    fig2 = build_learning_curve(a.returns)
    assert fig1 is not None and fig2 is not None
