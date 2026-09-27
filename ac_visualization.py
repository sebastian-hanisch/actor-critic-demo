"""Plotly-Abbildungen der Demo "Actor-Critic". Achsen sind gesperrt (fixedrange)."""

import numpy as np
import plotly.graph_objects as go

import ac_constants as C
import ac_grid as G
from ac_evaluation import METHOD_LABEL

CLIFF_COLOR = "#3a3a3a"
START_COLOR = "#8c6bb1"
TEXT_LIGHT = "#ffffff"
TEXT_DARK = "#14233B"
METHOD_COLOR = {"actor_critic": "#2e7d32", "reinforce": "#d62728"}


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def de(x, digits=2):
    return f"{x:.{digits}f}".replace(".", ",")


def build_grid(grid, V, action_probs=None):
    R_, Cc = grid.rows, grid.cols
    z = np.full((R_, Cc), np.nan)
    text = [["" for _ in range(Cc)] for _ in range(R_)]
    for r in range(R_):
        for c in range(Cc):
            s = grid.state_of((r, c))
            kind = grid.cell_kind((r, c))
            z[r, c] = np.nan if kind == "cliff" else V[s]
            if kind == "goal":
                text[r][c] = "Ziel"
            elif kind == "cliff":
                text[r][c] = ""
            elif action_probs is not None:
                text[r][c] = G.ACTION_ARROWS[int(np.argmax(action_probs[s]))]
    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=z, colorscale="RdYlGn", zmid=0, showscale=True, text=[[de(v, 1) if not np.isnan(v) else "" for v in row] for row in z],
                              hovertemplate="Zeile %{y}, Spalte %{x}: V=%{z:.2f}<extra></extra>", colorbar=dict(title="V(s)", thickness=14)))
    cliff_x = [c for r in range(R_) for c in range(Cc) if grid.cell_kind((r, c)) == "cliff"]
    cliff_y = [r for r in range(R_) for c in range(Cc) if grid.cell_kind((r, c)) == "cliff"]
    if cliff_x:
        fig.add_trace(go.Scatter(x=cliff_x, y=cliff_y, mode="markers", marker=dict(symbol="square", size=18, color=CLIFF_COLOR), showlegend=False, hovertemplate="Klippe<extra></extra>"))
    for r in range(R_):
        for c in range(Cc):
            kind = grid.cell_kind((r, c))
            if text[r][c]:
                color = TEXT_LIGHT if kind == "goal" else TEXT_DARK
                fig.add_annotation(x=c, y=r, text=text[r][c], showarrow=False, font=dict(size=12, color=color))
    sr, sc = grid.start
    fig.add_shape(type="rect", x0=sc - 0.45, x1=sc + 0.45, y0=sr - 0.45, y1=sr + 0.45, line=dict(color=START_COLOR, width=3))
    fig.update_yaxes(autorange="reversed", showticklabels=False)
    fig.update_xaxes(showticklabels=False)
    return _base(fig, 45 * grid.rows + 60)


def build_learning_curve(returns, window=10):
    y = np.asarray(returns, dtype=float)
    x = np.arange(1, len(y) + 1)
    fig = go.Figure()
    if len(y) >= window:
        smooth = np.convolve(y, np.ones(window) / window, mode="valid")
        x_smooth = x[window - 1:]
    else:
        smooth, x_smooth = y, x
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", marker=dict(color="#c7d4e6", size=3), name="Ertrag je Episode"))
    fig.add_trace(go.Scatter(x=x_smooth, y=smooth, mode="lines", line=dict(color="#1f77b4", width=2), name=f"Gleitender Durchschnitt ({window})"))
    fig.update_xaxes(title_text="Episode")
    fig.update_yaxes(title_text="Ertrag")
    return _base(fig, 300).update_layout(legend=dict(orientation="h", y=-0.3))


def build_method_comparison(exp):
    fig = go.Figure()
    for row in exp["rows"]:
        gaps = np.sort(np.asarray(row["gaps"]))
        fig.add_trace(go.Bar(x=list(range(1, len(gaps) + 1)), y=gaps, marker=dict(color=METHOD_COLOR[row["method"]]),
                              name=METHOD_LABEL[row["method"]], hovertemplate="Rang %{x}: Wert-Abstand %{y:.2f}<extra></extra>"))
    fig.add_hline(y=C.NEAR_OPTIMAL_GAP, line=dict(color="#2e7d32", dash="dash", width=1))
    fig.add_hline(y=C.STUCK_GAP_THRESHOLD, line=dict(color="#d62728", dash="dash", width=1))
    fig.update_xaxes(title_text="Seeds, je Methode aufsteigend sortiert")
    fig.update_yaxes(title_text="Wert-Abstand zu V*(Start)")
    return _base(fig, 380).update_layout(legend=dict(orientation="h", y=-0.3), barmode="group")
